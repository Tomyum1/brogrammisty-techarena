"""вызовы модели через openJiuwen: учёт бюджета, повторы и запись в calls.jsonl"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timezone

import certifi

# загрузчик сертификатов openJiuwen хочет POSIX флаги, это для запуска на винде
for _flag in ("O_NOFOLLOW", "O_CLOEXEC"):
    if not hasattr(os, _flag):
        setattr(os, _flag, 0)
os.environ.setdefault("SAFE_CERT_DIR", os.path.dirname(certifi.where()))

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402

from .config import Settings  # noqa: E402
from .telemetry import CallLog  # noqa: E402

# контейнер живёт час, дольше ждать лимиты незачем
MIN_ACTIVE_CALLS = 2           # минимум одновременных запросов после 429
RATE_LIMIT_SHRINK_GAP = 20.0   # секунд между шагами вниз
DEADLINE = time.monotonic() + float(os.environ.get("RUN_TIME_LIMIT_S", "3300"))

MAX_RETRIES = 7

# для локальных прогонов через OpenRouter, провайдер фиксирован ради стабильности
OPENROUTER_MODELS = {"qwen3.6-35b-a3b": "qwen/qwen3.6-35b-a3b", "qwen3.8-27b": "qwen/qwen3.8-27b"}
OPENROUTER_PROVIDER = {"only": [os.environ.get("OPENROUTER_PROVIDER", "primeintellect/fp8")], "allow_fallbacks": False}


def provider_request(base_url: str, model: str) -> tuple[str, dict]:
    """имя модели и дополнительные поля запроса для прокси по адресу base_url"""
    if "openrouter.ai" not in base_url:
        return model, {}
    base = model.removesuffix("-noreason")
    body = {"provider": OPENROUTER_PROVIDER, "reasoning": {"enabled": not model.endswith("-noreason")}}
    return OPENROUTER_MODELS.get(base, base), {"extra_body": body}
# ключ не пустили или бюджет кончился
FATAL_ERROR = re.compile(r"Error code: (401|402|403)\b")
TRANSIENT_ERROR = re.compile(r"Error code: (408|409|425|429|5\d\d)\b|timed? ?out|Connection|connect|reset|EOF", re.I)
RATE_LIMIT_RESET = re.compile(r"resets at:?\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def quiet_sdk_logs() -> None:
    """openJiuwen пишет в logging каждый промпт, глушим"""
    logging.disable(logging.CRITICAL)


class BudgetExhausted(RuntimeError):
    """прокси отказывает насовсем: бюджет кончился, ключ не принят или он просто недоступен"""


class Budget:
    """расход и состояние прокси, общие для всех потоков"""

    def __init__(self, limit: float | None):
        self.limit = limit
        self.spent = 0.0
        self.dead = False          # прокси перестал отвечать
        self.paused_until = 0.0    # до этого момента все ждут из-за лимита
        self.lock = threading.Lock()
        # предел одновременных запросов, после каждого 429 убавляем на один
        self.max_active: int | None = None
        self.active = 0
        self.last_shrink = 0.0
        self.slot_freed = threading.Condition(self.lock)

    def add(self, units: float) -> None:
        with self.lock:
            self.spent += units

    def comfortable(self, share: float = 0.6) -> bool:
        """потрачено меньше share бюджета, при неизвестном бюджете всегда да"""
        return self.limit is None or self.spent < share * self.limit

    def remaining(self) -> float | None:
        return None if self.limit is None else self.limit - self.spent

    def enter_call(self) -> None:
        with self.slot_freed:
            while self.max_active is not None and self.active >= self.max_active:
                self.slot_freed.wait(timeout=5)
            self.active += 1

    def leave_call(self) -> None:
        with self.slot_freed:
            self.active -= 1
            self.slot_freed.notify_all()

    def shrink_after_rate_limit(self) -> None:
        """после 429 держим на один запрос меньше, но не меньше двух"""
        with self.lock:
            now = time.monotonic()
            if now - self.last_shrink < RATE_LIMIT_SHRINK_GAP:
                return
            self.last_shrink = now
            current = self.max_active if self.max_active is not None else max(self.active, MIN_ACTIVE_CALLS + 1)
            self.max_active = max(MIN_ACTIVE_CALLS, current - 1)

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.paused_until = max(self.paused_until, time.monotonic() + seconds)

    def wait_if_paused(self) -> None:
        while (left := min(self.paused_until, DEADLINE) - time.monotonic()) > 0:
            time.sleep(min(left, 30) + random.random())


class LLM:
    def __init__(self, settings: Settings, budget: Budget, task_id: str, calls: CallLog | None):
        self.settings = settings
        self.budget = budget
        self.task_id = task_id
        self.calls = calls
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0
        self.models: list[str] = []
        self._client_config = ModelClientConfig(
            client_provider="OpenAI",
            api_key=settings.api_key,
            api_base=settings.base_url,
            timeout=120,
            max_retries=0,  # повторы делаем сами ниже, там видно состояние бюджета
            verify_ssl=settings.base_url.startswith("https"),
            ssl_cert=certifi.where(),
        )
        self._clients: dict[str, Model] = {}

    @property
    def cost(self) -> float:
        return self.completion_tokens + self.prompt_tokens / 10

    def _client(self, model: str) -> Model:
        if model not in self._clients:
            self._clients[model] = Model(self._client_config,
                                         ModelRequestConfig(model=model, temperature=0, top_p=1))
        return self._clients[model]

    def chat(self, messages: list[dict], *, model: str | None = None, max_tokens: int = 800) -> str:
        reply = self.request(messages, model=model, max_tokens=max_tokens)
        return reply.content if isinstance(reply.content, str) else str(reply.content or "")

    def request(self, messages: list[dict], *, model: str | None = None, max_tokens: int = 800,
                tools: list | None = None):
        """один вызов модели"""
        model = model or self.settings.model
        context_chars = sum(len(m.get("content") or "") for m in messages)
        attempt = 0
        while True:
            if self.budget.dead:
                raise BudgetExhausted("proxy marked dead")
            self.budget.wait_if_paused()
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            prompt_tokens = completion_tokens = None
            status = 200
            self.budget.enter_call()
            try:
                api_model, extra = provider_request(self.settings.base_url, model)
                reply = asyncio.run(self._client(model).invoke(
                    messages, model=api_model, temperature=0, max_tokens=max_tokens, tools=tools, **extra))
                prompt_tokens, completion_tokens = self._count(reply, context_chars)
                self.models.append(model)
                return reply
            except Exception as exc:  # openJiuwen заворачивает ошибки провайдера в свои исключения
                text = str(exc)
                code = re.search(r"Error code: (\d{3})", text)
                status = int(code.group(1)) if code else None
                if status != 429:  # неудачный запрос тоже списывается
                    lost = context_chars / 3 / 10 + (max_tokens if re.search(r"timed? ?out", text, re.I) else 0)
                    self.budget.add(lost)
                if FATAL_ERROR.search(text) and "not available" not in text.lower():
                    self.budget.dead = True
                    raise BudgetExhausted(text[:300]) from exc
                if status == 429 and time.monotonic() + 15 < DEADLINE:
                    self.budget.shrink_after_rate_limit()
                    self._pause_for_rate_limit(text)
                    continue  # лимит не считаем неудачной попыткой
                if not TRANSIENT_ERROR.search(text):
                    raise
                if attempt == MAX_RETRIES:
                    self.budget.dead = True  # ошибки не кончаются, считаем прокси пропавшим
                    raise BudgetExhausted(text[:300]) from exc
                delay = min(2 * 2 ** attempt, 30) * (0.75 + 0.5 * random.random())
                attempt += 1
            finally:
                self.budget.leave_call()
                if self.calls is not None:
                    self.calls.record(
                        task_id=self.task_id, phase="solve", model=model, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens, reasoning_tokens=None, reasoning_in_completion=True,
                        n_messages=len(messages), context_chars=context_chars, tool_calls=[],
                        latency_s=round(time.monotonic() - started, 3), ts=started_at, http_status=status,
                        retry=attempt,
                    )
            time.sleep(delay)

    def _count(self, reply, context_chars: int) -> tuple[int, int]:
        usage = reply.usage_metadata
        if usage is None:  # нет usage, прикидываем по символам
            prompt = context_chars // 3
            completion = (len(reply.content or "") + len(reply.reasoning_content or "")) // 3
        else:
            prompt, completion = usage.input_tokens or 0, usage.output_tokens or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.n_calls += 1
        self.budget.add(completion + prompt / 10)
        return prompt, completion

    def _pause_for_rate_limit(self, error_text: str) -> None:
        """все ждут до сброса лимита, ранние запросы только отодвигают его"""
        wait = 10.0
        reset = RATE_LIMIT_RESET.search(error_text)
        if reset:
            reset_at = datetime.strptime(reset.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            wait = (reset_at - datetime.now(timezone.utc)).total_seconds() + 2
        self.budget.pause(max(5.0, min(wait, DEADLINE - time.monotonic())))
