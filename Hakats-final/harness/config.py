"""связка с окружением: LLM_* раскладываем по именам JiuwenSwarm и openJiuwen и пишем .env на tmpfs"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_MODELS = (
    "qwen3.6-35b-a3b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b-noreason",
    "qwen3.8-27b",
)
DEFAULT_MODEL = "qwen3.8-27b-noreason"
DEFAULT_BUDGET = "100000"  # если BUDGET не передали


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    fallback_model: str | None  # последняя попытка, если основная модель не справилась
    data_dir: Path
    max_parallel: int
    token_budget: float | None  # в выходных токенах, None если не задан


def _clean(v: str | None) -> str:
    """срезаем кавычки, пробелы и случайные управляющие символы"""
    v = "".join(ch for ch in (v or "") if ch.isprintable())
    return v.strip().strip('"').strip("'").strip()


def load_settings() -> Settings:
    base_url = _clean(os.environ.get("LLM_BASE_URL") or os.environ.get("API_BASE"))
    api_key = _clean(os.environ.get("LLM_API_KEY") or os.environ.get("API_KEY"))
    if not base_url or not api_key:
        raise SystemExit("LLM_BASE_URL and LLM_API_KEY must be set")

    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        raise SystemExit(f"AGENT_MODEL={model!r} is not one of {ALLOWED_MODELS}")

    # одна модель на задачу, FALLBACK_MODEL включает модель с рассуждениями
    fallback = os.environ.get("FALLBACK_MODEL", "")
    if fallback and fallback not in ALLOWED_MODELS:
        raise SystemExit(f"FALLBACK_MODEL={fallback!r} is not one of {ALLOWED_MODELS}")

    home = os.environ.get("HOME", "/tmp")
    data_dir = Path(os.environ.get("JIUWENSWARM_DATA_DIR", os.path.join(home, "jiuwenswarm")))
    budget = _clean(os.environ.get("BUDGET") or os.environ.get("TOKEN_BUDGET")) or DEFAULT_BUDGET
    return Settings(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        fallback_model=fallback or None,
        data_dir=data_dir,
        max_parallel=int(os.environ.get("MAX_PARALLEL", "5")),
        token_budget=_budget(budget),
    )


def _budget(v: str) -> float:
    """бюджет из BUDGET, терпим пробелы, запятые и k, при мусоре берём значение по умолчанию"""
    text = v.replace(" ", "").replace("_", "").replace(",", "").lower()
    scale = 1000 if text.endswith("k") else 1
    try:
        value = float(text.removesuffix("k")) * scale
    except ValueError:
        value = float(DEFAULT_BUDGET)
    if value != value or value == float("inf"):  # nan и inf
        value = float(DEFAULT_BUDGET)
    return max(value, 0.0)


def apply_harness_env(s: Settings) -> None:
    os.environ["API_BASE"] = s.base_url
    os.environ["API_KEY"] = s.api_key
    os.environ["MODEL_NAME"] = s.model
    os.environ["MODEL_PROVIDER"] = "OpenAI"
    os.environ["JIUWENSWARM_DATA_DIR"] = str(s.data_dir)

    cfg = s.data_dir / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    env_file = cfg / ".env"
    env_file.write_text(
        f'API_BASE="{s.base_url}"\n'
        f'API_KEY="{s.api_key}"\n'
        f'MODEL_NAME="{s.model}"\n'
        "MODEL_PROVIDER=OpenAI\n",
        encoding="utf-8",
    )
    try:
        env_file.chmod(0o600)
    except OSError:
        pass
