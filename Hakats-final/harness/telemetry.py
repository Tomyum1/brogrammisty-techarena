"""пишем учёт каждой попытки в одном порядке из всех потоков"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path


class CallLog:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.lock = threading.Lock()
        self.counts: dict[str | None, int] = defaultdict(int)
        self.task_units: dict[str | None, int] = defaultdict(int)
        self.run_units = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8", newline="\n")

    def record(self, *, task_id: str | None, phase: str, model: str,
               prompt_tokens: int | None, completion_tokens: int | None,
               reasoning_tokens: int | None, reasoning_in_completion: bool,
               n_messages: int, context_chars: int, tool_calls: list[dict],
               latency_s: float, ts: str, http_status: int | None, retry: int) -> None:
        known = prompt_tokens is not None or completion_tokens is not None or reasoning_tokens is not None
        separate = 0 if reasoning_in_completion else reasoning_tokens or 0
        units = (prompt_tokens or 0) + 10 * ((completion_tokens or 0) + separate)
        with self.lock:
            self.counts[task_id] += 1
            self.task_units[task_id] += units
            self.run_units += units
            row = {
                "run_id": self.run_id, "task_id": task_id, "phase": phase,
                "call_idx": self.counts[task_id], "model": model, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "reasoning_tokens": reasoning_tokens,
                "reasoning_in_completion": reasoning_in_completion,
                "cost": units / 10 if known else None,
                "cum_cost_task": self.task_units[task_id] / 10, "cum_cost_run": self.run_units / 10,
                "n_messages": n_messages, "context_chars": context_chars, "tool_calls": tool_calls,
                "latency_s": latency_s, "ts": ts, "http_status": http_status, "retry": retry,
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
