"""решатели без токенов: ("answer", x), ("impossible", причина) или None, если что то не понято"""
from __future__ import annotations

from pathlib import Path

from .math_rules import solve_math
from .data_rules import solve_data
from .search_rules import solve_search

Verdict = tuple[str, str] | None


def solve_rules(task: dict, workdir: Path) -> Verdict:
    kind = (task.get("task_type") or "").lower()
    instruction = task.get("instruction", "")
    try:
        if kind == "math":
            v = solve_math(instruction)
        elif kind.startswith("data"):
            v = solve_data(instruction, workdir)
        elif kind == "search":
            v = solve_search(instruction, workdir)
        else:
            v = None
    except Exception:  # баг в решателе не должен ронять прогон
        return None
    if v is None or not v[1]:
        return None
    return v
