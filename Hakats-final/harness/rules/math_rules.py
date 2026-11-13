"""складская арифметика (A warehouse starts with N boxes. ...)"""
from __future__ import annotations

import re

from .engines import run_stock

N = r"(\d[\d,]*)"


def _n(s: str) -> int:
    return int(s.replace(",", ""))


# (регулярка, операция): каждое предложение должно совпасть с одной из них целиком, регистр не важен
OPS = [
    (rf"(?:then )?it ships {N} boxes per day for {N} days?", "ship"),
    (rf"(?:then )?it ships {N} boxes per day for {N} days? \(it can never ship more than it has, "
     rf"and no partial shipments are allowed\)", "ship_strict"),
    (rf"(?:then )?it receives {N} deliver(?:y|ies) of {N} boxes each", "receive"),
    (rf"(?:then )?the stock is split equally into {N} warehouses? and it keeps one share \(rounded down\)", "split"),
    (r"(?:then )?the stock is doubled by merging with a sister warehouse", "double"),
    (rf"(?:then )?it sells {N}% of the stock \(rounded down\)", "sell"),
    (rf"(?:then )?, ?if the stock is above {N} boxes, it ships {N} boxes; otherwise it receives {N} boxes", "cond"),
    (rf"then, if the stock is above {N} boxes, it ships {N} boxes; otherwise it receives {N} boxes", "cond"),
    (rf"(?:then )?it repeats the previous step {N} more times?", "repeat"),
    (rf"(?:then )?it receives as many boxes as the number of days so far times {N}", "days"),
    (rf"recall that it started with {N} boxes", "recall"),
]
# размытые количества, сколько именно не узнать
VAGUE = re.compile(r"\b(some|several|a few|many|a number of|an unknown number of|multiple)\b.*\b(boxes|deliver)", re.I)

QUESTION_BOXES = re.compile(
    r"^(how many boxes does it have at the end|what is the (?:warehouse'?s )?final (?:box|stock) count"
    r"|how many boxes (?:does the warehouse have|are left|remain)(?: at the end)?)\??$", re.I)
QUESTION_OTHER_UNIT = re.compile(r"^(how many|what is the)\b.*\b(pallets?|crates?|containers?|trucks?|kg|tons?)\b", re.I)


def _sentences(text: str) -> list[str]:
    text = re.split(r"\s+(?:Compute it with|Work only inside|Use bash)", text)[0]
    parts = re.split(r"(?<=[.?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def solve_math(instruction: str):
    sents = _sentences(instruction)
    if not sents:
        return None
    m = re.fullmatch(rf"a warehouse starts with {N} boxes\.", sents[0], re.I)
    if not m:
        return None
    start = _n(m.group(1))
    question = sents[-1]
    body = sents[1:-1]
    if not question.endswith("?"):
        return None

    ops = []
    vague = None
    for s in body:
        s0 = s.rstrip(".")
        for rx, op in OPS:
            mm = re.fullmatch(rx, s0, re.I)
            if mm:
                ops.append((op, [_n(g) for g in mm.groups()]))
                break
        else:
            if VAGUE.search(s0) and re.search(r"\b(ships|receives|deliver)", s0, re.I):
                vague = s0
                ops.append(("vague", []))
                continue
            return None  # незнакомое предложение, отдаём LLM

    if QUESTION_OTHER_UNIT.match(question) and "box" not in question.lower():
        unit = QUESTION_OTHER_UNIT.match(question).group(2)
        return ("impossible", f"the question asks for {unit}, but no conversion from boxes is given")
    if not QUESTION_BOXES.match(question):
        return None
    if vague:
        return ("impossible", f"quantity not given: {vague}")

    return run_stock({"start": start, "ops": [_as_spec_op(op, args) for op, args in ops]})


def _as_spec_op(op: str, args: list[int]) -> dict:
    """разобранное предложение в виде операции для общего движка склада (harness/rules/engines.py)"""
    if op in ("ship", "ship_strict"):
        return {"op": "ship", "per_day": args[0], "days": args[1], "strict": op == "ship_strict"}
    if op == "receive":
        return {"op": "receive", "count": args[0], "each": args[1]}
    if op == "split":
        return {"op": "split", "parts": args[0]}
    if op == "double":
        return {"op": "double"}
    if op == "sell":
        return {"op": "sell_percent", "percent": args[0]}
    if op == "cond":
        return {"op": "if_above", "threshold": args[0], "then_add": -args[1], "else_add": args[2]}
    if op == "repeat":
        return {"op": "repeat_previous", "times": args[0]}
    if op == "days":
        return {"op": "add_days_times", "factor": args[0]}
    if op == "recall":
        return {"op": "check_start", "value": args[0]}
    raise ValueError(op)
