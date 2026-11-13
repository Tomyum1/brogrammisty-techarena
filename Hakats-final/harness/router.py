"""один короткий вызов: модель переводит задачу в JSON спеку для точного решателя"""
from __future__ import annotations

import json
import os
import re

from .llm import LLM
from .observe import Observation
from .rules.engines import SpecError
from .tools import TaskTools

ROUTER_PROMPT = """Translate the task into a JSON spec for an exact solver. Reply with ONE compact single-line JSON object, no prose:
{"solver": "stock" | "table" | "lookup" | "none", "impossible": null or "<reason>", "spec": {...}}

Set "impossible" (and spec null) only when the task cannot be answered exactly from what is given: an amount is not stated ("some boxes"), the question asks for a unit or metric the data does not define (pallets, profit without cost data), the text contradicts itself (a second, different starting value) or asks for a period outside the stated range. A filter that matches nothing is NOT impossible.

stock - quantity that changes step by step. spec: {"start": int, "ops": [...]} with ops in order:
 {"op":"ship","per_day":int,"days":int,"strict":bool}  (strict: the text forbids shipping more than the stock)
 {"op":"receive","count":int,"each":int} {"op":"add","amount":int} {"op":"subtract","amount":int}
 {"op":"split","parts":int} (keep one share, rounded down) {"op":"multiply","factor":int} {"op":"double"}
 {"op":"sell_percent","percent":int} (rounded down) {"op":"if_above","threshold":int,"then_add":int,"else_add":int}
 {"op":"repeat_previous","times":int} {"op":"add_days_times","factor":int} (days so far = days of ship ops)
 {"op":"check_start","value":int} (the text restates the starting amount)
table - question over CSV files. spec: {"file":"sales.csv","restrict":{"file":..,"key":..}|null,"join":{"file":..,"on":..}|null,
 "skip_blank":[cols],"dedupe":"col"|null,"filters":[{"column":..,"op":"=="|"!="|">"|">="|"<"|"<="|"between"|"in","value":..}],
 "valid_range":{"column":..,"min":..,"max":..}|null,"measure":"units"|"units*price"|"<expression of columns>"|null,
 "aggregate":"sum"|"count"|"mean"|"min"|"max"|"median"|"count_distinct"|"max_group"|"min_group",
 "distinct":col|null,"group_by":col|null,"group_metric":"sum"|"mean"|"count","round":int|null,"lowercase":bool}
 Use only metrics the text defines; reference a column even if the file lacks it (the solver reports that).
lookup - question over markdown records. spec is one of:
 {"query":"path","start":name,"path":[link fields],"field":field|null}   e.g. HQ of the employer of X's mentor:
   {"query":"path","start":"X","path":["Mentor","Employer"],"field":"Headquarters"}; field null = the name reached
 {"query":"value","start":name,"path":[...],"field":field}   a field's value even if it has no record of its own
 {"query":"chain_end","start":name,"link":"Mentor"}
 {"query":"count","kind":"person","where":[{"path":[..],"field":..,"op":..,"value":..}]}
 {"query":"pick","kind":"person","where":[..],"map":[link fields],"by":field,"order":"min"|"max"}
none - anything else: code fixing, or any part of the question these specs cannot express exactly (consecutive or
 windowed rows, ranks, time ordering, sequences, rules from documents, several steps that depend on each other).
 When in doubt, choose none: a wrong spec gives a confidently wrong answer.
Rules: copy numbers and names exactly as written in the task (an alias like "Member 1234-05" stays as is), never compute anything yourself (4 trucks of 80 boxes is
{"op":"receive","count":4,"each":80}). One op per sentence, in order. Every condition in the question becomes a
filter or where-clause, even on a column or field the data may not have. In lookup, "path" holds only link fields
(like Mentor, Employer) and "field" is the attribute to read."""

PRELUDE_NOTE = re.compile(r"^YOUR SCRIPT RUNS AFTER A PRELUDE.*?(?=\n\n|\Z)", re.S)
FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
# по умолчанию выключен, включается ROUTED_KINDS=math,search
ROUTED_KINDS = set(filter(None, os.environ.get("ROUTED_KINDS", "").split(",")))
TOOL_FOR_SOLVER = {"stock": "solve_stock", "table": "solve_table", "lookup": "solve_lookup"}


def route(task: dict, obs: Observation, tools: TaskTools, llm: LLM) -> tuple[str, str, dict] | None:
    """возвращает (вердикт, значение, детали), или None если надо откатиться на другой путь"""
    # на data скрипт надёжнее табличного движка
    if obs.kind not in ROUTED_KINDS or not data_fits_engines(tools.workdir):
        return None
    task_text = task.get("instruction", "").strip()
    messages = [{"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"TASK:\n{task_text}\n\n{_context(obs)}"}]
    reply = llm.chat(messages, max_tokens=900)
    details = {"router_reply": reply}
    parsed = _parse_json(reply)
    if not isinstance(parsed, dict):
        return None
    solver, reason, spec = parsed.get("solver"), parsed.get("impossible"), parsed.get("spec")
    if reason and solver != "none":
        output = tools.call("declare_impossible", {"reason": str(reason)[:200]})
        return (output["verdict"], output["value"], details)
    if solver not in TOOL_FOR_SOLVER or not isinstance(spec, dict):
        return None
    try:
        output = tools.call(TOOL_FOR_SOLVER[solver], spec)
    except (SpecError, KeyError, TypeError, ValueError) as exc:
        details["tool_error"] = str(exc)[:200]
        return None
    except Exception as exc:  # openJiuwen заворачивает ошибки инструментов в свои исключения
        details["tool_error"] = str(exc)[:200]
        return None
    details["solver"] = TOOL_FOR_SOLVER[solver]
    return (output["verdict"], str(output["value"])[:300], details)


def _context(obs: Observation) -> str:
    """описание папки задачи без подсказок, которые нужны только для написания скрипта"""
    text = PRELUDE_NOTE.sub("", obs.text).strip()
    return f"WORKSPACE:\n{text}" if text else "WORKSPACE: (empty)"


def _parse_json(reply: str):
    text = reply.strip()
    fenced = FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


RULE_FILE = re.compile(r"(policy|rule|readme|instruction|note|glossary)", re.I)


def data_fits_engines(workdir) -> bool:
    """маршрутизируем, только если файлы понятны движкам и нет документов с правилами"""
    md_files = list(workdir.rglob("*.md")) if workdir.is_dir() else []
    if any(RULE_FILE.search(p.stem) for p in md_files):
        return False
    for path in md_files:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        headers = [line for line in lines if line.startswith("#")]
        fields = [line for line in lines if re.match(r"^[A-Z][\w ]*:\s", line)]
        if len(headers) != 1 or not fields:
            return False
    return True
