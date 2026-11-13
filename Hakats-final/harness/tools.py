"""действия агента как инструменты openJiuwen (LocalFunction с ToolCard)"""
from __future__ import annotations

import asyncio
from pathlib import Path

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard

from .observe import Observation, run_tests
from .rules.code_repair import repair
from .rules.engines import run_lookup, run_stock, run_table
from .workspace import apply_edits, apply_rewrite, run_script

INT = {"type": "integer"}
STR = {"type": "string"}

STOCK_CARD = ToolCard(
    name="solve_stock",
    description=(
        "Exact step-by-step arithmetic on a stock (boxes in a warehouse and the like). Give the starting amount "
        "and one op per sentence, in order, copying numbers from the text (never compute). Ops: "
        "ship{per_day,days,strict} (strict = the text forbids shipping more than the stock); receive{count,each}; "
        "add{amount}; subtract{amount}; split{parts} (keep one share, rounded down); multiply{factor}; double; "
        "sell_percent{percent} (sold amount rounded down); if_above{threshold,then_add,else_add}; "
        "repeat_previous{times} (the previous op that many more times); add_days_times{factor} "
        "(days so far = days of ship ops); check_start{value} (the text restates the starting amount)."),
    input_params={"type": "object", "required": ["start", "ops"], "properties": {
        "start": INT,
        "ops": {"type": "array", "items": {"type": "object", "required": ["op"], "properties": {
            "op": {"type": "string", "enum": ["ship", "receive", "add", "subtract", "split", "multiply", "double",
                                              "sell_percent", "if_above", "repeat_previous", "add_days_times",
                                              "check_start"]},
            "per_day": INT, "days": INT, "strict": {"type": "boolean"}, "count": INT, "each": INT,
            "amount": INT, "parts": INT, "factor": INT, "percent": INT, "threshold": INT,
            "then_add": INT, "else_add": INT, "times": INT, "value": INT}}}}},
)

FILTER = {"type": "object", "required": ["column", "op", "value"], "properties": {
    "column": STR,
    "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "between", "in"]},
    "value": {"description": "a value, [low, high] for between, or a list for in"}}}

TABLE_CARD = ToolCard(
    name="solve_table",
    description=(
        "Exact query over CSV files: optional restrict/join, skip rows with blank columns, de-duplicate, filter, "
        "then aggregate. Every condition in the question must be a filter, even on a column the file may lack "
        "(the tool reports a missing column). Use only metrics the text defines; measure is an expression of "
        "columns with + - * / such as units*price or units*(price-cost)."),
    input_params={"type": "object", "required": ["file", "aggregate"], "properties": {
        "file": STR,
        "restrict": {"type": "object", "description": "keep rows whose key is listed in another file",
                     "properties": {"file": STR, "key": STR}},
        "join": {"type": "object", "description": "add the columns of another file matched on a column",
                 "properties": {"file": STR, "on": STR}},
        "skip_blank": {"type": "array", "items": STR},
        "dedupe": {"type": "string", "description": "count rows with the same value of this column once"},
        "filters": {"type": "array", "items": FILTER},
        "valid_range": {"type": "object", "description": "a range the text says values must lie in",
                        "properties": {"column": STR, "min": {"type": "number"}, "max": {"type": "number"}}},
        "measure": STR,
        "aggregate": {"type": "string", "enum": ["sum", "count", "mean", "min", "max", "median", "count_distinct",
                                                 "max_group", "min_group"]},
        "distinct": STR, "group_by": STR,
        "group_metric": {"type": "string", "enum": ["sum", "mean", "count"]},
        "round": INT, "lowercase": {"type": "boolean"}}},
)

CONDITION = {"type": "object", "required": ["field", "value"], "properties": {
    "path": {"type": "array", "items": STR}, "field": STR,
    "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<="]}, "value": {}}}

LOOKUP_CARD = ToolCard(
    name="solve_lookup",
    description=(
        "Exact questions over the markdown records (# Name, then Field: value lines). query=path: start at a "
        "record, follow link fields (path, e.g. [Mentor, Employer]), then read field (omit field for the name "
        "reached). query=value: the final field's value even if no record exists for it. query=chain_end: follow "
        "link until a record has none. query=count: number of records of a kind matching all where conditions. "
        "query=pick: among records reached via map from the matching ones, the one with the min/max of field by."),
    input_params={"type": "object", "required": ["query"], "properties": {
        "query": {"type": "string", "enum": ["path", "value", "chain_end", "count", "pick"]},
        "start": STR, "path": {"type": "array", "items": STR}, "field": STR, "link": STR,
        "kind": STR, "where": {"type": "array", "items": CONDITION},
        "map": {"type": "array", "items": STR}, "by": STR, "order": {"type": "string", "enum": ["min", "max"]}}},
)

IMPOSSIBLE_CARD = ToolCard(
    name="declare_impossible",
    description=(
        "The task cannot be answered exactly from what is given: an amount is not stated ('some boxes'), the "
        "question asks for a unit or metric the data does not define, the text contradicts itself, or a period "
        "lies outside the stated range. A filter that matches nothing is NOT impossible."),
    input_params={"type": "object", "required": ["reason"], "properties": {"reason": STR}},
)

SOLVER_CARDS = [STOCK_CARD, TABLE_CARD, LOOKUP_CARD, IMPOSSIBLE_CARD]


class TaskTools:
    """инструменты openJiuwen, привязанные к папке одной задачи"""

    def __init__(self, workdir: Path, obs: Observation):
        self.workdir = workdir
        self.obs = obs
        self.tools = {
            "solve_stock": LocalFunction(STOCK_CARD, lambda **spec: _verdict(run_stock(_given(spec)))),
            "solve_table": LocalFunction(TABLE_CARD, lambda **spec: _verdict(run_table(workdir, _given(spec)))),
            "solve_lookup": LocalFunction(LOOKUP_CARD, lambda **spec: _verdict(run_lookup(workdir, _given(spec)))),
            "declare_impossible": LocalFunction(IMPOSSIBLE_CARD,
                                                lambda reason: {"verdict": "impossible", "value": reason}),
            "run_python": LocalFunction(
                ToolCard(name="run_python", description="Run Python in the task directory",
                         input_params={"type": "object", "properties": {"code": STR, "language": STR}}),
                lambda code, language="py": {"output": run_script(code, workdir, language, obs.prelude)}),
            "apply_edits": LocalFunction(
                ToolCard(name="apply_edits", description="Apply SEARCH/REPLACE edit blocks",
                         input_params={"type": "object", "properties": {"reply": STR}}),
                lambda reply: {"notes": apply_edits(reply, workdir, obs)}),
            "rewrite_library": LocalFunction(
                ToolCard(name="rewrite_library", description="Put functions from a plain code block into the library file",
                         input_params={"type": "object", "properties": {"code": STR}}),
                lambda code: {"file": apply_rewrite(code, workdir, obs)}),
            "repair_by_mutation": LocalFunction(
                ToolCard(name="repair_by_mutation",
                         description="Fix small injected bugs by mutation search against the visible tests",
                         input_params={"type": "object", "properties": {"library": STR, "test_module": STR}}),
                lambda library, test_module: {"fixes": repair(workdir, library, test_module)}),
            "run_tests": LocalFunction(
                ToolCard(name="run_tests", description="Run the task's visible tests",
                         input_params={"type": "object", "properties": {}}),
                lambda: dict(zip(("passed", "output"), run_tests(obs.test_cmd, workdir)))),
        }

    def call(self, name: str, arguments: dict) -> dict:
        """вызов инструмента через openJiuwen"""
        tool = self.tools[name]
        return asyncio.run(tool.invoke(arguments, skip_inputs_validate=True))


def _given(node):
    """openJiuwen ставит None в поля, которых модель не дала, убираем их"""
    if isinstance(node, dict):
        return {k: _given(v) for k, v in node.items() if v is not None}
    if isinstance(node, list):
        return [_given(v) for v in node]
    return node


def _verdict(result: tuple[str, str]) -> dict:
    verdict, value = result
    return {"verdict": verdict, "value": value}
