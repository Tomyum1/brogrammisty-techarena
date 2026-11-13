"""точные исполнители спек: ("answer", x) или ("impossible", причина), кривая спека кидает SpecError"""
from __future__ import annotations

import ast
import csv
import operator
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


class SpecError(ValueError):
    """спеку нельзя выполнить как написано: незнакомая операция, нет параметра, плохое выражение"""


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise SpecError(f"{name} must be an integer, got {value!r}")
    return int(value)


# арифметика склада

STOCK_OPS = ("ship", "receive", "add", "subtract", "split", "multiply", "double", "sell_percent",
             "if_above", "repeat_previous", "add_days_times", "check_start")


def run_stock(spec: dict):
    """складская арифметика по спеке {start, ops}"""
    start = _int(spec.get("start"), "start")
    stock, days, previous = start, 0, None
    for raw in spec.get("ops") or []:
        op = dict(raw)
        name = op.get("op")
        if name not in STOCK_OPS:
            raise SpecError(f"unknown stock op {name!r}")
        repeats = 1
        if name == "repeat_previous":
            if previous is None:
                raise SpecError("repeat_previous with nothing before it")
            repeats, op, name = _int(op.get("times"), "times"), previous, previous["op"]
        for _ in range(repeats):
            if name == "ship":
                amount = _int(op.get("per_day"), "per_day") * _int(op.get("days"), "days")
                if op.get("strict") and amount > stock:
                    return ("impossible", "the shipments exceed the stock, which the task forbids")
                stock -= amount
                days += _int(op.get("days"), "days")
            elif name == "receive":
                stock += _int(op.get("count"), "count") * _int(op.get("each"), "each")
            elif name == "add":
                stock += _int(op.get("amount"), "amount")
            elif name == "subtract":
                stock -= _int(op.get("amount"), "amount")
            elif name == "split":
                parts = _int(op.get("parts"), "parts")
                if parts == 0:
                    return ("impossible", "split into 0 parts (division by zero)")
                stock //= parts
            elif name == "multiply":
                stock *= _int(op.get("factor"), "factor")
            elif name == "double":
                stock *= 2
            elif name == "sell_percent":
                stock -= stock * _int(op.get("percent"), "percent") // 100
            elif name == "if_above":
                above = stock > _int(op.get("threshold"), "threshold")
                stock += _int(op.get("then_add" if above else "else_add"), "then_add/else_add")
            elif name == "add_days_times":
                stock += days * _int(op.get("factor"), "factor")
            elif name == "check_start":
                if _int(op.get("value"), "value") != start:
                    return ("impossible", f"contradiction: starts with {start}, later restated as {op['value']}")
        if name != "check_start":
            previous = op
    return ("answer", str(stock))


# запросы к CSV

_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


def _compile_measure(expression: str):
    """выражение по строке вроде units*(price-cost): только имена колонок, числа и + - * /"""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SpecError(f"measure {expression!r} may only use columns, numbers and + - * /")
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    code = compile(tree, "<measure>", "eval")
    return names, (lambda row: eval(code, {"__builtins__": {}}, row))  # noqa: S307 - validated above


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()  # разделитель берём тот, которого в заголовке больше всего, при равенстве запятая
        handle.seek(0)
        return list(csv.DictReader(handle, delimiter=max(',;\t|', key=first.count)))


def _to_number(text):
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


TABLE_AGGREGATES = ("sum", "count", "mean", "min", "max", "median", "count_distinct", "max_group", "min_group")

COMPARE = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
           "<": operator.lt, "<=": operator.le}


def run_table(workdir: Path, spec: dict):
    """фильтры, джойн и агрегаты по CSV файлу"""
    path = workdir / str(spec.get("file") or "")
    if not spec.get("file") or not path.is_file():
        return ("impossible", f"{spec.get('file')} does not exist")
    rows = _read_csv(path)
    columns = set(rows[0].keys()) if rows else set()

    restrict = spec.get("restrict")
    if restrict:
        other = workdir / restrict["file"]
        if not other.is_file():
            return ("impossible", f"{restrict['file']} does not exist")
        allowed = {r.get(restrict["key"]) for r in _read_csv(other)}
        rows = [r for r in rows if r.get(restrict["key"]) in allowed]
    join = spec.get("join")
    if join:
        other = workdir / join["file"]
        if not other.is_file():
            return ("impossible", f"{join['file']} does not exist")
        table = {r[join["on"]]: r for r in _read_csv(other)}
        joined = []
        for r in rows:
            match = table.get(r.get(join["on"]))
            joined.append({**r, **{k: v for k, v in (match or {}).items() if k != join["on"]}})
        rows = joined
        columns |= set(next(iter(table.values()), {}).keys())

    measure = spec.get("measure")
    aggregate = spec.get("aggregate")
    if aggregate == "count":
        measure = None  # для count measure не нужен
    if spec.get("group_by") and aggregate in ("sum", "mean", "count"):
        # у суммы по группам нет одного ответа, спрашивают о лидере
        spec = {**spec, "group_metric": spec.get("group_metric") or aggregate}
        aggregate = "max_group"
    if aggregate not in TABLE_AGGREGATES:
        raise SpecError(f"unknown aggregate {aggregate!r}")
    needed = set()
    evaluate = None
    if measure:
        needed, evaluate = _compile_measure(str(measure))
        needed = set(needed)
    filters = spec.get("filters") or []
    needed |= {f.get("column") for f in filters}
    if spec.get("group_by"):
        needed.add(spec["group_by"])
    missing = sorted(c for c in needed if c not in columns)
    if missing:
        return ("impossible", f"the data has no column {', '.join(repr(c) for c in missing)}")

    limits = spec.get("valid_range")
    if limits:
        for f in filters:
            if f.get("column") == limits.get("column"):
                values = f["value"] if isinstance(f["value"], list) else [f["value"]]
                for v in values:
                    if not limits["min"] <= v <= limits["max"]:
                        return ("impossible", f"{v} is outside the stated range {limits['min']}-{limits['max']}")

    records = []
    for raw in rows:
        record = {k: _to_number(v) for k, v in raw.items()}
        if any(record.get(c) is None for c in spec.get("skip_blank") or []):
            continue
        records.append(record)
    if spec.get("dedupe"):
        records = list({r.get(spec["dedupe"]): r for r in records}.values())

    for f in filters:
        column, op, value = f.get("column"), f.get("op", "=="), f.get("value")
        if op == "between":
            low, high = value
            records = [r for r in records if r[column] is not None and low <= r[column] <= high]
        elif op == "in":
            records = [r for r in records if r[column] in value]
        elif op in COMPARE:
            records = [r for r in records if r[column] is not None and COMPARE[op](r[column], value)]
        else:
            raise SpecError(f"unknown filter op {op!r}")

    def value_of(record):
        result = evaluate(record) if evaluate else 1
        if result is None:
            raise SpecError("measure on a blank value")
        return result

    try:
        if aggregate == "count":
            return ("answer", str(len(records)))
        if aggregate == "sum":
            return ("answer", _plain_number(sum(value_of(r) for r in records), spec.get("round")))
        if aggregate in ("mean", "min", "max", "median"):
            if not records:
                return ("impossible", f"no rows match, the {aggregate} is undefined")
            values = sorted(Decimal(str(value_of(r))) for r in records)
            if aggregate == "mean":
                result = sum(values) / len(values)
            elif aggregate == "median":
                middle = len(values) // 2
                result = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
            else:
                result = values[0] if aggregate == "min" else values[-1]
            return ("answer", _plain_number(result, spec.get("round")))
        if aggregate == "count_distinct":
            column = spec.get("distinct")
            if column not in columns:
                return ("impossible", f"the data has no column {column!r}")
            return ("answer", str(len({r[column] for r in records})))

        group = spec.get("group_by")
        if not group:
            raise SpecError("max_group/min_group need group_by")
        metric = spec.get("group_metric", "sum")
        sums, counts = defaultdict(float), defaultdict(int)
        for r in records:
            sums[r[group]] += value_of(r)
            counts[r[group]] += 1
        if not sums:
            return ("impossible", "no rows match")
        score = {"sum": lambda g: sums[g], "mean": lambda g: sums[g] / counts[g],
                 "count": lambda g: counts[g]}.get(metric)
        if score is None:
            raise SpecError(f"unknown group_metric {metric!r}")
        ranked = sorted(((g, score(g)) for g in sums), key=lambda kv: kv[1], reverse=aggregate == "max_group")
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            raise SpecError("tie for the top group")
        name = str(ranked[0][0])
        return ("answer", name.lower() if spec.get("lowercase") else name)
    except TypeError as exc:
        raise SpecError(f"non-numeric value in the measure: {exc}") from exc


def _plain_number(value, digits) -> str:
    if digits is not None:
        return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-int(digits)), rounding=ROUND_HALF_UP))
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return str(value)


# база знаний

class KnowledgeBase:
    """записи markdown: # Title, потом строки Field: value, псевдонимы ведут на свою запись"""

    def __init__(self, workdir: Path):
        self.records: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}
        folder = workdir / "kb"
        files = sorted(folder.glob("*.md")) if folder.is_dir() else sorted(workdir.rglob("*.md"))
        for path in files:
            title, fields = None, {}
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if not title or title in self.records:
                raise SpecError("unexpected knowledge base layout")
            fields["_kind"] = path.stem.split("_")[0]
            self.records[title] = fields
        for title, fields in self.records.items():
            for key, value in fields.items():
                if "known as" in key.lower() or key.lower().startswith("alias"):
                    for alias in value.split(","):
                        if alias.strip() and alias.strip() not in self.records:
                            self.aliases[alias.strip()] = title
        if not self.records:
            raise SpecError("empty knowledge base")
        # поля со ссылками на другие записи
        resolves, filled = defaultdict(int), defaultdict(int)
        for fields in self.records.values():
            for key, value in fields.items():
                if value and not key.startswith("_"):
                    filled[key] += 1
                    resolves[key] += self.title(value) is not None
        self.link_fields = {k for k in filled if resolves[k] * 2 >= filled[k]}

    def normalize(self, path: list[str], field: str | None) -> tuple[list[str], str | None]:
        """если поле попало в конец пути, переносим его из пути в field"""
        links = []
        for step in path or []:
            if step in self.link_fields:
                links.append(step)
            elif field is None or step == field:
                field = step
        if links and field == links[-1]:  # путь [Employer] и поле Employer: его надо прочитать, а не идти по нему
            links.pop()
        if field in NAME_FIELDS:  # Name значит сама запись
            field = None
        return links, field

    def title(self, name: str) -> str | None:
        name = (name or "").strip()
        if name in self.records:
            return name
        return self.aliases.get(name)

    def follow(self, start: str, path: list[str]) -> str:
        """запись, до которой доходим от start по ссылкам, на разрыве кидаем Impossible"""
        current = self.title(start)
        if current is None:
            raise Impossible(f"no record for {start}")
        for link in path:
            value = self.records[current].get(link, "")
            if not value:
                raise Impossible(f"{current} has no {link}")
            target = self.title(value)
            if target is None:
                raise Impossible(f"{link} of {current} is {value}, but there is no record for it")
            current = target
        return current

    def field(self, title: str, name: str) -> str:
        value = self.records[title].get(name, "")
        if not value:
            raise Impossible(f"{title} has no {name}")
        return value


NAME_FIELDS = {"Name", "name", "Title", "title", "_title"}


class Impossible(Exception):
    pass


def _matches(kb: KnowledgeBase, title: str, condition: dict) -> bool:
    path, field = kb.normalize(condition.get("path") or [], condition.get("field"))
    try:
        target = kb.follow(title, path)
    except Impossible:
        return False
    value = kb.records[target].get(field, "") if field else target
    wanted = condition.get("value")
    if isinstance(wanted, str) and isinstance(value, str) and (field is None or field in kb.link_fields):
        # имена сравниваем по записи, тогда и псевдоним из вопроса совпадёт
        value, wanted = kb.title(value) or value, kb.title(wanted) or wanted
    if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
        try:
            value = float(value)
        except ValueError:
            return False
    op = condition.get("op", "==")
    if op not in COMPARE:
        raise SpecError(f"unknown op {op!r}")
    return COMPARE[op](value, wanted)


def run_lookup(workdir: Path, spec: dict):
    """вопросы к базе знаний: path, value, chain_end, count и pick"""
    kb = KnowledgeBase(workdir)
    query = spec.get("query")
    try:
        if query in ("path", "value"):
            path, field = kb.normalize(list(spec.get("path") or []), spec.get("field"))
            spec = {**spec, "field": field}
            if query == "value":
                base = kb.follow(spec.get("start", ""), path)
                return ("answer", kb.field(base, spec["field"]))
            end = kb.follow(spec.get("start", ""), path)
            return ("answer", kb.field(end, spec["field"]) if spec.get("field") else end)
        if query == "chain_end":
            link = spec.get("link", "Mentor")
            current = kb.follow(spec.get("start", ""), [])
            seen = {current}
            while kb.records[current].get(link):
                current = kb.follow(current, [link])
                if current in seen:
                    raise Impossible(f"the {link} chain loops")
                seen.add(current)
            return ("answer", current)
        if query in ("count", "pick"):
            kind = spec.get("kind")
            where = spec.get("where") or []
            def select(kind_filter):
                return [t for t, f in kb.records.items()
                        if (not kind_filter or f["_kind"] == kind_filter) and all(_matches(kb, t, c) for c in where)]
            chosen = select(kind)
            if not chosen and kind:
                chosen = select(None)  # модель иногда пишет тип ответа, а не тип записей
            if query == "count":
                return ("answer", str(len(chosen)))
            if not chosen:
                raise Impossible("no records match")
            reached = {kb.follow(t, list(spec.get("map") or [])) for t in chosen}
            scored = {t: float(kb.field(t, spec["by"])) for t in reached}
            best = (min if spec.get("order", "min") == "min" else max)(scored.values())
            top = [t for t, v in scored.items() if v == best]
            if len(top) != 1:
                raise SpecError("tie")
            return ("answer", top[0])
    except Impossible as exc:
        return ("impossible", str(exc))
    raise SpecError(f"unknown query {query!r}")


def run_spec(kind: str, workdir: Path, spec: dict):
    if kind == "stock":
        return run_stock(spec)
    if kind == "table":
        return run_table(workdir, spec)
    if kind == "lookup":
        return run_lookup(workdir, spec)
    raise SpecError(f"unknown solver {kind!r}")

