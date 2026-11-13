"""агрегаты по CSV с продажами (The file sales.csv has columns ... How many units ...)"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

QUOTED = r"'([^']+)'"  # значение в кавычках, например 'widget'
MONTHS = r"(?:in|during|for) months (\d+) (?:through|to|-) (\d+)(?: inclusive)?"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        first = fh.readline()  # разделитель берём тот, которого в заголовке больше всего, при равенстве запятая
        fh.seek(0)
        return list(csv.DictReader(fh, delimiter=max(',;\t|', key=first.count)))


def _num(v: str):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return float(v)


def _sentences(text: str) -> list[str]:
    text = re.split(r"\s+(?:Use bash|Work only inside|Compute it with)", text)[0]
    return [s.strip() for s in re.split(r"(?<=[.?])\s+(?=[A-Z])", text.strip()) if s.strip()]


def _normalize_question(q: str) -> str:
    q = re.sub(r"^Find how many", "How many", q)
    q = re.sub(r"^Calculate the total", "What was the total", q)
    q = re.sub(r"^Calculate the mean", "What was the mean", q)
    q = re.sub(r"^Identify the (region|product|category) with the highest", r"Which \1 had the highest", q)
    return q


def solve_data(instruction: str, wd: Path):
    sents = _sentences(instruction)
    if len(sents) < 2:
        return None
    head = sents[0]
    m = (re.fullmatch(r"The file (\S+\.csv) has columns ([\w,]+) \((\d+) rows\)\.", head)
         or re.fullmatch(r"Read the (\d+) records in (\S+\.csv) with fields ([\w,]+)\.", head))
    if not m:
        return None
    if head.startswith("The file"):
        fname, cols = m.group(1), m.group(2).split(",")
    else:
        fname, cols = m.group(2), m.group(3).split(",")
    path = wd / fname
    if not path.is_file():
        return ("impossible", f"{fname} does not exist")
    rows = _read(path)
    header = list(rows[0].keys()) if rows else cols
    if [c.strip() for c in header] != cols:
        return None  # описание не совпадает с файлом, пусть смотрит модель

    flags = {"revenue": False, "profit": False, "clean": False, "month_range": None,
             "join": None, "restrict": None, "where": None}
    question = None
    rest = sents[1:]
    for i, s in enumerate(rest):
        if s == "Revenue is units*price.":
            flags["revenue"] = True
        elif s == "Profit is units*(price-cost).":
            flags["profit"] = True
        elif s == "Skip rows with empty units and count duplicate order_id values only once.":
            flags["clean"] = True
        elif mm := re.fullmatch(r"Months are numbered (\d+)\W+(\d+), and requested reporting periods must be "
                                r"within this range\.", s):
            flags["month_range"] = (int(mm.group(1)), int(mm.group(2)))
        elif mm := re.fullmatch(r"Join (\S+\.csv) \(([\w,]+)\) on (\w+)\.", s):
            flags["join"] = (mm.group(1), mm.group(2).split(","), mm.group(3))
        elif mm := re.fullmatch(r"Restrict to orders listed in (\S+\.csv) by (\w+)\.", s):
            flags["restrict"] = (mm.group(1), mm.group(2))
        elif s.endswith("?") or (question is not None):
            question = (question + " " + s) if question else s
        else:
            return None  # незнакомая инструкция
    if not question:
        return None

    # Round to 2 decimal places (half up) или Answer with ... in lowercase
    digits = None
    if mm := re.search(r"\s*Round to (\d+) decimal places? \(half up\)(?:, e\.g\. [\d.]+)?\.$", question):
        digits = int(mm.group(1))
        question = question[: mm.start()]
    lower = False
    if mm := re.search(r"\s*Answer with the (\w+) name in lowercase\.$", question):
        lower = True
        question = question[: mm.start()]
    question = question.strip()

    if mm := re.match(r"For orders with (\w+) above (\d+(?:\.\d+)?), (.*)$", question):
        col, thr, question = mm.group(1), float(mm.group(2)), mm.group(3)
        if col not in header:
            return ("impossible", f"the data has no column '{col}'")
        flags["where"] = (col, thr)
        question = question[0].upper() + question[1:]
    question = _normalize_question(question)

    if flags["restrict"]:
        rf, key = flags["restrict"]
        if not (wd / rf).is_file():
            return ("impossible", f"{rf} does not exist")
        ids = {r.get(key) for r in _read(wd / rf)}
        rows = [r for r in rows if r.get(key) in ids]
    cost = None
    if flags["join"]:
        jf, jcols, key = flags["join"]
        if not (wd / jf).is_file():
            return ("impossible", f"{jf} does not exist")
        table = {r[key]: r for r in _read(wd / jf)}
        cost = table
    recs = []
    for r in rows:
        units = _num(r.get("units", ""))
        if units is None:
            if flags["clean"]:
                continue
            return None  # пустые значения без правила, неоднозначно
        rec = {k: (v or "").strip() for k, v in r.items()}
        rec["units"] = units
        rec["price"] = _num(r.get("price", ""))
        rec["month"] = _num(r.get("month", ""))
        if rec["price"] is None or rec["month"] is None:
            return None
        recs.append(rec)
    if flags["clean"]:
        recs = list({r["order_id"]: r for r in recs}.values())
    elif len({r.get("order_id") for r in recs}) != len(recs):
        return None  # дубликаты без правила, неоднозначно
    if flags["where"]:
        col, thr = flags["where"]
        recs = [r for r in recs if (_num(r[col]) or 0) > thr]

    mm = re.fullmatch(rf"(.*?) {MONTHS}\?", question)
    if not mm:
        return None
    core, a, b = mm.group(1), int(mm.group(2)), int(mm.group(3))
    if a > b:
        return ("impossible", f"empty period: months {a} through {b}")
    lo, hi = flags["month_range"] or (1, 12)
    if not (lo <= a <= hi and lo <= b <= hi):
        if flags["month_range"] or not any(r["month"] > 12 for r in recs):
            return ("impossible", f"months {a}-{b} are outside the valid range {lo}-{hi}")
        return None
    period = [r for r in recs if a <= r["month"] <= b]

    def filt(rs, product=None, region=None, category=None):
        out = rs
        if product is not None:
            out = [r for r in out if r["product"] == product]
        if region is not None:
            out = [r for r in out if r["region"] == region]
        if category is not None:
            out = [r for r in out if cost and r["product"] in cost and cost[r["product"]].get("category") == category]
        return out

    def profit_of(r):
        c = cost.get(r["product"]) if cost else None
        if c is None:
            raise LookupError(r["product"])
        return r["units"] * (r["price"] - _num(c["cost"]))

    def need_profit():
        if not flags["profit"] or cost is None or "cost" not in (flags["join"][1] if flags["join"] else []):
            return ("impossible", "profit is not defined by the data (no cost information)")
        return None

    # сколько единиц продано
    if m2 := re.fullmatch(r"How many units of product " + QUOTED + r" were sold(?: in region " + QUOTED + r")?", core):
        rs = filt(period, m2.group(1), m2.group(2))
        return ("answer", _fmt(sum(r["units"] for r in rs)))
    # сколько заказов
    if m2 := re.fullmatch(r"How many orders for product " + QUOTED + r" were placed(?: in region " + QUOTED + r")?", core):
        return ("answer", str(len(filt(period, m2.group(1), m2.group(2)))))
    # общая выручка или прибыль по товару или категории, можно с регионом
    if m2 := re.fullmatch(r"What was the total (revenue|profit) for (product|category) " + QUOTED
                          + r"(?: in region " + QUOTED + r")?", core):
        metric, what, name, region = m2.groups()
        if metric == "revenue":
            if not flags["revenue"]:
                return None
            rs = filt(period, name if what == "product" else None, region)
            if what == "category":
                return None
            return ("answer", _fmt(sum(r["units"] * r["price"] for r in rs)))
        if bad := need_profit():
            return bad
        rs = filt(period, name if what == "product" else None, region,
                  name if what == "category" else None)
        return ("answer", _fmt(sum(profit_of(r) for r in rs)))
    # средняя цена из списка
    if m2 := re.fullmatch(r"What was the mean of the listed prices for product " + QUOTED
                          + r"(?: in region " + QUOTED + r")?", core):
        rs = filt(period, m2.group(1), m2.group(2))
        if not rs or digits is None:
            return None
        return ("answer", _round(Decimal(sum(r["price"] for r in rs)) / len(rs), digits))
    if re.fullmatch(r"What was the mean per-order profit for product " + QUOTED + r"(?: in region " + QUOTED + r")?", core):
        if bad := need_profit():
            return bad
        return None
    # у кого больше всех
    if m2 := re.fullmatch(r"Which (region|product) had the highest (total )?(revenue|profit)", core):
        group, _, metric = m2.groups()
        if metric == "profit":
            if bad := need_profit():
                return bad
            val = profit_of
        else:
            if not flags["revenue"]:
                return None
            val = (lambda r: r["units"] * r["price"])
        totals = defaultdict(float)
        for r in period:
            totals[r[group]] += val(r)
        if len(totals) < 1:
            return None
        top = sorted(totals.items(), key=lambda kv: -kv[1])
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None  # ничья, неоднозначно
        name = top[0][0]
        return ("answer", name.lower() if lower else name)
    return None


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != int(v):
            return None  # type: ignore[return-value]
        v = int(v)
    return str(v)


def _round(d: Decimal, digits: int) -> str:
    return str(d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
