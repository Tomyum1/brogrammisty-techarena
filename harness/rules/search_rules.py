"""вопросы к базе знаний вида the employer of the mentor of X в спеки поиска"""
from __future__ import annotations

import re
from pathlib import Path

from .engines import run_lookup

RELATIONS = {"employer": "Employer", "mentor": "Mentor", "previous employer": "Previous employer",
             "current employer": "Employer"}
RELATION = "|".join(sorted((re.escape(k) for k in RELATIONS), key=len, reverse=True))
NAME = re.compile(r"[A-Z][\w-]*(?: [A-Z0-9][\w-]*)+")

PREFIX = re.compile(r"(?:The directory kb/ contains a small knowledge base of markdown files\.|"
                    r"Look up the answer in the Markdown records under kb/\.)\s+(.+)", re.S)


def parse_entity(expression: str) -> tuple[str, list[str]]:
    """the employer of the mentor of X's mentor превращается в ('X', ['Mentor', 'Mentor', 'Employer'])"""
    expression = expression.strip()
    outer = re.fullmatch(rf"the ({RELATION}) of (.+)", expression)
    if outer:
        start, path = parse_entity(outer.group(2))
        return start, path + [RELATIONS[outer.group(1)]]
    possessive = re.fullmatch(rf"(.+?)'s ({RELATION})", expression)
    if possessive:
        start, path = parse_entity(possessive.group(1))
        return start, path + [RELATIONS[possessive.group(2)]]
    if not NAME.fullmatch(expression):
        raise ValueError(f"not a name: {expression}")
    return expression, []


def to_spec(question: str) -> dict | None:
    field_questions = [
        (r"In which city is the headquarters of (.+)\?", "Headquarters"),
        (r"In which year was (.+) founded\?", "Founded"),
        (r"In which year was (.+) born\?", "Born"),
        (r"What is the role of (.+)\?", "Role"),
    ]
    for pattern, field in field_questions:
        if match := re.fullmatch(pattern, question):
            start, path = parse_entity(match.group(1))
            return {"query": "path", "start": start, "path": path, "field": field}
    if match := re.fullmatch(r"(?:Who is|What is the full name of|What is the name of) (.+)\?", question):
        start, path = parse_entity(match.group(1))
        if not path:
            return None
        # ответ это значение последней ссылки, есть для него запись или нет
        return {"query": "value", "start": start, "path": path[:-1], "field": path[-1]}
    if match := re.fullmatch(r"Starting at (.+?), follow the Mentor field to the end of the chain\. What is the "
                             r"full name of the last person, who has no listed mentor\?", question):
        return {"query": "chain_end", "start": match.group(1), "link": "Mentor"}
    if match := re.fullmatch(r"How many people born before (\d+) currently work for a company headquartered "
                             r"in ([A-Z][\w -]*)\?", question):
        return {"query": "count", "kind": "person",
                "where": [{"field": "Born", "op": "<", "value": int(match.group(1))},
                          {"path": ["Employer"], "field": "Headquarters", "value": match.group(2)}]}
    if match := re.fullmatch(r"Which current employer of a direct mentee of (.+) was founded earliest\?", question):
        return {"query": "pick", "kind": "person", "where": [{"field": "Mentor", "value": match.group(1)}],
                "map": ["Employer"], "by": "Founded", "order": "min"}
    return None


def solve_search(instruction: str, workdir: Path):
    text = re.split(r"\s+(?:Use the file tools|Work only inside|Use bash)", instruction)[0].strip()
    match = PREFIX.fullmatch(text)
    if not match:
        return None
    try:
        spec = to_spec(match.group(1).strip())
    except ValueError:
        return None
    if spec is None:
        return None
    return run_lookup(workdir, spec)
