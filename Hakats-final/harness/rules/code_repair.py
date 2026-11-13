"""починка мелких багов перебором одиночных мутаций, без токенов"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNNER = r"""
import json, sys, unittest
sys.path.insert(0, '.')
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
res = unittest.TestResult()
suite.run(res)
print(json.dumps({"run": res.testsRun, "bad": len(res.failures) + len(res.errors)}))
"""

# только плюс и минус, умножение и деление случайно проходят тесты на пустом входе
SWAP_BIN = {ast.Add: [ast.Sub], ast.Sub: [ast.Add]}
SWAP_CMP = {ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.Lt: [ast.Gt, ast.LtE], ast.Gt: [ast.Lt, ast.GtE],
            ast.LtE: [ast.GtE, ast.Lt], ast.GtE: [ast.LtE, ast.Gt], ast.In: [ast.NotIn], ast.NotIn: [ast.In],
            ast.Is: [ast.IsNot], ast.IsNot: [ast.Is]}
SWAP_CALL = {"sorted": ["reversed", "list"], "reversed": ["sorted", "list", "iter"], "iter": ["reversed"], "min": ["max"], "max": ["min"],
             "any": ["all"], "all": ["any"], "sum": ["len"], "len": ["sum"]}
SWAP_ATTR = {"append": ["extend", "insert"], "pop": ["popitem"], "lower": ["upper"], "upper": ["lower"],
             "startswith": ["endswith"], "endswith": ["startswith"], "lstrip": ["rstrip", "strip"],
             "rstrip": ["lstrip", "strip"], "keys": ["values"], "values": ["keys"]}


def single_mutations(src: str, extra_values: tuple = ()):
    """(описание, новый исходник) для каждой одиночной правки"""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def span(node):
        return (offsets[node.lineno - 1] + _col(lines[node.lineno - 1], node.col_offset),
                offsets[node.end_lineno - 1] + _col(lines[node.end_lineno - 1], node.end_col_offset))

    def replaced(node, new_node):
        a, b = span(node)
        return src[:a] + ast.unparse(new_node) + src[b:]

    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        try:
            if isinstance(node, ast.BinOp):
                for op in SWAP_BIN.get(type(node.op), []):
                    yield f"op {ast.unparse(node)}", replaced(node, ast.BinOp(node.left, op(), node.right))
                if isinstance(node.op, ast.Sub):
                    yield f"swap {ast.unparse(node)}", replaced(node, ast.BinOp(node.right, ast.Sub(), node.left))
                if isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(node.right, ast.Constant):
                    yield f"drop {ast.unparse(node)}", replaced(node, node.left)
            elif isinstance(node, ast.AugAssign):
                for op in SWAP_BIN.get(type(node.op), []):
                    yield f"aug {ast.unparse(node)}", replaced(node, ast.AugAssign(node.target, op(), node.value))
            elif isinstance(node, ast.Compare):
                # и в цепочках вроде a <= x <= b меняем каждое сравнение по отдельности
                for position, current in enumerate(node.ops):
                    for op in SWAP_CMP.get(type(current), []):
                        ops = list(node.ops)
                        ops[position] = op()
                        yield f"cmp {ast.unparse(node)}", replaced(node, ast.Compare(node.left, ops, node.comparators))
                if len(node.ops) == 1:
                    yield from _membership_mutations(node, extra_values, replaced)
            elif isinstance(node, ast.Constant) and type(node.value) is bool:
                yield f"bool {node.value}->{not node.value}", replaced(node, ast.Constant(not node.value))
            elif isinstance(node, ast.Constant) and type(node.value) is int:
                for v in {node.value + 1, node.value - 1, 0, 1} - {node.value}:
                    yield f"const {node.value}->{v}", replaced(node, ast.Constant(v))
            elif isinstance(node, ast.BoolOp):
                other = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                yield f"bool {ast.unparse(node)}", replaced(node, ast.BoolOp(other, node.values))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                yield f"not {ast.unparse(node)}", replaced(node, node.operand)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SWAP_CALL:
                for name in SWAP_CALL[node.func.id]:
                    yield f"call {node.func.id}->{name}", replaced(
                        node, ast.Call(ast.Name(name, ast.Load()), node.args, node.keywords))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SWAP_ATTR:
                for name in SWAP_ATTR[node.func.attr]:
                    f = ast.Attribute(node.func.value, name, ast.Load())
                    yield f"attr {node.func.attr}->{name}", replaced(node, ast.Call(f, node.args, node.keywords))
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and type(node.slice.value) is int:
                for v in (0, -1, 1):
                    if v != node.slice.value:
                        yield f"index {node.slice.value}->{v}", replaced(
                            node, ast.Subscript(node.value, ast.Constant(v), node.ctx))
        except (ValueError, IndexError):
            continue
    # return product там, где имели в виду return result: пробуем другие локальные имена функции
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        names = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})
        for ret in [n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)]:
            for name in names:
                if name != ret.value.id:
                    yield f"return {ret.value.id}->{name}", replaced(ret.value, ast.Name(name, ast.Load()))


def _membership_mutations(node: ast.Compare, extra_values: tuple, replaced):
    """добавить или убрать вариант в x in (...)"""
    group = node.comparators[0]
    if not (isinstance(node.ops[0], (ast.In, ast.NotIn)) and isinstance(group, (ast.Tuple, ast.List, ast.Set))
            and all(isinstance(e, ast.Constant) for e in group.elts)):
        return
    present = [e.value for e in group.elts]
    kind = type(group)
    for i in range(len(present)):
        if len(present) > 1:
            fewer = kind(elts=[ast.Constant(v) for j, v in enumerate(present) if j != i], ctx=ast.Load())
            yield f"drop {present[i]!r} from {ast.unparse(group)}", replaced(group, fewer)
    for value in extra_values:
        if value not in present and all(type(value) is type(v) for v in present):
            more = kind(elts=[ast.Constant(v) for v in present] + [ast.Constant(value)], ctx=ast.Load())
            yield f"add {value!r} to {ast.unparse(group)}", replaced(group, more)


def _short_strings(test_source: str) -> tuple:
    """короткие строки из тестов: кандидаты на пропущенный вариант в перечислении"""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return ()
    found = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and 0 < len(n.value) <= 12 and n.value.strip() == n.value}
    return tuple(sorted(found))[:40]


def _col(line: str, byte_col: int) -> int:
    """в ast смещения в байтах UTF-8, переводим их в символы"""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def count_passing(wd: Path, lib: str, src: str, test_module: str, timeout: float) -> tuple[int, int] | None:
    tmp = Path(tempfile.mkdtemp(prefix="mut"))
    try:
        for p in wd.iterdir():
            if p.is_file() and p.suffix == ".py":
                shutil.copy2(p, tmp / p.name)
        (tmp / lib).write_text(src, encoding="utf-8")
        from ..observe import sandbox_env  # здесь, чтобы не было циклического импорта
        env = sandbox_env()  # без интернета, как в песочнице
        r = subprocess.run([sys.executable, "-c", RUNNER, test_module], cwd=tmp, capture_output=True,
                           text=True, timeout=timeout, env=env)
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        d = json.loads(line)
        return d["run"] - d["bad"], d["run"]
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def repair(wd: Path, lib: str = "mylib.py", test_module: str = "test_mylib",
           time_limit: float = 90.0, workers: int | None = None) -> list[str] | None:
    """чиним lib на месте, если мутации проводят все видимые тесты, иначе None"""
    path = wd / lib
    if not path.is_file() or not (wd / f"{test_module}.py").is_file():
        return None
    original = path.read_text(encoding="utf-8")
    src = original
    deadline = time.monotonic() + time_limit
    base = count_passing(wd, lib, src, test_module, 20)
    if base is None:
        return None
    best, total = base
    if total == 0:
        return None
    applied: list[str] = []
    workers = workers or max(2, (os.cpu_count() or 4))
    extra_values = _short_strings((wd / f"{test_module}.py").read_text(encoding="utf-8", errors="replace"))
    for _ in range(8):
        if best == total:
            break
        try:
            all_mutations = list(single_mutations(src, extra_values))
        except SyntaxError:
            return None
        seen, candidates = set(), []
        for description, mutated in all_mutations:
            if mutated not in seen and mutated != src:
                seen.add(mutated)
                candidates.append((description, mutated))
        if time.monotonic() > deadline:
            return None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(lambda c: count_passing(wd, lib, c[1], test_module, 10), candidates))
        # больше прошедших тестов, при равенстве ранний кандидат
        ranked = [(score[0], -i, candidates[i]) for i, score in enumerate(scores) if score and score[1] == total]
        if not ranked:
            return None
        top_score, _, (description, mutated) = max(ranked)
        if top_score <= best:
            return None  # ни одна одиночная правка больше не помогает
        best, src = top_score, mutated
        applied.append(description)
    if best != total:
        return None
    path.write_text(src, encoding="utf-8")
    return applied
