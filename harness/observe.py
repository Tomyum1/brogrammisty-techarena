"""описание задачи для модели и предпроверки без токенов"""
from __future__ import annotations

import ast
import csv
import importlib.util
import io
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .preludes import (BASE_PRELUDE, CSV_HINT, CSV_PRELUDE, KB_HINT, KB_PRELUDE, MD_TABLES_HINT, MD_TABLES_PRELUDE,
                       SECTIONS_HINT, SECTIONS_PRELUDE, file_variables, file_variables_hint, file_variables_prelude)

TEXT_INLINE = 2500    # текстовые файлы целиком до такого размера
BIG_FILE_HEAD = 900   # столько символов показываем из начала файла, который целиком не влез
PY_INLINE = 9000      # исходники на питоне показываем целиком до такого размера
TEST_OUTPUT = 1800    # сколько символов вывода тестов оставляем модели
TEST_TIMEOUT = 60
KB_MIN_FILES = 4      # с такого числа md файлов считаем их базой знаний

STDLIB = set(getattr(sys, "stdlib_module_names", ()))


@dataclass
class Observation:
    kind: str                       # math, data, search, coding или generic
    text: str                       # что видит модель кроме самого условия
    files: list[str] = field(default_factory=list)
    test_cmd: list[str] | None = None
    sources: dict[str, bytes] = field(default_factory=dict)     # исходники библиотеки до любых правок
    test_files: dict[str, bytes] = field(default_factory=dict)  # исходные байты, возвращаем их после каждого шага
    tests_pass_initially: bool = False
    impossible: str | None = None   # причина, если невыполнимость видно без модели
    prelude: str = ""               # код на питоне, который выполняется перед каждым скриптом модели
    rule: tuple | None = None       # ("answer" или "impossible", значение) от harness.rules
    features: dict = field(default_factory=dict)


def observe(task: dict, workdir: Path) -> Observation:
    instruction = task.get("instruction", "")
    files = _list_files(workdir) if workdir.is_dir() else []
    obs = Observation(kind=task_kind(task, files), text="", files=files)
    obs.features = {"n_files": len(files), "bytes": 0, "csv_rows": 0, "py_chars": 0, "kb_files": 0,
                    "instr_chars": len(instruction)}
    obs.impossible = _missing_input_files(instruction, files)

    md_files = [f for f in files if f.endswith(".md")]
    # базе знаний из карточек делаем сводку и даём загрузчик KB
    # остальной markdown маленький, показываем как есть
    is_kb = len(md_files) >= KB_MIN_FILES and all(_is_record_card(workdir / f) for f in md_files)
    sections = _describe_files(workdir, files, is_kb, obs.features, obs.sources)
    if is_kb:
        obs.features["kb_files"] = len(md_files)
        sections.append(_describe_knowledge_base(workdir, md_files, instruction))

    hints = []
    obs.prelude = BASE_PRELUDE
    if any(f.endswith(".csv") for f in files):
        obs.prelude += CSV_PRELUDE
        hints.append(CSV_HINT)
    if obs.kind != "coding" and (named := file_variables(files)):
        obs.prelude += file_variables_prelude(named)
        hints.append(file_variables_hint(named))
    if is_kb:
        obs.prelude += KB_PRELUDE
        hints.append(KB_HINT)
    elif any(re.search(r"^\|.*\|\s*$", _read(workdir / f), re.M) for f in md_files):
        obs.prelude += MD_TABLES_PRELUDE
        hints.append(MD_TABLES_HINT)
    if not is_kb and any(SECTION_HEADER.search(_read(workdir / f)) for f in md_files):
        obs.prelude += SECTIONS_PRELUDE
        hints.append(SECTIONS_HINT)
    if hints:
        sections.insert(0, "YOUR SCRIPT RUNS AFTER A PRELUDE THAT ALREADY DEFINES: " + "; ".join(hints)
                        + "\nDo NOT open, parse or rebuild these files or variables yourself; write only "
                          "the few lines that compute the answer.")

    if obs.kind == "coding" or any(Path(f).name.startswith("test_") for f in files):
        sections += _prepare_tests(obs, workdir, instruction)

    if note := _unicode_note(workdir, files):
        sections.append(note)

    if not files:
        sections.append("(the workspace is empty)")
    else:
        listing = [f for f in files if not (is_kb and f.endswith(".md"))]
        if listing:
            sections.insert(0, "files: " + ", ".join(listing))
    obs.text = "\n\n".join(sections)
    return obs


def _unicode_note(workdir: Path, files: list[str]) -> str | None:
    """предупреждение о составных буквах (é как e плюс U+0301), с ними сравнение строк не сходится"""
    decomposed = []
    for name in files:
        text = _read(workdir / name)
        if text and unicodedata.normalize("NFC", text) != text:
            decomposed.append(name)
    if not decomposed:
        return None
    return (f"NOTE: {', '.join(decomposed[:5])} contain decomposed Unicode (a letter plus a combining mark, "
            "e.g. 'e' + U+0301 looks like 'é'). Strings that look equal may differ: do not normalize unless the task "
            "says so, take literal values from the data in code instead of retyping them, and remember each "
            "combining mark is a separate character (and UTF-16 unit).")


def task_kind(task: dict, files: list[str]) -> str:
    declared = (task.get("task_type") or "").lower()
    if declared.startswith("data"):
        return "data"
    if declared in ("math", "search", "coding"):
        return declared
    # task_type нормального нет, угадываем по файлам
    if any(Path(f).name.startswith("test_") and f.endswith(".py") for f in files):
        return "coding"
    if any(f.endswith(".csv") for f in files):
        return "data"
    if any(f.endswith(".md") for f in files):
        return "search"
    return "math" if not files else "generic"



def _list_files(workdir: Path) -> list[str]:
    return [p.relative_to(workdir).as_posix() for p in sorted(workdir.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.name != "answer.txt"
            and not (p.parent == workdir and p.name == "task.json")]  # task.json виден, когда файлы лежат без workspace/


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")  # utf-8-sig срезает BOM


# task.json и answer.txt условие упоминает, но читать их не нужно
NOT_INPUTS = {"answer.txt", "task.json"}


def _missing_input_files(instruction: str, files: list[str]) -> str | None:
    """если условие читает файл, которого в папке нет, задача невыполнима"""
    mentioned = set(re.findall(r"\b([\w-]+\.(?:csv|json|txt|py|md|xlsx|tsv))\b", instruction))
    present = {Path(f).name for f in files}

    def is_output(name: str) -> bool:  # в write ... to result.csv файл надо создать, а не прочитать
        position = instruction.find(name)
        before = instruction[max(0, position - 60):position]
        return bool(re.search(r"\b(write|create|save|output|generate|produce)\b", before, re.I))

    missing = sorted(n for n in mentioned if n not in present and n not in NOT_INPUTS and not is_output(n))
    return f"referenced file(s) missing from workspace: {', '.join(missing)}" if missing else None


def _describe_files(workdir: Path, files: list[str], is_kb: bool, features: dict,
                    sources: dict | None = None) -> list[str]:
    sections = []
    sources = {} if sources is None else sources
    for name in files:
        path = workdir / name
        size = path.stat().st_size
        features["bytes"] += size
        if name.endswith(".csv"):
            summary, row_count = _describe_csv(path, name)
            features["csv_rows"] += row_count
            sections.append(summary)
        elif name.endswith(".py"):
            source = _read(path)
            if not Path(name).name.startswith("test"):
                sources[name] = path.read_bytes()
            features["py_chars"] += len(source)
            if len(source) > PY_INLINE:
                source = source[:PY_INLINE] + "\n# ...[truncated]"
            sections.append(f"### {name}\n```python\n{source}\n```")
        elif is_kb and name.endswith(".md"):
            continue
        elif size <= TEXT_INLINE:
            sections.append(f"### {name}\n{_read(path)}")
        else:
            sections.append(_describe_big_file(path, name, size))
    return sections


def _describe_big_file(path: Path, name: str, size: int) -> str:
    """у большого файла показываем начало, чтобы модель видела заголовок"""
    text = _read(path)
    if not text.strip() or sum(ch.isprintable() or ch in "\n\r\t" for ch in text[:2000]) < 0.9 * len(text[:2000]):
        return f"### {name}: {size} bytes (not shown)"
    head = text[:BIG_FILE_HEAD]
    if "\n" in head.strip():
        head = head[: head.rstrip().rfind("\n")] if len(text) > BIG_FILE_HEAD else head
    return (f"### {name}: {size} bytes, {text.count(chr(10)) + 1} lines, only the beginning is shown:\n"
            f"{head.rstrip()}\n...")


def _describe_csv(path: Path, name: str) -> tuple[str, int]:
    """заголовок, первые строки и факты по колонкам"""
    text = _read(path)
    delimiter = max(",;\t|", key=text.split("\n", 1)[0].count)  # бывают выгрузки через ; или табуляцию
    table = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not table:
        return f"### {name}: empty file", 0
    header, body = table[0], table[1:]
    lines = [f"### {name}: {len(body)} data rows" + (f", delimiter {delimiter!r}" if delimiter != "," else ""),
             "first lines:"] + [delimiter.join(r) for r in table[:4]]
    for index, column in enumerate(header):
        values = [r[index] for r in body if index < len(r)]
        blanks = sum(1 for v in values if v.strip() == "")
        numbers = []
        for v in values:
            try:
                numbers.append(float(v))
            except ValueError:
                pass
        distinct = sorted(set(v for v in values if v.strip()))
        note = f"- {column}:"
        if numbers and len(numbers) >= len(values) - blanks:
            note += f" numeric min={min(numbers):g} max={max(numbers):g}"
            if len(distinct) <= 12:
                note += f" values={sorted(distinct, key=float)}"
        elif len(distinct) <= 15:
            note += f" values={distinct}"
        else:
            note += f" {len(distinct)} distinct, e.g. {distinct[:5]}"
        if blanks:
            note += f" blanks={blanks}"
        lines.append(note)
    ids = [r[0] for r in body if r]
    if header and len(set(ids)) != len(ids):
        lines.append(f"note: column {header[0]} has {len(ids) - len(set(ids))} duplicate values")
    return "\n".join(lines), len(body)


# запись вида ## T1 и сразу под ней строка поле: значение
SECTION_HEADER = re.compile(r"^## .+\n(?:\s*\n)*[\w ]+:", re.M)


def _is_record_card(path: Path) -> bool:
    """заголовок # Title и строки Field: value, одна запись на файл"""
    lines = _read(path).splitlines()
    return (sum(1 for line in lines if line.startswith("#")) == 1
            and any(re.match(r"^[A-Z][\w ]*:\s", line) for line in lines))


def parse_md_record(text: str) -> tuple[str | None, dict]:
    title, fields = None, {}
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return title, fields


def _describe_knowledge_base(workdir: Path, md_files: list[str], instruction: str) -> str:
    """формат записей, заполненность полей и записи из вопроса"""
    records = {f: parse_md_record(_read(workdir / f)) for f in md_files}
    groups: dict[str, list[str]] = {}
    for f in md_files:
        groups.setdefault(Path(f).stem.split("_")[0], []).append(f)
    folders = sorted({Path(f).parent.as_posix() + "/" for f in md_files})
    lines = [f"### knowledge base: {len(md_files)} markdown records in {', '.join(folders)}",
             "Each record is `# <Name>` followed by `Field: value` lines."]
    for prefix, group in groups.items():
        filled: dict[str, int] = {}
        for f in group:
            for key, value in records[f][1].items():
                if value:
                    filled[key] = filled.get(key, 0) + 1
        lines.append(f"- {prefix}_*.md: {len(group)} records; fields (non-empty count): "
                     + ", ".join(f"{k} {n}" for k, n in filled.items()))

    by_name = {title: f for f, (title, _) in records.items() if title}
    for f, (_, fields) in records.items():  # псевдонимы из Also known as ведут на ту же запись
        for key, value in fields.items():
            if "known as" in key.lower() or "alias" in key.lower():
                for alias in value.split(","):
                    by_name.setdefault(alias.strip(), f)
    named = [n for n in by_name if re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", instruction)]
    shown = named[:3] or [records[group[0]][0] for group in groups.values()][:2]
    lines.append("records named in the task:" if named else "sample records:")
    for name in shown:
        f = by_name[name]
        via_alias = f" (matched by alias '{name}')" if records[f][0] != name else ""
        lines.append(f"--- {f}{via_alias}\n{_read(workdir / f).strip()[:500]}")
    if not named:
        lines.append("(no record title matches a name in the task)")
    return "\n".join(lines)



def _prepare_tests(obs: Observation, workdir: Path, instruction: str) -> list[str]:
    """гоняем видимые тесты и запоминаем их файлы"""
    obs.test_cmd = _test_command(instruction, obs.files)
    for f in obs.files:
        if Path(f).name.startswith("test") and f.endswith(".py"):
            obs.test_files[f] = (workdir / f).read_bytes()
    if not obs.test_cmd:
        return []
    passed, output = run_tests(obs.test_cmd, workdir)
    obs.tests_pass_initially = passed
    if NO_INTERNET_MARK in output:  # сети нет, правкой не починить
        obs.impossible = "tests need internet access, which the sandbox does not have"

    # модуля нет нигде, правкой не починить
    local_modules = {Path(f).stem for f in obs.files if f.endswith(".py")}
    for test_file, source in obs.test_files.items():
        for module in _imported_modules(source.decode("utf-8", "replace")):
            if module not in local_modules and module not in STDLIB and not _installed(module):
                obs.impossible = obs.impossible or f"{test_file} imports missing module '{module}'"
        reason = needs_internet(source.decode("utf-8", "replace")) or             missing_fixture(source.decode("utf-8", "replace"), obs.files)
        if reason:
            obs.impossible = obs.impossible or f"{test_file}: {reason}"
        if not passed:
            clash = contradicting_tests(source.decode("utf-8", "replace"))
            if clash:
                obs.impossible = obs.impossible or f"{test_file}: {clash}"
    obs.features["n_tests"] = sum(len(re.findall(r"^\s*def test", s.decode("utf-8", "replace"), re.M))
                                  for s in obs.test_files.values())
    obs.features["n_fail"] = len(re.findall(r"^(?:FAIL|ERROR):", output, re.M))
    return [f"### output of `python3 {' '.join(obs.test_cmd[1:])}`\n{output}"]


def _test_command(instruction: str, files: list[str]) -> list[str] | None:
    named = re.search(r"python3?\s+([\w./-]+\.py)", instruction)
    if named and named.group(1) in files:
        return [sys.executable, named.group(1)]
    tests = [f for f in files if Path(f).name.startswith("test") and f.endswith(".py")]
    if len(tests) == 1:
        return [sys.executable, tests[0]]
    if tests:
        return [sys.executable, "-m", "unittest", "-q"] + [t[:-3].replace("/", ".") for t in tests]
    return None


# на винде дочерний процесс иногда падает с STATUS_DLL_INIT_FAILED
WINDOWS_SPAWN_FAILURE = (3221225794, -1073741502)


def run_proc(cmd: list[str], workdir: Path, timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    """subprocess.run с повтором при этой ошибке винды"""
    for attempt in range(4):
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env,
                              encoding="utf-8", errors="replace")
        if proc.returncode not in WINDOWS_SPAWN_FAILURE:
            return proc
        time.sleep(1 + attempt)
    return proc


NETBLOCK_DIR = Path(__file__).parent / "netblock"
NO_INTERNET_MARK = "no internet in the sandbox"  # так netblock подписывает свои ошибки


def sandbox_env() -> dict:
    """окружение для тестов без сети (см. netblock)"""
    paths = [str(NETBLOCK_DIR)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1")


def run_tests(cmd: list[str], workdir: Path) -> tuple[bool, str]:
    try:
        proc = run_proc(cmd, workdir, TEST_TIMEOUT, sandbox_env())
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {TEST_TIMEOUT}s"
    return proc.returncode == 0, compact_test_output((proc.stdout + "\n" + proc.stderr).strip())


# убираем шум unittest: заголовки трейсбеков, крышечки, диффы
UNITTEST_NOISE = re.compile(
    r"^(Traceback \(most recent call last\):|\s*\^+\s*$|First (differing|extra) element.*|"
    r"(First|Second) list contains.*|[-+?] .*|\s*$|Ran \d+ tests? in .*)")


def compact_test_output(output: str) -> str:
    """оставляем упавший тест, строку где упало и текст ошибки"""
    output = re.sub(r"\n-{50,}\n", "\n", output)
    output = re.sub(r"={50,}\n", "", output)
    output = re.sub(r'File "[^"]*[\\/]([^\\/"]+)"', r'File "\1"', output)  # абсолютные пути заменяем на имена файлов
    output = "\n".join(line for line in output.splitlines() if not UNITTEST_NOISE.match(line))
    if len(output) > TEST_OUTPUT:
        output = output[: TEST_OUTPUT - 400] + "\n...[truncated]...\n" + output[-400:]
    return output


def _imported_modules(source: str) -> set[str]:
    modules = set()
    for match in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", source, re.M):
        if match.group(1):
            modules.add(match.group(1).split(".")[0])
        else:
            for part in match.group(2).split(","):
                modules.add(part.strip().split(" ")[0].split(".")[0])
    return {m for m in modules if m}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# в песочнице сети нет, такой тест не пройдёт
NETWORK_CALL = re.compile(r"\b(urlopen|requests\.(get|post|head|put|request)|http\.client|create_connection|urllib3)\b")
EXTERNAL_URL = re.compile(r"https?://(?!localhost\b|127\.0\.0\.1\b|\[::1\])[\w-]+\.[\w.-]+")


def needs_internet(source: str) -> str | None:
    """тест ходит во внешнюю сеть и ничего не подменяет"""
    if re.search(r"\bmock\b|\bpatch\(|monkeypatch", source):
        return None
    if NETWORK_CALL.search(source) and EXTERNAL_URL.search(source):
        return "tests need internet access, which the sandbox does not have"
    return None


FILE_NAME = re.compile(r"^[\w./-]+\.(json|csv|txt|dat|ya?ml|toml|ini|cfg|bin|db|sqlite|pkl|xml|md|tsv|parquet)$")


def missing_fixture(source: str, files: list[str]) -> str | None:
    """тест на верхнем уровне читает файл, которого нет, а создавать файлы нельзя"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = {Path(f).name for f in files} | set(files)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and FILE_NAME.match(sub.value):
                if sub.value not in names and Path(sub.value).name not in names:
                    return f"tests read '{sub.value}' at import time, but the file does not exist"
    return None


def contradicting_tests(source: str) -> str | None:
    """два теста с одинаковым входом и разными ожидаемыми ответами"""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    seen: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test")):
            continue
        expected = []

        class Hide(ast.NodeTransformer):
            def visit_Assign(self, assign):
                if (len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name)
                        and assign.targets[0].id == "expected" and _is_literal(assign.value)):
                    expected.append(ast.unparse(assign.value))
                    assign.value = ast.Constant("?")
                return assign

            def visit_Call(self, call):
                self.generic_visit(call)
                if (isinstance(call.func, ast.Attribute) and call.func.attr == "assertEqual"
                        and len(call.args) == 2 and _is_literal(call.args[1])):
                    expected.append(ast.unparse(call.args[1]))
                    call.args[1] = ast.Constant("?")
                return call

        body = ast.unparse(ast.Module(body=[Hide().visit(stmt) for stmt in node.body], type_ignores=[]))
        if not expected:
            continue
        values = "|".join(expected)
        if body in seen and seen[body][1] != values:
            return f"{seen[body][0]} and {node.name} check the same input but expect different results"
        seen.setdefault(body, (node.name, values))
    return None


def _is_literal(node) -> bool:
    import ast
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False
