"""действия в папке задачи: запуск скриптов модели и применение её правок кода"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .observe import Observation, run_proc
from .replies import edit_blocks, fix_python_slips

SCRIPT_TIMEOUT = 60
OUTPUT_LIMIT = 1500  # сколько символов вывода скрипта отдаём модели

_script_env: dict | None = None


def script_environment() -> dict:
    """окружение скриптов с python3 в PATH, на винде через шим"""
    global _script_env
    if _script_env is None:
        env = dict(os.environ)
        if os.name == "nt" or shutil.which("python3") is None:
            shim_dir = Path(tempfile.mkdtemp(prefix="py3shim"))
            interpreter = Path(sys.executable).as_posix()
            shim = shim_dir / "python3"
            shim.write_text(f'#!/bin/sh\nexec "{interpreter}" "$@"\n', encoding="utf-8", newline="\n")
            shim.chmod(0o755)
            env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        _script_env = env
    return _script_env


def run_script(code: str, workdir: Path, language: str = "py", prelude: str = "") -> str:
    """запускаем один блок от модели в папке задачи и отдаём его (подрезанный) вывод"""
    if language == "py":
        code = fix_python_slips(code)
        if prelude:
            code = prelude + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix="." + language, delete=False,
                                     encoding="utf-8", newline="\n") as handle:
        handle.write(code)
        script = handle.name
    command = [sys.executable, script] if language == "py" else ["bash", script]
    try:
        proc = run_proc(command, workdir, SCRIPT_TIMEOUT, script_environment())
        output = proc.stdout
        if proc.returncode != 0:
            output += f"\n[exit {proc.returncode}]\n" + proc.stderr[-800:]
    except subprocess.TimeoutExpired:
        # обычно это петля ссылок, задача невыполнима
        output = (f"[script timed out after {SCRIPT_TIMEOUT}s: a loop never ended. If a lookup chain in the "
                  "data loops back on itself, the task is impossible: reply IMPOSSIBLE: <reason>. Otherwise the "
                  "search is too slow: memoize or prune it]")
    finally:
        Path(script).unlink(missing_ok=True)
    if len(output) > OUTPUT_LIMIT:
        output = output[: OUTPUT_LIMIT - 300] + "\n...[truncated]...\n" + output[-300:]
    return output.strip() or "(no output)"


def apply_edits(reply: str, workdir: Path, observation: Observation) -> list[str]:
    """применяем блоки SEARCH/REPLACE и возвращаем модели замечания о правках, которые не легли"""
    notes = []
    for filename, old, new in edit_blocks(reply):
        filename = filename.strip() or next(
            (f for f in observation.files if f.endswith(".py") and f not in observation.test_files), "")
        path = (workdir / filename).resolve()
        if filename in observation.test_files or not str(path).startswith(str(workdir.resolve())):
            notes.append(f"refused edit of {filename}: test files are read-only")
            continue
        if not path.is_file():
            notes.append(f"edit failed: {filename} not found")
            continue
        source = path.read_text(encoding="utf-8")
        if old in source:
            source = source.replace(old, new, 1)
        else:
            # сравниваем без пробелов в конце строк
            pattern = r"\n".join(re.escape(line.rstrip()) + r"[ \t]*" for line in old.split("\n"))
            match = re.search(pattern, source)
            if match:
                source = source[: match.start()] + new + source[match.end():]
            else:
                # модель неточно цитирует код, поэтому целые функции подменяем по имени
                swapped = replace_definitions(source, new) if definition_names(old) else None
                if swapped is None:
                    current = "\n\n".join(definition_text(source, name) or "" for name in definition_names(old))
                    current = current.strip() or (source if len(source) <= 6000 else "")
                    notes.append(f"edit failed: SEARCH text not found in {filename}."
                                 + (f" Current code:\n{current}" if current else ""))
                    continue
                source = swapped
        path.write_text(source, encoding="utf-8")
    return notes


TOP_LEVEL_DEFINITION = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)", re.M)


def definition_names(code: str) -> list[str]:
    return TOP_LEVEL_DEFINITION.findall(code)


def definition_span(source: str, name: str) -> tuple[int, int] | None:
    """границы def или class верхнего уровня вместе с декораторами"""
    start = re.search(rf"^(?:@.*\n)*(?:async\s+def|def|class)\s+{re.escape(name)}\b", source, re.M)
    if not start:
        return None
    header_end = source.find("\n", start.end())
    if header_end == -1:
        return start.start(), len(source)
    following = re.compile(r"^\S", re.M).search(source, header_end + 1)
    end = following.start() if following else len(source)
    while end > start.start() and source[end - 2: end] == "\n\n":  # пустые строки между функциями оставляем снаружи
        end -= 1
    return start.start(), end


def definition_text(source: str, name: str) -> str | None:
    span = definition_span(source, name)
    return source[span[0]: span[1]].rstrip() if span else None


def replace_definitions(source: str, new_code: str) -> str | None:
    for name in definition_names(new_code):
        span = definition_span(source, name)
        body = definition_text(new_code, name)
        if span is None or body is None:
            return None
        rest = source[span[1]:].lstrip("\n")
        source = source[: span[0]] + body + ("\n\n\n" + rest if rest else "\n")
    return source if definition_names(new_code) else None


IMPORT_LINE = re.compile(r"^(?:import|from)\s+\S.*$", re.M)


def apply_rewrite(code: str, workdir: Path, observation: Observation) -> str | None:
    """функции из обычного блока подменяют функции mylib.py по имени, все сразу заменяют файл"""
    names = definition_names(code)
    if not names or re.search(r"^(?:import|from)\s+(?:unittest|test_)", code, re.M):
        return None
    if re.search(r"^(?:from\s+(\w+)\s+import|import\s+(\w+))", code, re.M):
        imported = {a or b for a, b in re.findall(r"^(?:from\s+(\w+)\s+import|import\s+(\w+))", code, re.M)}
    else:
        imported = set()
    best, best_overlap = None, 0
    for name in observation.sources:
        if not name.endswith(".py") or name in observation.test_files or Path(name).stem in imported:
            continue
        path = workdir / name
        if not path.is_file():
            continue
        overlap = len(set(names) & set(definition_names(path.read_text(encoding="utf-8", errors="replace"))))
        if overlap > best_overlap:
            best, best_overlap = name, overlap
    if best is None:
        return None
    path = workdir / best
    source = path.read_text(encoding="utf-8")
    if set(definition_names(source)) <= set(names):
        path.write_text(code.rstrip() + "\n", encoding="utf-8")  # блок переопределяет всё, это новый файл целиком
        return best
    for name in names:
        body = definition_text(code, name)
        if body is None:
            continue
        span = definition_span(source, name)
        if span is None:
            # новая вспомогательная функция: ставим её перед первой функцией файла
            first = TOP_LEVEL_DEFINITION.search(source)
            at = first.start() if first else len(source)
            source = source[:at] + body + "\n\n\n" + source[at:]
        else:
            rest = source[span[1]:].lstrip("\n")
            source = source[: span[0]] + body + ("\n\n\n" + rest if rest else "\n")
    missing = [line for line in IMPORT_LINE.findall(code.split("\ndef ")[0]) if line not in source]
    if missing:
        source = "\n".join(missing) + "\n" + source
    path.write_text(source, encoding="utf-8")
    return best


def restore_tests(workdir: Path, observation: Observation) -> None:
    """возвращаем исходные файлы тестов, что бы с ними ни сделали скрипт или правка"""
    for name, original in observation.test_files.items():
        path = workdir / name
        if not path.is_file() or path.read_bytes() != original:
            path.write_bytes(original)


def write_answer(workdir: Path, value: str) -> None:
    (workdir / "answer.txt").write_text(value + "\n", encoding="utf-8")
