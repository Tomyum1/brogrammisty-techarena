"""разбор ответов модели и починка её частых промахов"""
from __future__ import annotations

import re

FENCED_BLOCK = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.S)
BLOCK_LANGUAGES = {"python": "py", "py": "py", "python3": "py", "": "py",
                   "bash": "sh", "sh": "sh", "shell": "sh", "console": "sh"}
EDIT_BLOCK = re.compile(r"<<<<<<< SEARCH[ \t]*([^\n]*)\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
ANSWER_LINE = re.compile(r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", re.M)
IMPOSSIBLE_LINE = re.compile(r"^\s*\**IMPOSSIBLE\**\s*:?\s*(.*)$", re.M)
# двоеточие обязательно, иначе за CHECK примем комментарий вида # Check ...
CHECK_LINE = re.compile(r"^\W*CHECK\W*:\s*(.*)$", re.M | re.I)

# иногда модель кладёт команды шелла в блок ```python, такие запускаем через bash
SHELL_COMMAND = re.compile(r"^\s*(python3?\s+(-c|\S+\.py)|bash|echo|cat|ls|grep|cd|awk|sed|head|tail|wc|find)\b")


def last_match(pattern: re.Pattern, text: str) -> str | None:
    found = pattern.findall(text)
    return found[-1] if found else None


def impossible_verdict(reply: str) -> str | None:
    """причина, если первая строка CHECK говорит что задачу не решить"""
    match = CHECK_LINE.search(reply.split("```")[0])  # строка CHECK стоит до кода, внутри блоков её не ищем
    if not match:
        return None
    verdict = match.group(1).strip()
    if re.match(r"(ok|okay|fine|possible|solvable)\b", verdict, re.I):
        return None
    if re.search(r"impossible|missing|contradict|undefined|not (given|defined|specified)|unknown", verdict, re.I):
        return verdict[:200]
    return None


NO_MATCHING_ROWS = re.compile(r"\bno\b[^.;]{0,25}\brows?\b|\bcontains? only ['\"`]", re.I)
ABSENT_QUOTED = re.compile(r"(.{0,25})['\"`]([^'\"`]+)['\"`] (is not present|does not exist|is absent|not found|does not appear)", re.I)
MISSING_STRUCTURE = re.compile(r"\b(no|lacks?|without|missing)\b[^,.;]{0,30}\b(columns?|fields?|files?)\b", re.I)
STRUCTURE_WORD = re.compile(r"\b(columns?|fields?|files?|headers?|keys?|tables?)\b", re.I)


def absent_value_reason(reason: str) -> bool:
    """отказ из-за пустого отбора, а не из-за нехватки колонки или файла"""
    if MISSING_STRUCTURE.search(reason):
        return False
    if NO_MATCHING_ROWS.search(reason):
        return True
    quoted = ABSENT_QUOTED.search(reason)
    if not quoted:
        return False
    before, name = quoted.group(1), quoted.group(2)
    return not STRUCTURE_WORD.search(before) and not re.search(r"\.\w{2,5}$", name)


def _is_shell(code: str) -> bool:
    try:
        compile(code, "<reply>", "exec")
        return False
    except SyntaxError:
        return bool(SHELL_COMMAND.match(code))


def code_blocks(reply: str) -> list[tuple[str, str]]:
    """запускаемые блоки кода по порядку, как (язык, код), язык это py или sh"""
    blocks = []
    for tag, code in FENCED_BLOCK.findall(reply):
        language = BLOCK_LANGUAGES.get(tag.lower())
        if language == "py" and _is_shell(code):
            language = "sh"
        if language and code.strip():
            blocks.append((language, code))
    return blocks


def edit_blocks(reply: str) -> list[tuple[str, str, str]]:
    """блоки SEARCH/REPLACE как (имя файла, старый текст, новый текст)"""
    return EDIT_BLOCK.findall(reply)


def fix_python_slips(code: str) -> str:
    """чиним голую строку ANSWER: expr и print(ANSWER: x)"""
    def bare_line(match: re.Match) -> str:
        tag, rest = match.group(1), match.group(2).strip()
        if tag == "IMPOSSIBLE":
            return f"print({('IMPOSSIBLE: ' + rest)!r})"
        return f"print('ANSWER:', {rest})"

    code = re.sub(r"^(ANSWER|IMPOSSIBLE)\s*:\s*(.+)$", bare_line, code, flags=re.M)
    try:
        compile(code, "<reply>", "exec")
        return code
    except SyntaxError:
        pass

    def broken_print(match: re.Match) -> str:
        tag, rest = match.group(1), match.group(2)
        if "{" in rest:  # тут явно хотели f строку
            return f'print(f"{tag}: {rest}")'
        return f"print('{tag}:', {rest})"

    fixed = re.sub(r"print\(\s*(ANSWER|IMPOSSIBLE)\s*:\s*(.*?)\)\s*$", broken_print, code, flags=re.M)
    try:
        compile(fixed, "<reply>", "exec")
        return fixed
    except SyntaxError:
        return code


def normalize_answer(value: str, check: str | None) -> str:
    """снимаем обёртку, которую модель вешает на значение ({1152}, '1152', `Oslo`, 1735.0, точка в конце)"""
    answer = value.strip().strip("`").strip()
    if re.fullmatch(r"\{[^{}:]*\}|\[[^\[\]]*\]|\([^()]*\)", answer) and "," not in answer:
        answer = answer[1:-1].strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
        answer = answer[1:-1].strip()
    if not re.fullmatch(r"\d+\.\d*", answer):  # 12.30 не трогаем, а точку в конце предложения убираем
        answer = answer.rstrip(".")
    if check == "exact_int":
        try:
            number = float(answer.replace(",", "").replace(" ", ""))
            if number == int(number):
                answer = str(int(number))
        except (ValueError, OverflowError):  # inf и nan тоже бывают
            pass
    return answer


def single_value_output(output: str, check: str | None) -> str | None:
    """если скрипт напечатал ровно одно короткое значение и больше ничего, это и есть ответ"""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if len(lines) != 1 or "[exit" in output or "timed out" in output or output.strip() == "(no output)":
        return None
    value = lines[0].strip()
    if check == "exact_int":
        return value if re.fullmatch(r"-?\d[\d,]*(\.0+)?", value) else None
    return value if len(value) <= 40 and ":" not in value else None


# числа словами, 0, 1 и 2 не берём
NUMBER_WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
                "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
TASK_BOILERPLATE = re.compile(r"\s*Work only inside the current directory.*", re.S)


def unused_numbers(instruction: str, code: str) -> list[str]:
    """числа из условия, которых нет в скрипте, так ловим потерянное условие"""
    text = TASK_BOILERPLATE.sub("", instruction)
    wanted: dict[str, float] = {}
    for match in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.]*\w)", text):
        if float(match.group(0)) >= 3:
            wanted[match.group(0)] = float(match.group(0))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            wanted[word] = NUMBER_WORDS[word]
    in_code = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", code)]
    list_sizes = {lit.count(",") + 1 for lit in re.findall(r"[\[(]([^\[\]()]*\S[^\[\]()]*)[\])]", code)}

    def present(value: float) -> bool:
        slack = 1 if value.is_integer() else 1e-9
        return (any(abs(x - value) <= slack or abs(x - value / 100) < 1e-9 for x in in_code)
                or value in list_sizes)

    return [name for name, value in wanted.items() if not present(value)]
