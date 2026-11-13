"""решаем одну задачу, от бесплатных путей к разговору с моделью"""
from __future__ import annotations

import difflib
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .llm import LLM, Budget, BudgetExhausted
from .observe import Observation, observe
from .replies import (ANSWER_LINE, IMPOSSIBLE_LINE, absent_value_reason, code_blocks, edit_blocks, impossible_verdict, last_match,
                      normalize_answer, single_value_output, unused_numbers)
from .router import route
from .tools import TaskTools
from .telemetry import CallLog
from .workspace import restore_tests, write_answer

SYSTEM_PROMPT = """You solve one task in a sandbox directory. The harness reads files, runs tests and writes answer.txt itself.
First line of your reply: `CHECK: ok`, or `CHECK: impossible - <reason>` if the task cannot be solved as stated. It is impossible when: a needed number, file, column, field or record does not exist ("some boxes", "several deliveries"); it asks for a quantity the data cannot define (profit, cost, discount, pallets... without such data - never substitute another metric); the statement contradicts itself or the data (a second, different starting value; periods outside the stated range; division by zero); tests contradict each other, assert something false, or need network or a missing module; it asks for a fact the records say they do not keep (what was actually assigned or done, when only a rule-based recommendation exists); a needed value is only bounded (only \"between a and b\" with no further information) and the answer changes within that range. A filter that matches no rows is NOT impossible (answer 0).
Then, if ok, exactly one of:
A) One short ```python block, no comments. Its last line prints `ANSWER: <value>`, or `IMPOSSIBLE: <reason>` when the code finds needed data missing (broken or looping lookup chain, missing field).
B) `ANSWER: <value>` - only when one record states the value directly: no arithmetic, no filtering, no choosing among records (eligibility rules, earliest, best, ties need A).
C) Code fix: edit blocks, one per change, old lines copied exactly:
<<<<<<< SEARCH mylib.py
old lines
=======
new lines
>>>>>>> REPLACE
Use exact integer arithmetic (//, no floats) and apply each rounding exactly where the text says (e.g. sells 33% rounded down: stock -= stock * 33 // 100). For counting questions, when the candidates are few (up to ~10^7), enumerate them and test every condition of the text literally instead of using a formula. Format the answer exactly as the task asks; integers without .0. Never edit test files. Never guess.
Scripts start with the task data already loaded (see WORKSPACE): use those variables, never re-read or re-parse the files. In edit blocks change only the lines that need it. Hidden tests also check code the visible tests do not reach: read every function in the file and fix off-by-one, boundary and empty-input bugs there too.
Output nothing else: no analysis, no explanations, no restating the task."""

# замечания модели, когда её ответ сам по себе ничего не дал
BARE_CHECK = re.compile(r"CHECK:\s*ok\.?", re.I)
CONTINUE_AFTER_CHECK = "Continue: now the python block, the ANSWER line or the edit blocks."
CUT_OFF_NOTE = ("Your reply was cut off at the length limit and nothing ran. Send the whole solution again, much shorter: "
                "no comments, no blank lines, no explanations.")
ABSENT_VALUE_NOTE = ("A value that no row has does not make the task impossible: the filter then matches nothing and "
                     "the answer is the empty result (0 for a count or a sum). Send the python block that computes the answer.")
# выключено: math без рассуждений в комментариях ошибается чаще
FIRST_REMINDER = os.environ.get("FIRST_REMINDER", "0") == "1"
FIRST_REMINDER_TEXT = ("\n\nReply format reminder: the CHECK line, then one compact python block (or edit blocks) "
                       "with no comments and no blank lines.")

# лимиты на задачу, coding даём больше, правки и вывод тестов там длиннее
MAX_TURNS = 3
COST_CAP = {"coding": 2500, "other": 2500}          # единиц бюджета на задачу
REPLY_TOKENS = {"coding": 1500, "other": 1100}      # max_tokens для быстрой модели
MATH_REPLY_TOKENS = int(os.environ.get("MATH_REPLY_TOKENS", "1700"))  # math рассуждает в комментариях к коду
MUTATION_TRUSTED_TESTS = 6                          # с такого числа видимых тестов правку из перебора не перепроверяем
SECOND_PASS_CAP = 1.5                               # во сколько раз больше лимит на задачу во втором проходе
ROOMY_BOOST = float(os.environ.get("ROOMY_BOOST", "1.8"))  # во сколько раз поднять лимит задачи, пока бюджета много
ROOMY_UNTIL = 0.4                                    # много значит потрачено меньше этой доли
VERIFY = os.environ.get("VERIFY", "0") == "1"         # выключено: ответы почти всегда совпадают, токенов больше
VERIFY_MIN_REWARD = 3                               # перепроверяем задачи от стольких баллов
VERIFY_UNTIL = 0.5                                  # и только пока потрачено меньше половины бюджета
REASONING_MIN_REWARD = 2                            # на задачах за 1 балл не окупается
# так скрипт печатает пустой результат, в answer.txt их не пишем
EMPTY_ANSWERS = {"none", "nan", "null", "[]", "{}", "()", "''", '""', "set()"}
# после зелёных тестов сверяем код с наивной версией на случайных входах, около 1.5k единиц
REVIEW = os.environ.get("REVIEW", "1") == "1"
REVIEW_MODE = os.environ.get("REVIEW_MODE", "fuzz")  # fuzz: сверка с наивной версией, read: просто перечитать код
REVIEW_MIN_REWARD = 3                              # на задачах подешевле вызов не окупается
REVIEW_UNTIL = 0.5                                  # и только пока потрачено меньше половины бюджета
REASONING_REPLY_TOKENS = 2000                       # max_tokens для последней попытки с рассуждениями


def reward_weight(task: dict) -> float:
    """доля лимита на задачу по баллам: 0.6 за 1 балл и до 1.4 за 5 и больше"""
    reward = float(task.get("reward", 1) or 1)
    return min(1.4, max(0.6, 0.4 + 0.2 * reward))


@dataclass
class TaskResult:
    """одна строка файла с результатами"""
    task_id: str
    status: str  # solved, impossible, gave_up, error или skipped
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: int = 0                      # сколько было вызовов модели
    seconds: float = 0.0
    error: str | None = None
    answer: str | None = None
    extra: dict = field(default_factory=dict)
    predicted_cost: float | None = None
    predicted_p_success: float | None = None
    predicted_impossible: bool | None = None
    decision: str = "abandon"
    abandon_reason: str | None = None
    model_path: list[str] = field(default_factory=list)
    planner_order: int | None = None

    @property
    def cost(self) -> float:
        """единицы бюджета: выходные токены плюс входные делённые на 10"""
        return self.completion_tokens + self.prompt_tokens / 10


def solve(task: dict, workdir: Path, settings: Settings, calls: CallLog | None = None,
          budget: Budget | None = None, observation: Observation | None = None,
          second_pass: bool = False, free_only: bool = False) -> TaskResult:
    started = time.time()
    result = TaskResult(task_id=task["task_id"], status="gave_up")
    observation = observation or observe(task, workdir)
    llm = LLM(settings, budget or Budget(None), result.task_id, calls)
    attempt = Attempt(task, workdir, observation, llm, result, second_pass, free_only)

    try:
        attempt.run()
    except BudgetExhausted as exc:
        result.status, result.abandon_reason, result.error = "error", "budget_exhausted", str(exc)[:300]
    except Exception as exc:  # одна сломанная задача не должна останавливать прогон
        result.status, result.abandon_reason, result.error = "error", "error", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if observation.test_files:
            restore_tests(workdir, observation)

    result.prompt_tokens, result.completion_tokens = llm.prompt_tokens, llm.completion_tokens
    result.steps, result.model_path = llm.n_calls, llm.models
    result.extra["kind"] = observation.kind
    result.seconds = round(time.time() - started, 2)
    return result


class Attempt:
    """всё, что происходит с одной задачей, от бесплатных путей до разговора с моделью"""

    def __init__(self, task: dict, workdir: Path, observation: Observation, llm: LLM, result: TaskResult,
                 second_pass: bool = False, free_only: bool = False):
        # второй проход сразу на модели с рассуждениями
        self.second_pass = second_pass
        # бюджета нет, пробуем только бесплатные пути
        self.free_only = free_only
        self.task = task
        self.workdir = workdir
        self.obs = observation
        self.llm = llm
        self.result = result
        self.check = task.get("check")
        self.kind = observation.kind
        self.tools = TaskTools(workdir, observation)
        self.messages: list[dict] = []
        self.notes: list[str] = []  # замечания модели на следующий ход
        self.numbers_checked = False  # про потерянные числа из условия напоминаем один раз
        self.held_output: str | None = None  # вывод с ответом, который придержали ради этого напоминания
        self.zero_checked = False  # про ответ 0 переспрашиваем один раз
        self.unchanged_rewrite = False  # последний блок с кодом библиотеки ничего в файле не поменял
        self.last_script: str | None = None  # прошлый скрипт модели и его вывод, чтобы не гонять тот же код дважды
        self.last_output = ""

    # чем закончилось

    def finish(self, status: str, reason: str | None = None, answer: str | None = None) -> None:
        self.result.status = status
        self.result.abandon_reason = reason
        if status == "impossible":
            # у невыполнимой задачи файла с ответом быть не должно
            (self.workdir / "answer.txt").unlink(missing_ok=True)
        if answer is not None:
            write_answer(self.workdir, answer)
            self.result.answer = answer
            self.result.decision = "solve"


    def run(self) -> None:
        obs = self.obs
        if self.second_pass:
            for name, original in obs.sources.items():  # начинаем с чистого кода, а не с правок первой попытки
                (self.workdir / name).write_bytes(original)
            self.converse()
            return
        if obs.rule:
            verdict, value = obs.rule
            self.result.extra["solver"] = "rules"
            if verdict == "answer":
                self.finish("solved", answer=normalize_answer(value, self.check))
            else:
                self.finish("impossible", "rules: " + value)
            return
        if obs.impossible:
            self.finish("impossible", "prescreen: " + obs.impossible)
            return
        if self.kind == "coding" and obs.test_cmd and obs.tests_pass_initially:
            self.finish("solved", answer="DONE")
            return
        if self.kind == "coding":
            fixes = self.mutation_repair()
            if fixes:
                self.result.extra.update(solver="mutation", fixes=fixes)
                self.finish("solved", answer="DONE")
                if self.worth_reviewing():
                    self.review_for_hidden_tests()
                return
        if self.free_only:
            self.finish("gave_up", "budget_cap")
            return
        if self.kind != "coding" and self.route_to_engine():
            return
        self.converse()
        if self.worth_verifying():
            self.verify()
        if self.worth_reviewing():
            self.review_for_hidden_tests()

    def worth_verifying(self) -> bool:
        """перепроверяем ответ, только если задача дорогая по баллам, а бюджета ещё много"""
        return (self.kind != "coding" and self.result.status == "solved" and VERIFY
                and float(self.task.get("reward", 1) or 0) >= VERIFY_MIN_REWARD
                and self.llm.budget.comfortable(VERIFY_UNTIL))

    def verify(self) -> None:
        """решаем ещё раз с нуля, при расхождении третий голос даёт модель с рассуждениями"""
        votes = [self.result.answer]
        self.converse()
        votes.append(self.result.answer if self.result.status == "solved" else None)
        if votes[1] != votes[0]:
            self.converse(model=self.llm.settings.fallback_model)
            votes.append(self.result.answer if self.result.status == "solved" else None)
        counted = Counter(v for v in votes if v is not None)
        best, count = counted.most_common(1)[0]
        if count == 1 and votes[-1] is not None:
            best = votes[-1]  # все разные: больше всего доверяем модели с рассуждениями
        self.result.extra["votes"] = votes
        self.finish("solved", answer=best)

    def worth_reviewing(self) -> bool:
        return (REVIEW and not self.free_only and self.kind == "coding" and self.result.status == "solved"
                and bool(self.obs.test_cmd)
                and float(self.task.get("reward", 1) or 0) >= REVIEW_MIN_REWARD
                and self.llm.budget.comfortable(REVIEW_UNTIL))

    def review_for_hidden_tests(self) -> None:
        """сверка уже решённой задачи: если бюджет кончился посреди неё, решение всё равно остаётся"""
        try:
            self.compare_with_naive_version()
        except BudgetExhausted:
            restore_tests(self.workdir, self.obs)

    def compare_with_naive_version(self) -> None:
        """скрытые тесты зовут те же функции на других входах, сверяем с наивной версией"""
        libraries = {name: (self.workdir / name).read_text(encoding="utf-8", errors="replace")
                     for name in self.obs.sources if (self.workdir / name).is_file()}
        if not libraries or sum(len(text) for text in libraries.values()) > 12000:
            return
        code = "\n\n".join(f"### {name}\n```python\n{text}\n```" for name, text in libraries.items())
        tests = "\n\n".join(f"### {name}\n```python\n{raw.decode('utf-8', 'replace')}\n```"
                            for name, raw in self.obs.test_files.items())
        context = (f"TASK:\n{self.task.get('instruction', '').strip()}\n\nCODE (visible tests pass now):\n{code}\n\n"
                   f"VISIBLE TESTS:\n{tests}\n\n")
        edit_format = ("SEARCH/REPLACE edit blocks:\n<<<<<<< SEARCH mylib.py\nold lines\n=======\nnew lines\n"
                       ">>>>>>> REPLACE\n")
        if REVIEW_MODE == "fuzz":
            # сверка с наивной версией надёжнее, чем чтение кода моделью
            question = (context + "Hidden tests call the same functions on other inputs. Write one python block, "
                        "no comments: import the module, write the simplest obviously correct brute-force version of "
                        "each tested function (only from the docstrings and tests, not from the current code), call "
                        "both on 300 random small inputs including empty and size 1, compare results and raised "
                        "exceptions, and print `MISMATCH: <call> got=<value> want=<value>` for at most 5 differences, "
                        "then `DONE`. Output nothing else.")
            messages = [{"role": "user", "content": question}]
            reply = self.llm.chat(messages, max_tokens=REPLY_TOKENS["coding"])
            blocks = code_blocks(reply)
            if not blocks:
                return
            # скрипт лежит во временной папке, поэтому модуль задачи ищем по пути рабочей папки
            script = f"import sys\nsys.path.insert(0, {str(self.workdir.resolve())!r})\n" + blocks[0][1]
            output = self.tools.call("run_python", {"code": script, "language": "py"})["output"]
            restore_tests(self.workdir, self.obs)
            self.result.extra["review"] = output[:300]
            if "MISMATCH" not in output:
                return
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"script output:\n{output}\n\nIf the brute force is right, fix "
                          f"the module with {edit_format}If the brute force is wrong, reply OK. Output nothing else."}]
            reply = self.llm.chat(messages, max_tokens=REPLY_TOKENS["coding"])
        else:
            question = (context + "Hidden tests call the same functions on other inputs: empty input, sizes 0 and 1, "
                        "boundaries, ties, invalid input that docstrings or error messages describe, branches the "
                        "visible tests skip. Read every function line by line. If a real bug remains, reply with "
                        f"{edit_format}that fix only it. Keep everything the visible tests check. If there is no clear "
                        "bug, reply OK. Output nothing else.")
            reply = self.llm.chat([{"role": "user", "content": question}], max_tokens=REPLY_TOKENS["coding"])
            self.result.extra["review"] = reply.strip()[:300]
        if edit_blocks(reply):
            notes = self.tools.call("apply_edits", {"reply": reply})["notes"]
        elif self.rewrite_from_blocks(code_blocks(reply)):
            notes = []
        else:
            return
        if not notes and self.tools.call("run_tests", {})["passed"]:
            self.result.extra["review_applied"] = True
            return
        for name, text in libraries.items():  # правка сломала видимые тесты или не легла: возвращаем как было
            (self.workdir / name).write_text(text, encoding="utf-8")
        restore_tests(self.workdir, self.obs)

    def route_to_engine(self) -> bool:
        """модель переводит задачу в спеку для точного решателя"""
        routed = route(self.task, self.obs, self.tools, self.llm)
        if routed is None:
            return False
        verdict, value, details = routed
        self.result.extra.update(solver="router:" + details.get("solver", "?"),
                                 router_reply=details.get("router_reply"))
        if verdict == "answer" and self.acceptable(normalize_answer(value, self.check)):
            self.finish("solved", answer=normalize_answer(value, self.check))
            return True
        # невыполнимость от роутера перепроверяем скриптом, спека могла не так понять задачу
        self.result.extra["router_verdict"] = f"{verdict}: {value}"[:200]
        return False

    def mutation_repair(self) -> list[str] | None:
        """бесплатная первая попытка для coding, если в задаче одна библиотека и один файл тестов"""
        tests = list(self.obs.test_files)
        libraries = [f for f in self.obs.files
                     if f.endswith(".py") and f not in self.obs.test_files and "/" not in f]
        if len(tests) != 1 or len(libraries) != 1 or "/" in tests[0]:
            return None
        library = self.workdir / libraries[0]
        original = library.read_text(encoding="utf-8")
        fixes = self.tools.call("repair_by_mutation", {"library": libraries[0],
                                                       "test_module": Path(tests[0]).stem})["fixes"]
        if not fixes:
            return None
        accepted = self.tools.call("run_tests", {})["passed"]  # свой тестовый запуск задачи тоже должен пройти
        if accepted and self.obs.features.get("n_tests", 0) < MUTATION_TRUSTED_TESTS and not self.free_only:
            # мало тестов: мутация могла пройти их случайно, сверяем с описанием
            # без бюджета оставляем правку с зелёными тестами
            try:
                accepted = self.confirm_change(original, library.read_text(encoding="utf-8"))
            except BudgetExhausted:
                pass
        if not accepted:
            library.write_text(original, encoding="utf-8")
            return None
        return fixes

    def confirm_change(self, before: str, after: str) -> bool:
        diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                            "before/mylib.py", "after/mylib.py", n=2))
        instruction = self.task.get("instruction", "").strip()
        question = (f"TASK:\n{instruction}\n\nCURRENT CODE (before the change):\n```python\n{before}\n```\n\n"
                    f"PROPOSED CHANGE:\n```diff\n{diff}```\n\n"
                    "Does this change make the code correct for every input the task describes, not just the "
                    "visible tests? Answer with one word: YES or NO.")
        reply = self.llm.chat([{"role": "user", "content": question}], max_tokens=5)
        self.result.extra["mutation_review"] = reply.strip()[:20]
        return reply.strip().upper().startswith("YES")


    def converse(self, model: str | None = None) -> None:
        start_cost = self.llm.cost  # лимит считаем на этот разговор, при перепроверке их несколько
        group = "coding" if self.kind == "coding" else "other"
        cost_cap, reply_tokens = COST_CAP[group] * reward_weight(self.task), REPLY_TOKENS[group]
        if self.kind == "math":
            reply_tokens = MATH_REPLY_TOKENS
        if self.second_pass:
            cost_cap *= SECOND_PASS_CAP  # на второй проход идёт только лишний бюджет, его не жалко
            reply_tokens = REASONING_REPLY_TOKENS
        turns = MAX_TURNS
        if ROOMY_BOOST > 1 and self.llm.budget.comfortable(ROOMY_UNTIL):
            # бюджета много: больше места и лишний ход
            cost_cap *= ROOMY_BOOST
            turns += 1
        first_message = f"TASK:\n{self.task.get('instruction', '').strip()}\n\nWORKSPACE:\n{self.obs.text}"
        if FIRST_REMINDER:
            first_message += FIRST_REMINDER_TEXT
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": first_message}]
        self.result.extra["messages"] = self.messages  # сохраняется как траектория задачи

        for turn in range(turns):
            # стопаемся до вызова, который пробьёт лимит
            spent = self.llm.cost - start_cost
            if turn and spent + spent / turn * 1.2 > cost_cap:
                if not self.take_held_answer():
                    self.finish("gave_up", "task_cost_cap")
                return
            # последняя попытка на модели с рассуждениями, пока потрачено меньше 60% бюджета
            last_chance = (turn == turns - 1 and turn > 0 and self.llm.budget.comfortable()
                           and float(self.task.get("reward", 1) or 0) >= REASONING_MIN_REWARD)
            use_model = model or (self.llm.settings.fallback_model if (last_chance or self.second_pass) else None)
            reply = self.llm.chat(self.messages, model=use_model,
                                  max_tokens=REASONING_REPLY_TOKENS if use_model else reply_tokens)
            self.messages.append({"role": "assistant", "content": reply})
            if BARE_CHECK.fullmatch(reply.strip()):
                # только строка CHECK: просим продолжение в том же ходу
                self.messages.append({"role": "user", "content": CONTINUE_AFTER_CHECK})
                reply = self.llm.chat(self.messages, model=use_model,
                                      max_tokens=REASONING_REPLY_TOKENS if use_model else reply_tokens)
                self.messages.append({"role": "assistant", "content": reply})

            # CHECK: impossible, но ниже есть ANSWER или скрипт: разбираем как обычно
            stated = last_match(ANSWER_LINE, reply)
            changed_mind = ((stated is not None and not stated.lower().startswith(("impossible", "none", "n/a")))
                            or bool(code_blocks(reply) or edit_blocks(reply)))
            if turn == 0 and not changed_mind and (reason := impossible_verdict(reply)):
                if self.kind == "data" and not self.zero_checked and absent_value_reason(reason):
                    # пустой отбор это ответ 0, а не невыполнимость
                    self.zero_checked = True  # ответ 0 тут ожидаемый
                    self.messages.append({"role": "user", "content": ABSENT_VALUE_NOTE})
                    continue
                self.finish("impossible", "check: " + reason)
                return
            feedback = self.handle_reply(reply, turn)
            if feedback is None:
                return
            if reply.count("```") % 2 == 1 and not edit_blocks(reply):
                # блок не закрыт: упёрлись в лимит длины
                feedback = [CUT_OFF_NOTE]
            self.messages.append({"role": "user", "content": "\n\n".join(feedback)})
        if not self.take_held_answer():  # напоминание пришлось на последний ход: берём придержанный ответ
            self.finish("gave_up", "turn_limit")

    def handle_reply(self, reply: str, turn: int, last_turn: bool = False) -> list[str] | None:
        """делаем то, что просит ответ, возвращаем замечания на следующий ход или None, если задача решена"""
        has_tests = bool(self.obs.test_cmd)
        feedback: list[str] = []

        if edit_blocks(reply):
            feedback += self.tools.call("apply_edits", {"reply": reply})["notes"]
        elif has_tests and (rewritten := self.rewrite_from_blocks(code_blocks(reply))):
            if self.unchanged_rewrite:
                # тот же код, что в файле: при temperature 0 модель его повторит
                feedback.append(f"Your code is identical to the current {rewritten}: nothing changed. Read the failing "
                                "test, find the line that gives the wrong result and change it.")
            else:
                feedback.append(f"(your code block was written into {rewritten})")
        elif blocks := code_blocks(reply):
            script = "\n".join(code.strip() for _, code in blocks[:4])
            if script == self.last_script and not has_tests:
                # тот же скрипт: берём ответ из прошлого вывода или сдаёмся
                if not (self.take_held_answer() or self.finish_from_output(self.last_output)):
                    self.finish("gave_up", "repeated_script")
                return None
            output = "\n".join(self.tools.call("run_python", {"code": code, "language": language})["output"]
                               for language, code in blocks[:4])
            self.last_script, self.last_output = script, output
            if self.obs.test_files:
                restore_tests(self.workdir, self.obs)
            if not has_tests and not last_turn and (note := self.lost_numbers_note(blocks, output)
                                                    or self.zero_answer_note(output)):
                self.held_output = output
                return [f"script output:\n{output}", note]
            if not has_tests and self.finish_from_output(output):
                self.messages.append({"role": "tool", "content": output})  # сохраняется только в траектории
                return None
            feedback.append(f"script output:\n{output}")
            feedback += self.notes
            self.notes.clear()
        else:
            reason = last_match(IMPOSSIBLE_LINE, reply)
            answer = last_match(ANSWER_LINE, reply)
            if reason is not None and answer is None:
                self.finish("impossible", "model: " + reason[:200])
                return None
            if answer is not None and not has_tests:
                # после запуска скрипта ответ текстом принимаем, только если скрипт его и напечатал
                previous_output = next((m["content"] for m in reversed(self.messages[:-1])
                                        if m["role"] == "user"), "")
                value = normalize_answer(answer, self.check)
                if turn == 0 or value in previous_output:
                    self.finish("solved", answer=value)
                    return None
                feedback.append("That value is not in the script output. "
                                "Compute it with a python block that prints ANSWER: <value>.")
            if not has_tests:
                feedback.append("Reply with a python block, ANSWER: <value> or IMPOSSIBLE: <reason>.")

        if has_tests:
            tested = self.tools.call("run_tests", {})
            passed, test_output = tested["passed"], tested["output"]
            if passed:
                self.finish("solved", answer="DONE")
                return None
            feedback.append(f"tests still fail:\n{test_output}")
            if turn >= 1:
                feedback.append("If the tests contradict each other or assert something false, "
                                "reply IMPOSSIBLE: <reason>.")
        return feedback

    def rewrite_from_blocks(self, blocks: list[tuple[str, str]]) -> str | None:
        """блок с функциями библиотеки без SEARCH/REPLACE считаем их новой версией"""
        before = {name: (self.workdir / name).read_bytes() for name in self.obs.sources if (self.workdir / name).is_file()}
        changed = None
        for language, code in blocks:
            if language == "py":
                changed = self.tools.call("rewrite_library", {"code": code})["file"] or changed
        if changed:
            restore_tests(self.workdir, self.obs)
            def lines(raw: bytes) -> list[bytes]:
                return [line.rstrip() for line in raw.splitlines() if line.strip()]
            self.unchanged_rewrite = lines((self.workdir / changed).read_bytes()) == lines(before.get(changed, b""))
        return changed

    def lost_numbers_note(self, blocks: list[tuple[str, str]], output: str) -> str | None:
        """число из условия не встретилось в скрипте: один раз просим перечитать условие"""
        if self.kind != "math" or self.numbers_checked or last_match(ANSWER_LINE, output) is None:
            return None
        self.numbers_checked = True
        missing = unused_numbers(self.task.get("instruction", ""), "\n".join(code for _, code in blocks))
        if not missing:
            return None
        return (f"The task mentions {', '.join(missing)}, but the script never uses it. Re-read every condition of "
                "the task. If one is missing, fix the script; if the number truly does not matter, "
                "reply with the same ANSWER line.")

    def zero_answer_note(self, output: str) -> str | None:
        """ответ 0 обычно значит неверно прочитанное условие: один раз переспрашиваем"""
        if self.zero_checked or self.kind == "coding":
            return None
        answer = last_match(ANSWER_LINE, output)
        if answer is None or not re.fullmatch(r"\s*-?0+(?:\.0+)?\s*", answer):
            return None
        self.zero_checked = True
        return ("The result is 0, so nothing matched. Re-read the task: check how you read every name, literal, "
                "letter and condition (a single capital letter may be a value, not a placeholder) and which file and "
                "field you used. If something was misread, fix the script; if 0 is truly right, reply with the same "
                "ANSWER line.")

    def take_held_answer(self) -> bool:
        """ходы кончились после напоминания про числа: берём ответ, который придержали"""
        return self.held_output is not None and self.finish_from_output(self.held_output)

    def finish_from_output(self, output: str) -> bool:
        """заканчиваем задачу по выводу скрипта, если там есть ответ или невыполнимость"""
        answer = last_match(ANSWER_LINE, output)
        reason = last_match(IMPOSSIBLE_LINE, output)
        if answer and answer.upper().startswith("IMPOSSIBLE"):  # print('ANSWER:', 'IMPOSSIBLE: ...')
            answer, reason = None, answer
        if reason is not None and answer is None:
            self.finish("impossible", "model: " + reason[:200])
            return True
        if answer is None and reason is None:
            written = self.workdir / "answer.txt"  # скрипт послушался условия и сам записал ответ в файл
            if written.is_file():
                text = written.read_text(encoding="utf-8", errors="replace").strip()
                answer = text if text and "\n" not in text else None
        if answer is None:
            answer = single_value_output(output, self.check)
        if answer is not None and self.acceptable(normalize_answer(answer, self.check)):
            self.finish("solved", answer=normalize_answer(answer, self.check))
            return True
        if answer is not None and answer.strip().lower() in EMPTY_ANSWERS:
            self.notes.append(f"The script found nothing (printed {answer!r}). Check the conditions against the records; "
                              "if no record can match, reply IMPOSSIBLE: <reason>.")
        elif answer is not None:
            self.notes.append(f"The task needs an integer answer, but the script printed {answer!r}.")
        return False

    def acceptable(self, answer: str) -> bool:
        """у exact_int не бывает дробного или пустого ответа"""
        if not answer or answer.strip().lower() in EMPTY_ANSWERS:
            return False
        if re.match(r"<(?:built-in|function|bound method|class|generator|module)\b", answer.strip()):
            return False  # напечатан объект, а не значение
        return self.check != "exact_int" or bool(re.fullmatch(r"-?\d+", answer))
