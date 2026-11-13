from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

# /app только для чтения, а openJiuwen пишет логи в ./logs, поэтому работаем из tmpfs
START_DIR = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_scratch = os.path.join(os.environ.get("HOME") or tempfile.gettempdir(), "techarena_run")
os.makedirs(_scratch, exist_ok=True)
os.chdir(_scratch)

from harness.config import apply_harness_env, load_settings  # noqa: E402
from harness.llm import Budget, quiet_sdk_logs  # noqa: E402
from harness.observe import observe  # noqa: E402
from harness.planner import Plan, plan, settled_without_model  # noqa: E402
from harness.rules import solve_rules  # noqa: E402
from harness.solver import TaskResult, solve  # noqa: E402
from harness.telemetry import CallLog  # noqa: E402

SECOND_PASS = os.environ.get("SECOND_PASS", "0") == "1"  # выключен: токенов больше, баллов столько же
SECOND_PASS_UNTIL = 0.85
BUDGET_SAFETY = 0.95  # свой подсчёт токенов может чуть разойтись с прокси


def load_tasks(tasks_dir: Path) -> list[tuple[Path, dict]]:
    tasks = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        task_file = task_dir / "task.json"
        if not task_file.is_file():
            continue
        try:
            task = json.loads(task_file.read_text(encoding="utf-8-sig"))  # utf-8-sig из-за BOM
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {task_dir.name}: bad task.json ({exc})", flush=True)
            continue
        task.setdefault("task_id", task_dir.name)
        tasks.append((task_dir, task))
    return tasks


def workdir_of(task_dir: Path) -> Path:
    """если workspace/ нет, а файлы лежат рядом с task.json, работаем в папке задачи"""
    workspace = task_dir / "workspace"
    if workspace.is_dir():
        return workspace
    if any(p.name not in ("task.json", "answer.txt") for p in task_dir.iterdir()):
        return task_dir
    return workspace


def observe_all(tasks: list[tuple[Path, dict]], workers: int):
    """бесплатный проход по всем задачам, в потоках из-за тестов coding"""
    def look(item):
        task_dir, task = item
        workdir = workdir_of(task_dir)
        workdir.mkdir(exist_ok=True)
        obs = observe(task, workdir)
        obs.rule = solve_rules(task, workdir)
        return task_dir, task, obs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(look, tasks))


class ResultStore:
    """results.json, атомарно переписывается после каждой готовой задачи"""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.results: dict[str, dict] = {}
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, result: TaskResult) -> None:
        with self.lock:
            self.results[result.task_id] = {**asdict(result), "cost": result.cost}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.results, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def total_cost(self) -> float:
        with self.lock:
            return sum(r["cost"] for r in self.results.values())


def run_one(item: Plan, settings, calls: CallLog, budget: Budget, second_pass: bool = False) -> TaskResult:
    task_id = item.task["task_id"]
    # без бюджета coding ещё может решиться без модели
    remaining = budget.remaining()
    out_of_budget = ("budget_exhausted" if budget.dead else
                     "budget_cap" if remaining is not None and remaining < 0.5 * item.est_cost else None)  # оценки с запасом
    free_only = (bool(out_of_budget) and not second_pass
                 and (item.obs.kind == "coding" or settled_without_model(item.obs)))
    if out_of_budget and not free_only:
        return TaskResult(task_id=task_id, status="skipped", abandon_reason=out_of_budget)
    if second_pass and not budget.comfortable(SECOND_PASS_UNTIL):
        return TaskResult(task_id=task_id, status="skipped", abandon_reason="second_pass_budget")

    workdir = workdir_of(item.task_dir)
    result = solve(item.task, workdir, settings, calls, budget, item.obs, second_pass=second_pass, free_only=free_only)
    if free_only and result.status == "gave_up":
        result.status, result.abandon_reason = "skipped", out_of_budget

    # у невыполнимой задачи answer.txt быть не должно
    answer = workdir / "answer.txt"
    if result.status == "solved" and answer.is_file():
        if answer != item.task_dir / "answer.txt":  # без workspace/ ответ уже лежит где надо
            shutil.copyfile(answer, item.task_dir / "answer.txt")
    else:
        answer.unlink(missing_ok=True)
        (item.task_dir / "answer.txt").unlink(missing_ok=True)

    result.predicted_cost, result.predicted_p_success = item.est_cost, item.p_success
    result.predicted_impossible = bool(item.obs.impossible)
    result.extra.update(features=item.obs.features, obs_chars=len(item.obs.text),
                        instr_chars=len(item.task.get("instruction", "")))
    return result


def save_trajectory(folder: Path, result: TaskResult) -> None:
    messages = result.extra.pop("messages", None)
    if not messages:
        return
    folder.mkdir(exist_ok=True)
    record = {"task_id": result.task_id, "status": result.status, "answer": result.answer,
              "cost": result.cost, "messages": messages}
    (folder / f"{result.task_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")


def main() -> int:
    # чтобы символы вне кодировки консоли не роняли прогон
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    parser.add_argument("--out", default=None, help="results JSON (default: $HOME/techarena_results.json)")
    parser.add_argument("--limit", type=int, default=None, help="solve at most N tasks (debugging)")
    parser.add_argument("--run-id", help="run name in calls.jsonl (default: the results folder name)")
    args = parser.parse_args()

    settings = load_settings()
    apply_harness_env(settings)
    quiet_sdk_logs()

    tasks_dir = Path(START_DIR, args.tasks_dir)
    results_path = Path(START_DIR, args.out or os.path.join(os.environ.get("HOME", "/tmp"),
                                                            "techarena_results.json"))
    calls_path = results_path.with_name("calls.jsonl")
    if results_path.resolve() == calls_path.resolve():
        parser.error("the results file cannot be called calls.jsonl")
    store = ResultStore(results_path)
    trajectories = results_path.with_name("trajectories")
    calls = CallLog(calls_path, args.run_id or results_path.resolve().parent.name)

    limit = settings.token_budget * BUDGET_SAFETY if settings.token_budget is not None else None
    budget = Budget(limit)
    started = time.time()
    queue = plan(observe_all(load_tasks(tasks_dir), os.cpu_count() or 4))
    if args.limit:
        queue = queue[: args.limit]
    by_rules = sum(1 for p in queue if p.obs.rule)
    by_prescreen = sum(1 for p in queue if p.obs.impossible and not p.obs.rule)
    print(f"[plan] {len(queue)} tasks: {by_rules} settled by rules, {by_prescreen} impossible by prescreen, "
          f"estimated cost {sum(p.est_cost for p in queue):.0f}, model={settings.model}, "
          f"parallel={settings.max_parallel}, {time.time() - started:.1f}s", flush=True)

    def run_pass(items: list[Plan], second_pass: bool) -> None:
        with ThreadPoolExecutor(max_workers=settings.max_parallel) as pool:
            futures = {pool.submit(run_one, item, settings, calls, budget, second_pass): (item, position)
                       for position, item in enumerate(items)}
            for future in as_completed(futures):
                item, position = futures[future]
                task_id = item.task["task_id"]
                try:
                    result = future.result()
                except Exception as exc:
                    result = TaskResult(task_id=task_id, status="error", abandon_reason="error",
                                        error=f"{type(exc).__name__}: {exc}"[:500])
                if second_pass:
                    if result.status == "skipped":
                        continue
                    first = store.results[task_id]
                    # в учёте бюджета задача стоит обе попытки
                    result.prompt_tokens += first["prompt_tokens"]
                    result.completion_tokens += first["completion_tokens"]
                    result.steps += first["steps"]
                    result.model_path = first["model_path"] + result.model_path
                    result.extra["second_pass"] = True
                result.planner_order = position
                save_trajectory(trajectories, result)
                store.add(result)
                tag = "[retry]" if second_pass else "[done]"
                print(f"{tag} {task_id}: {result.status} answer={result.answer!r} why={result.abandon_reason} "
                      f"calls={result.steps} cost={result.cost:.0f} total={store.total_cost():.0f} "
                      f"{result.seconds}s", flush=True)

    run_pass(queue, second_pass=False)

    # остаток бюджета отдаём нерешённым задачам, начиная с самых дорогих
    if SECOND_PASS and settings.fallback_model and budget.comfortable(SECOND_PASS_UNTIL):
        unsolved = [item for item in queue
                    if store.results.get(item.task["task_id"], {}).get("status") in ("gave_up", "error")
                    and store.results[item.task["task_id"]].get("abandon_reason") != "budget_exhausted"]
        unsolved.sort(key=lambda item: -float(item.task.get("reward", 1) or 0))
        if unsolved:
            print(f"[second pass] {len(unsolved)} unsolved tasks, spent {budget.spent:.0f}", flush=True)
            run_pass(unsolved, second_pass=True)

    solved = sum(1 for r in store.results.values() if r["status"] == "solved")
    print(f"[end] solved {solved}/{len(queue)} in {time.time() - started:.0f}s, "
          f"cost={store.total_cost():.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
