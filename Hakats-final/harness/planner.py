"""порядок задач по ожидаемым баллам на единицу бюджета: reward * p_success / est_cost"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .observe import Observation

FITTED_MODEL = Path(__file__).with_name("cost_model.json")

# средние по типам: out токенов за вызов, calls вызовов на задачу, p доля решённых
DEFAULT_MODEL = {
    "chars_per_token": 3.6,
    "system_tokens": 420,   # системный промпт
    "kinds": {
        "math":    {"out": 130, "calls": 1.3, "p": 0.80},
        "data":    {"out": 155, "calls": 1.4, "p": 0.80},
        "search":  {"out": 230, "calls": 1.8, "p": 0.65},
        "coding":  {"out": 380, "calls": 2.5, "p": 0.50},
        "generic": {"out": 300, "calls": 2.0, "p": 0.50},
    },
}


def load_cost_model() -> dict:
    try:
        return json.loads(FITTED_MODEL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_MODEL


@dataclass
class Plan:
    task_dir: Path
    task: dict
    obs: Observation
    est_cost: float
    p_success: float
    score: float


def settled_without_model(obs: Observation) -> bool:
    return bool(obs.impossible or obs.rule)


KINDS = ["math", "data", "search", "coding"]
SIZE_FEATURES = ["instr_chars", "obs_chars", "n_files", "csv_rows", "kb_files", "py_chars", "n_tests", "n_fail"]


def feature_vector(kind: str, features: dict, obs_chars: int, instr_chars: int) -> list[float]:
    """признаки задачи для обученной оценки"""
    values = {**features, "obs_chars": obs_chars, "instr_chars": instr_chars}
    vector = [1.0] + [1.0 if kind == k else 0.0 for k in KINDS[1:]]
    vector += [math.log1p(float(values.get(name, 0) or 0)) for name in SIZE_FEATURES]
    return vector


def estimate(task: dict, obs: Observation, cost_model: dict) -> tuple[float, float]:
    """(ожидаемые единицы бюджета, вероятность решить) для одной задачи"""
    if settled_without_model(obs):
        return 0.0, 0.0
    if "cost_coef" in cost_model:  # обученная оценка стоимости
        x = feature_vector(obs.kind, obs.features, len(obs.text), len(task.get("instruction", "")))
        log_cost = sum(a * b for a, b in zip(x, cost_model["cost_coef"]))
        cost = math.exp(log_cost + cost_model["cost_log_variance"] / 2)
        p_success = 1 / (1 + math.exp(-sum(a * b for a, b in zip(x, cost_model["success_coef"]))))
        return cost, p_success
    kind = cost_model["kinds"].get(obs.kind, cost_model["kinds"]["generic"])
    prompt_tokens = cost_model["system_tokens"] + (
        len(task.get("instruction", "")) + len(obs.text)) / cost_model["chars_per_token"]
    calls = kind["calls"]
    # каждый вызов заново шлёт разговор, примерно 1.3 первого промпта
    input_units = prompt_tokens * (1 + (calls - 1) * 1.3) / 10
    return input_units + kind["out"] * calls, kind["p"]


def plan(items: list[tuple[Path, dict, Observation]]) -> list[Plan]:
    cost_model = load_cost_model()
    plans = []
    for task_dir, task, obs in items:
        cost, p_success = estimate(task, obs, cost_model)
        reward = float(task.get("reward", 1) or 0)
        score = float("inf") if settled_without_model(obs) else reward * p_success / max(cost, 1.0)
        plans.append(Plan(task_dir, task, obs, round(cost, 1), p_success, score))
    plans.sort(key=lambda p: (-p.score, p.est_cost))
    return plans

