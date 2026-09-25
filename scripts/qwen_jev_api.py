"""Jev System One wire format backed by Qwen3-8B letter log probabilities.

This module has no GPU imports.  It implements the published Choice, Noul and
Score request/answer shapes; the scorer is injected so the wire protocol can
be tested without downloading a model.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def strict_json(data: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON number")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def render(value: object) -> str:
    # Preserve object insertion order, as in the archived Qwen letter scorer.
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def description(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, (str, dict, list)):
        raise ValueError("instructions and descriptions must be text, an object, or an array")
    return render(value)


def compile_request(request: object, *, model_name: str, max_questions: int = 4096,
                    max_candidates: int = 65536) -> list[dict]:
    if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
        raise ValueError("request requires state and questions")
    if request.get("model") not in (None, model_name, "qwen3-8b", "jev-latest", "jev-1.13.0", "open-jev"):
        raise ValueError("requested model is not loaded; see /v1/models")
    state, questions = request["state"], request["questions"]
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, a JSON object, or an array")
    state = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    if not isinstance(questions, Mapping) or not 1 <= len(questions) <= max_questions:
        raise ValueError("questions must be a nonempty object within the server question limit")
    records = []
    for question_id, definition in questions.items():
        if not isinstance(question_id, str) or not isinstance(definition, Mapping):
            raise ValueError("question IDs must be strings and definitions must be objects")
        kind = definition.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("question type must be choice, score, or noul")
        instruction = description(definition.get("instructions"))
        criteria = definition.get("criteria")
        record = {"id": question_id, "state": state, "kind": kind, "question": instruction}
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice requires between 1 and 255 candidates")
            if any(not isinstance(key, str) for key in criteria):
                raise ValueError("Choice candidate names must be strings")
            record["keys"] = list(criteria)
            # The archived Qwen letter protocol displays descriptions; the
            # Jev candidate key remains a software-only response label.
            record["options"] = [key if (desc := description(value, optional=True)) is None
                                 else desc for key, value in criteria.items()]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires an array of 2 to 10 descriptive levels")
            record["keys"] = [str(i) for i in range(len(criteria))]
            record["options"] = [description(value) for value in criteria]
            record["legend"] = dict(zip(record["keys"], criteria))
        else:
            if criteria is not None:
                if not isinstance(criteria, Mapping) or set(criteria) != {"true", "false"}:
                    raise ValueError("Noul criteria must contain true and false descriptions")
                options = [description(criteria["false"]), description(criteria["true"])]
            else:
                options = ["No", "Yes"]
            record["keys"] = ["false", "true"]
            record["options"] = options
        records.append(record)
    if sum(len(record["options"]) for record in records) > max_candidates:
        raise ValueError("request exceeds the candidate limit")
    return records


def softmax(values: list[float]) -> list[float]:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("scores must be finite")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def choice_confidence(probs: list[float]) -> float:
    return 1.0 if len(probs) == 1 else (max(probs) - 1 / len(probs)) / (1 - 1 / len(probs))


def score_confidence(probs: list[float]) -> float:
    mode = max(range(len(probs)), key=probs.__getitem__)
    center = (len(probs) - 1) / 2
    uniform_deviation = sum(abs(i - center) for i in range(len(probs))) / len(probs)
    return max(0.0, 1.0 - sum(p * abs(i - mode) for i, p in enumerate(probs)) / uniform_deviation)


def format_answer(record: dict, probs: list[float]) -> dict:
    if len(probs) != len(record["keys"]) or any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError("invalid model probability row")
    total = sum(probs)
    if not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=1e-6):
        raise ValueError("model probabilities must sum to one")
    probs = [p / total for p in probs]
    kind, keys = record["kind"], record["keys"]
    if kind == "noul":
        return {"type": "noul", "noul": probs[1]}
    mapping = dict(zip(keys, probs))
    if kind == "choice":
        return {"type": "choice", "choice": keys[max(range(len(probs)), key=probs.__getitem__)],
                "probabilities": mapping, "confidence": choice_confidence(probs)}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
            "probabilities": mapping, "confidence": score_confidence(probs), "legend": record["legend"]}


class Predictor:
    """The scorer returns (probabilities, input tokens, forward passes, method)."""

    def __init__(self, scorer, *, model_name: str = "qwen3-8b", max_questions: int = 4096,
                 max_candidates: int = 65536):
        self.scorer = scorer
        self.model_name = model_name
        self.max_questions = max_questions
        self.max_candidates = max_candidates

    def predict(self, request: object) -> dict:
        records = compile_request(request, model_name=self.model_name,
                                  max_questions=self.max_questions, max_candidates=self.max_candidates)
        start = time.perf_counter()
        answers, input_tokens, forwards, methods = {}, 0, 0, set()
        for record in records:
            probs, tokens, passes, method = self.scorer.score(record)
            answers[record["id"]] = format_answer(record, probs)
            input_tokens += tokens
            forwards += passes
            methods.add(method)
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "metadata": {"method": ",".join(sorted(methods)),
                             "inference_seconds": time.perf_counter() - start,
                             "forward_passes": forwards, "max_length": self.scorer.max_length,
                             "probability_calibration": "none"}}
