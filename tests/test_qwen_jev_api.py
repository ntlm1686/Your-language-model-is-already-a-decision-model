"""Fast wire-schema tests; no model weights or GPU required."""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen_jev_api import Predictor, compile_request, strict_json
from serve_qwen_jev import make_server
from eval_qwen_jev_api import body_for


class FakeScorer:
    max_length = 8192

    def score(self, record):
        kind = record["kind"]
        probs = {"choice": [0.9, 0.1], "noul": [0.02, 0.98],
                 "score": [0.0, 0.7, 0.3]}[kind]
        return probs, 12, 2, "letter_logprob_rotations"


class WireTests(unittest.TestCase):
    def setUp(self):
        self.body = {"model": "jev-1.13.0", "state": {"ticket": "Broken shoes"},
                     "questions": {
                         "department": {"type": "choice", "instructions": "Which team?",
                                        "criteria": {"returns": "Exchanges and refunds", "shipping": "Delivery"}},
                         "refund": {"type": "noul", "instructions": "Is a refund requested?",
                                    "criteria": {"true": "Explicit request", "false": "No request"}},
                         "severity": {"type": "score", "instructions": "How severe?",
                                      "criteria": ["Cosmetic", "Degraded", "Blocking"]},
                     }}

    def test_three_jev_answer_shapes_and_expected_score(self):
        result = Predictor(FakeScorer()).predict(self.body)
        self.assertEqual(result["model"], "qwen3-8b")
        self.assertEqual(result["usage"], {"input_tokens": 36, "output_tokens": 0})
        answers = result["answers"]
        self.assertEqual(set(answers), set(self.body["questions"]))
        self.assertEqual(answers["department"], {
            "type": "choice", "choice": "returns",
            "probabilities": {"returns": 0.9, "shipping": 0.1}, "confidence": 0.8})
        self.assertEqual(answers["refund"], {"type": "noul", "noul": 0.98})
        self.assertEqual(answers["severity"]["type"], "score")
        self.assertAlmostEqual(answers["severity"]["score"], 1.3)
        self.assertEqual(answers["severity"]["legend"],
                         {"0": "Cosmetic", "1": "Degraded", "2": "Blocking"})
        self.assertEqual(answers["severity"]["probabilities"],
                         {"0": 0.0, "1": 0.7, "2": 0.3})
        json.dumps(result, allow_nan=False)

    def test_structured_state_descriptions_and_choice_255(self):
        body = {"state": ["first", "second"], "questions": {"q": {
            "type": "choice", "instructions": {"task": "Pick"},
            "criteria": {str(i): None for i in range(255)}}}}
        records = compile_request(body, model_name="qwen3-8b")
        self.assertEqual(len(records[0]["options"]), 255)
        self.assertEqual(records[0]["question"], '{"task": "Pick"}')
        self.assertEqual(records[0]["state"], ["first", "second"])

    def test_noul_uses_descriptions_in_no_yes_order(self):
        records = compile_request(self.body, model_name="qwen3-8b")
        self.assertEqual(records[0]["options"], ["Exchanges and refunds", "Delivery"])
        self.assertEqual(records[1]["options"], ["No request", "Explicit request"])
        self.assertEqual(records[1]["question"], "Is a refund requested?")

    def test_jevbench_runner_uses_labels_order_without_gold(self):
        row = {"id": "case", "state": "State", "expected": "beta",
               "labels": ["beta", "alpha"], "question": {"type": "choice",
               "instructions": "Pick", "criteria": {"alpha": "A", "beta": "B"}}}
        body = body_for(row)
        self.assertEqual(list(body["questions"]["decision"]["criteria"]), ["beta", "alpha"])
        self.assertNotIn("expected", json.dumps(body))
        self.assertEqual(list(row["question"]["criteria"]), ["alpha", "beta"])

    def test_bad_requests_rejected(self):
        with self.assertRaises(ValueError):
            strict_json(b'{"state":1,"state":2}')
        with self.assertRaises(ValueError):
            strict_json(b'{"state": NaN}')
        for body in ({"state": "x"}, {"state": "x", "questions": {}},
                     {"model": "different", **{k: v for k, v in self.body.items() if k != "model"}}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                compile_request(body, model_name="qwen3-8b")

    def test_http_schema_and_errors(self):
        server = make_server(Predictor(FakeScorer()), port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            request = urllib.request.Request(base + "/v1/systemone",
                data=json.dumps(self.body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
            self.assertEqual(body["answers"]["department"]["choice"], "returns")
            with urllib.request.urlopen(base + "/v1/models") as response:
                self.assertEqual(json.load(response)["models"][0]["id"], "qwen3-8b")
            bad = urllib.request.Request(base + "/v1/systemone", data=b'{"state":42}',
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad)
            self.assertEqual(caught.exception.code, 422)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
