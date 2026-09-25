"""Flatten official Jev-path showcase questions into a replay JSONL."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    records = []
    for path in sorted((ROOT / "official_examples").glob("*.json")):
        ev = json.loads(path.read_text())["eval"]
        for case_id, case in ev["cases"].items():
            for node in case["models"]["typesafe"]["nodes"]:
                if not node.get("ran"):
                    continue
                for qid, qidx in node["questions"].items():
                    records.append({
                        "workflow": ev["id"], "case_id": case_id,
                        "node": node["node"], "question_id": qid,
                        "state": ev["documents"][node["doc"]],
                        "question": ev["questions"][qidx],
                        "saved_jev_answer": node.get("answers", {}).get(qid),
                        "reference": case.get("reference_answers", {})
                                         .get(node["node"], {}).get(qid),
                    })
    out = ROOT / "public_question_replay.jsonl"
    with out.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} question instances to {out}")


if __name__ == "__main__":
    main()
