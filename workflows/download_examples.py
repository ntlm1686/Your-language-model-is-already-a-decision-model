"""Download the four public TypeSafe workflow example assets.

The official viewer publishes only five selected examples per workflow, not
the complete 711-case evaluation corpus or executable policy harness.
"""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
NAMES = (
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
)


def main():
    out = ROOT / "official_examples"
    out.mkdir(exist_ok=True)
    manifest = []
    for name in NAMES:
        url = f"https://evals.typesafe.ai/{name}-cases.js"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        body = response.text.strip()
        prefix = "__VIEWER_DATA__("
        if not body.startswith(prefix) or not body.endswith(");"):
            raise ValueError(f"unexpected asset format: {url}")
        data = json.loads(body[len(prefix):-2])
        ev = data["eval"]
        if ev["id"] != name:
            raise ValueError(f"unexpected workflow id: {ev['id']}")
        path = out / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        manifest.append({
            "workflow": name,
            "source": url,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "reported_total_cases": ev["n_cases"],
            "public_example_cases": len(ev["cases"]),
            "question_definitions": len(ev["questions"]),
            "file": str(path),
        })
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
