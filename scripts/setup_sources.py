"""Fetch pinned upstream code, model heads and public benchmark inputs.

Large model weights, third-party repositories, and raw trajectories are kept
outside Git. Only the requested components are downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "CLM": ("https://github.com/Contrastive-LM/CLM.git", "d43894a1c97c83d2bb1b5cfcaac5759f72ece8a6"),
    "jevbench": ("https://github.com/fstandhartinger/jevbench.git", "26eb72d4e0e60d8ace0adfc77a384063442561cd"),
    "jev-phishing-bench": ("https://github.com/anisselbd/jev-phishing-bench.git", "1d56e8c64d029a9554a0874e2ef2901ed196e230"),
    "When2Call": ("https://github.com/NVIDIA/When2Call.git", "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"),
    "MetaTool": ("https://github.com/HowieHwong/MetaTool.git", "35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb"),
    "Open-Jev": ("https://github.com/Zefan-Cai/Open-Jev.git", "3308a15ccd7eea1df7a37d6ddc39b023b801ba16"),
}
MODEL_REV = "b968826d9c46dd6066d109eabc6255188de91218"
HEAD_REVS = {
    "Contrastive-LM/CLM-v0.1-8B": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
    "Contrastive-LM/deepswe-clm-heads-8k": "c60876f3fdf7a75dc58d33b776e469b7e903d0ee",
}
OPEN_JEV_REVS = {
    "ZefanCai/Open-Jev-2B": "0c7aa498b1627be8da4acf34c863ff0ee0a92785",
    "ZefanCai/Open-Jev-9B": "47e966881e489511c0c7f5633a9e1960a676a551",
}
DEEPSWE_REPO = "kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B"
DEEPSWE_REV = "e58b83310346104971fd17c42f9955e7cd5e3b25"
WEBPRM_REPO = "ZYao720/WEBPRMBENCH"
WEBPRM_REV = "badefe7a5372e73b0736186b598e299292d4dd59"
BFCL_REV = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_FILES = {
    "BFCL_v4_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "possible_answer/BFCL_v4_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
}
RUNS = (
    "qwen3.8-27b-mini-swe-xhigh",
    "qwen3.8-27b-claude-code-xhigh",
    "qwen3.8-27b-pi-medium",
    "qwen3.8-27b-pi-xhigh",
)


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def get_git_repo(name: str) -> None:
    url, commit = SOURCES[name]
    path = ROOT / "external" / name
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        run("git", "init", str(path))
        run("git", "remote", "add", "origin", url, cwd=path)
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                             capture_output=True, text=True, check=False)
    if current.returncode == 0 and current.stdout.strip() == commit:
        return
    # Fetch only the pinned commit. HTTP/1.1 avoids some HTTP/2 proxy failures.
    run("git", "-c", "http.version=HTTP/1.1", "fetch", "--depth=1", "--filter=blob:none", "origin", commit, cwd=path)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=path)


def get_head(repo: str, filename: str, outdir: Path) -> None:
    path = hf_hub_download(repo_id=repo, filename=filename, revision=HEAD_REVS[repo], local_dir=outdir)
    print(path)


def get_webprm() -> None:
    path = hf_hub_download(
        repo_id=WEBPRM_REPO, repo_type="dataset", revision=WEBPRM_REV,
        filename="data/test-00000-of-00001.jsonl", local_dir=ROOT / "data/webprm",
    )
    # Hub stores the repository-relative path under local_dir/data/.
    target = ROOT / "data/webprm/test-00000-of-00001.jsonl"
    source = Path(path)
    if source != target:
        source.replace(target)
    print(target)


def get_deepswe() -> None:
    tasks = json.loads((ROOT / "data/deepswe/heldout_tasks.json").read_text())
    for run_name in RUNS:
        for task in tasks:
            for filename in ("trajectory.json.gz", "result.json"):
                name = f"runs/{run_name}/tasks/datacurve__{task}/{filename}"
                hf_hub_download(
                    repo_id=DEEPSWE_REPO, repo_type="dataset", revision=DEEPSWE_REV,
                    filename=name, local_dir=ROOT / "data/deepswe",
                )
    print(f"downloaded {len(tasks)} tasks x {len(RUNS)} runs x 2 files")


def get_bfcl() -> None:
    base = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/" + BFCL_REV +
            "/berkeley-function-call-leaderboard/bfcl_eval/data/")
    for name, expected in BFCL_FILES.items():
        path = ROOT / "data/bfcl" / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        response = requests.get(base + name, timeout=60)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != expected:
            raise ValueError(f"BFCL source checksum mismatch for {name}: {digest}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        print(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("components", nargs="+", choices=("repos", "heads", "model", "qwen_tokenizer", "webprm", "deepswe", "phishing", "when2call", "metatool", "bfcl", "open_jev"))
    args = p.parse_args()
    for component in args.components:
        if component == "repos":
            for name in ("CLM", "jevbench"):
                get_git_repo(name)
        elif component == "heads":
            get_head("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt", ROOT / "models/reference_head")
            get_head("Contrastive-LM/deepswe-clm-heads-8k", "best_head.pt", ROOT / "models/deepswe_head")
        elif component == "model":
            print(snapshot_download(repo_id="Qwen/Qwen3-8B", revision=MODEL_REV,
                                    local_dir=ROOT / "models/qwen3_8b"))
        elif component == "qwen_tokenizer":
            print(snapshot_download(repo_id="Qwen/Qwen3-8B", revision=MODEL_REV,
                                    local_dir=ROOT / "models/qwen3_8b_tokenizer",
                                    allow_patterns=["config.json", "tokenizer*", "vocab*", "merges*",
                                                    "added_tokens.json", "special_tokens_map.json",
                                                    "chat_template.jinja"]))
        elif component == "webprm":
            get_webprm()
        elif component == "deepswe":
            get_deepswe()
        elif component == "phishing":
            get_git_repo("jev-phishing-bench")
            run("python", "prepare_data.py", cwd=ROOT / "external/jev-phishing-bench")
        elif component == "when2call":
            get_git_repo("When2Call")
        elif component == "metatool":
            get_git_repo("MetaTool")
        elif component == "bfcl":
            get_bfcl()
        elif component == "open_jev":
            get_git_repo("Open-Jev")
            for repo, revision in OPEN_JEV_REVS.items():
                size = repo.rsplit("-", 1)[-1].lower()
                print(snapshot_download(repo_id=repo, revision=revision,
                                        local_dir=ROOT / "models" / f"open_jev_{size}"))


if __name__ == "__main__":
    main()
