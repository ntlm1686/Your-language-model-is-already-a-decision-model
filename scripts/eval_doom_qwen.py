"""Play ViZDoom Defend the Center with the local Qwen Jev-format API.

Environment and text rendering follow AlexWortega/openjev code/doom.py at
revision 058a6c24911b46d908fbe23541390f8af3df3e4d. The published code
and this adapter use labels-buffer-derived text, not raw pixels, for this run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import vizdoom as vzd


ACTIONS = ("turn_left", "turn_right", "attack")
BUTTONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
FRAME_SKIP = 4
ENEMY_NAMES = {
    "Zombieman": "zombie soldier", "ShotgunGuy": "shotgun guard", "Imp": "imp",
    "Demon": "pinky demon", "MarineChainsaw": "chainsaw marine",
    "MarineChainsawVzd": "chainsaw marine", "ChaingunGuy": "chaingunner",
    "HellKnight": "hell knight", "Cacodemon": "cacodemon", "LostSoul": "lost soul",
    "Revenant": "revenant", "BaronOfHell": "baron of hell",
}
POSITION_OPTIONS = {
    "turn_left": "The nearest enemy is to the left of the crosshair.",
    "turn_right": "The nearest enemy is to the right of the crosshair.",
    "attack": "The nearest enemy is exactly on the crosshair.",
}
POSITION_NONE_OPTIONS = {
    "turn_left": "The nearest enemy is to the left of the crosshair, or no enemy is visible.",
    "turn_right": "The nearest enemy is to the right of the crosshair.",
    "attack": "The nearest enemy is exactly on the crosshair.",
}
ACTION_OPTIONS = {
    "turn_left": "Turn left to search for or aim at an enemy.",
    "turn_right": "Turn right to aim at an enemy.",
    "attack": "Fire the pistol at an enemy under the crosshair.",
}
RULE_OPTIONS = {
    "turn_left": "Turn left if no enemy is visible, or the nearest enemy is left of the crosshair.",
    "turn_right": "Turn right if the nearest visible enemy is right of the crosshair.",
    "attack": "Fire only if the nearest visible enemy is within 0.03 of the crosshair and ammunition remains.",
}


def make_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "defend_the_center.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(2100)
    game.init()
    assert len(game.get_available_buttons()) == len(BUTTONS)
    return game


def parse_state(game: vzd.DoomGame) -> dict | None:
    state = game.get_state()
    if state is None:
        return None
    height, width = state.screen_buffer.shape[:2]
    enemies = []
    for label in state.labels:
        if label.object_name not in ENEMY_NAMES or label.width == 0:
            continue
        offset = (label.x + label.width / 2) / width - 0.5
        enemies.append({"name": ENEMY_NAMES[label.object_name], "off": float(offset),
                        "size": float(label.height / height)})
    enemies.sort(key=lambda enemy: abs(enemy["off"]))
    return {
        "enemies": enemies,
        "ammo": int(game.get_game_variable(vzd.GameVariable.AMMO2)),
        "health": int(game.get_game_variable(vzd.GameVariable.HEALTH)),
        "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
        "frame": state.screen_buffer,
    }


def render_text(state: dict) -> str:
    """Match the author's labels-buffer-to-text renderer exactly."""
    if state["enemies"]:
        parts = []
        for enemy in state["enemies"][:4]:
            side = ("right of" if enemy["off"] > 0.015 else
                    "left of" if enemy["off"] < -0.015 else "exactly on")
            distance = ("very close" if enemy["size"] > 0.45 else
                        "close" if enemy["size"] > 0.25 else "far")
            parts.append(f"a {enemy['name']} {abs(enemy['off']):.2f} to the {side} the crosshair ({distance})"
                         .replace("to the exactly on", "exactly on"))
        seen = "Visible enemies: " + "; ".join(parts) + "."
    else:
        seen = "No enemies are visible right now."
    return (
        f"Doom, Defend the Center. You stand in the middle of a circular arena with a pistol "
        f"({state['ammo']} bullets, health {state['health']}). "
        f"Enemies walk toward you from all sides and attack when close; you can only turn left, turn right, or fire. "
        f"Screen offsets are fractions of the screen width (0 = crosshair, 0.5 = screen edge); "
        f"one turn step moves the view by about 0.05. {seen} "
        f"A shot hits only if an enemy is within about 0.03 of the crosshair."
    )


def oracle(state: dict) -> int:
    """Author's labels-buffer heuristic; reference ceiling, not a learned policy."""
    if not state["enemies"]:
        return 0
    nearest = state["enemies"][0]
    if abs(nearest["off"]) < 0.03:
        return 2 if state["ammo"] > 0 else (0 if nearest["off"] < 0 else 1)
    return 0 if nearest["off"] < 0 else 1


def api_choice(state: dict, variant: str, url: str, timeout: float,
               endpoint_model: str, expected_model: str,
               api_key: str | None = None) -> tuple[int, dict, float, int]:
    options = {"qwen_position": POSITION_OPTIONS,
               "qwen_position_none": POSITION_NONE_OPTIONS,
               "qwen_action": ACTION_OPTIONS,
               "qwen_rule": RULE_OPTIONS}
    criteria = options["qwen_position_none" if variant == "api_position_none" else variant]
    instructions = ("Which statement about the nearest visible enemy is most accurate? "
                    "Choose its matching action." if variant.startswith("qwen_position") or
                    variant == "api_position_none" else
                    "Which immediate action should maximize kills and survival?")
    body = {"state": render_text(state), "questions": {"action": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}
    if endpoint_model:
        body["model"] = endpoint_model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
        if response.status != 200:
            raise ValueError(f"decision API returned HTTP {response.status}")
    latency = time.perf_counter() - started
    if expected_model and result.get("model") != expected_model:
        raise ValueError("unexpected decision API model")
    answer = result["answers"]["action"]
    probabilities = answer["probabilities"]
    if answer["type"] != "choice" or set(probabilities) != set(criteria):
        raise ValueError("invalid decision API response")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
        raise ValueError("decision probabilities do not sum to one")
    selected = answer["choice"]
    if selected != max(probabilities, key=probabilities.get):
        raise ValueError("selected action disagrees with probabilities")
    return ACTIONS.index(selected), probabilities, latency, result["usage"]["input_tokens"]


def append_video(writer, state: dict, action: int, latency: float | None) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(state["frame"])
    canvas = Image.new("RGB", (image.width, image.height + 48), (15, 20, 26))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    milliseconds = "—" if latency is None else f"{latency * 1000:.0f} ms"
    draw.text((8, image.height + 5),
              f"kills {state['kills']}  ammo {state['ammo']}  health {state['health']}  "
              f"action {ACTIONS[action]}  decision {milliseconds}", fill=(230, 235, 235))
    writer.append_data(__import__("numpy").asarray(canvas))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--policies", nargs="+", default=["random", "oracle", "qwen_position", "qwen_action"],
                        choices=["random", "oracle", "qwen_position", "qwen_position_none",
                                 "qwen_action", "qwen_rule", "api_position_none"])
    parser.add_argument("--endpoint-model", default="qwen3-8b",
                        help="model field sent to the API; empty string omits it")
    parser.add_argument("--expected-model", default="qwen3-8b",
                        help="resolved response model; empty string accepts any")
    parser.add_argument("--output", type=Path, default=Path("results/doom/qwen_text.json"))
    parser.add_argument("--trace", type=Path, default=Path("results/doom/qwen_text_steps.jsonl"))
    parser.add_argument("--video", type=Path, help="MP4 for the selected policy's first episode")
    parser.add_argument("--video-policy", default="qwen_position",
                        help="which selected policy to record at its first episode")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-wall-seconds", type=float,
                        help="Stop each episode after this much real elapsed time, including inference")
    parser.add_argument("--api-key-file", type=Path,
                        help="read a hosted API token from an ignored local file")
    parser.add_argument('--warmup', action='store_true', help='Warm the initial-state request before the clock starts')
    parser.add_argument('--hard-deadline', action='store_true', help='Discard decisions returned after the wall-clock deadline')
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must be unique")
    api_key = args.api_key_file.read_text().strip() if args.api_key_file else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
    game = make_game()
    rng = random.Random(0)
    summary = {"environment": "ViZDoom defend_the_center.cfg", "vizdoom_version": vzd.__version__,
               "upstream_revision": "058a6c24911b46d908fbe23541390f8af3df3e4d",
               "observation": "labels-buffer-derived text", "frame_skip": FRAME_SKIP,
               "seed_base": args.seed_base, "episodes": args.episodes,
               "max_wall_seconds": args.max_wall_seconds,
               "hard_deadline": args.hard_deadline, "warmup": args.warmup,
               "endpoint_model": args.endpoint_model, "expected_model": args.expected_model,
               "results": {}}
    try:
        with args.trace.open("w") as trace:
            for policy in args.policies:
                episodes, latencies, action_counts, total_tokens = [], [], Counter(), 0
                for episode in range(args.episodes):
                    seed = args.seed_base + episode
                    game.set_seed(seed)
                    game.new_episode()
                    if args.warmup and policy not in ('random', 'oracle'):
                        api_choice(parse_state(game), policy, args.url, args.timeout,
                                   args.endpoint_model, args.expected_model, api_key)
                    episode_start = time.monotonic()
                    deadline = (episode_start + args.max_wall_seconds
                                if args.max_wall_seconds is not None else None)
                    video_writer = None
                    if args.video and policy == args.video_policy and episode == 0:
                        import imageio.v2 as imageio
                        video_writer = imageio.get_writer(args.video, fps=17, codec="libx264",
                                                          quality=7, macro_block_size=16)
                    steps, episode_tokens = 0, 0
                    try:
                        while not game.is_episode_finished():
                            if deadline is not None and time.monotonic() >= deadline:
                                break
                            state = parse_state(game)
                            if state is None:
                                break
                            started = time.perf_counter()
                            elapsed_before = time.monotonic() - episode_start
                            if policy == "random":
                                action, probabilities, tokens = rng.randrange(3), None, 0
                                latency = time.perf_counter() - started
                            elif policy == "oracle":
                                action, probabilities, tokens = oracle(state), None, 0
                                latency = time.perf_counter() - started
                            else:
                                remaining = max(.001, deadline - time.monotonic()) if deadline and args.hard_deadline else args.timeout
                                try:
                                    action, probabilities, latency, tokens = api_choice(
                                        state, policy, args.url, min(args.timeout, remaining),
                                        args.endpoint_model, args.expected_model, api_key)
                                except (TimeoutError, urllib.error.URLError):
                                    if deadline and args.hard_deadline and time.monotonic() >= deadline:
                                        break
                                    raise
                                if deadline and args.hard_deadline and time.monotonic() >= deadline:
                                    break
                            if video_writer is not None:
                                append_video(video_writer, state, action, latency)
                            trace.write(json.dumps({"policy": policy, "episode": episode,
                                                    "seed": seed, "step": steps, "action": ACTIONS[action],
                                                    "kills_before": state["kills"], "ammo": state["ammo"],
                                                    "health": state["health"], "enemies": state["enemies"],
                                                    "probabilities": probabilities, "latency_s": latency,
                                                    "elapsed_before_s": elapsed_before,
                                                    "elapsed_after_s": time.monotonic() - episode_start,
                                                    "input_tokens": tokens}, ensure_ascii=False) + "\n")
                            trace.flush()
                            latencies.append(latency)
                            action_counts[ACTIONS[action]] += 1
                            total_tokens += tokens
                            episode_tokens += tokens
                            game.make_action(BUTTONS[action], FRAME_SKIP)
                            steps += 1
                    finally:
                        if video_writer is not None:
                            video_writer.close()
                    result = {"seed": seed, "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                              "reward": float(game.get_total_reward()), "steps": steps,
                              "input_tokens": episode_tokens,
                              "wall_seconds": round(time.monotonic() - episode_start, 3),
                              "timed_out": bool(deadline is not None and not game.is_episode_finished())}
                    episodes.append(result)
                    print(f"{policy} episode {episode+1}/{args.episodes}: "
                          f"kills={result['kills']} steps={steps}", flush=True)
                kills = [item["kills"] for item in episodes]
                summary["results"][policy] = {"episodes": episodes, "mean_kills": statistics.mean(kills),
                                              "mean_reward": statistics.mean(item["reward"] for item in episodes),
                                              "mean_steps": statistics.mean(item["steps"] for item in episodes),
                                              "p50_latency_ms": round(1000 * statistics.median(latencies), 2),
                                              "p95_latency_ms": round(1000 * sorted(latencies)[math.ceil(.95 * len(latencies)) - 1], 2),
                                              "mean_latency_ms": round(1000 * statistics.mean(latencies), 2),
                                              "action_counts": dict(action_counts),
                                              "input_tokens": total_tokens}
    finally:
        game.close()
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: {"mean_kills": value["mean_kills"],
                              "p50_latency_ms": value["p50_latency_ms"]}
                      for name, value in summary["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
