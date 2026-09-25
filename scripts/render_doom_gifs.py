"""Render a Doom replay with the Choice request, scores, and game telemetry.

The trace records the shared Choice state, options, and response at every step.
This renderer is for the Qwen/CLM/Jev/Open-Jev client videos. The upstream
AlexWortega videos already contain their own score and input panels.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval_doom_qwen import ACTIONS, POSITION_NONE_OPTIONS, render_text


COLORS = {"turn_left": (239, 186, 80), "turn_right": (77, 199, 150),
          "attack": (238, 104, 92)}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size: int, bold: bool = False):
    path = FONT_PATH.replace("Mono.ttf", "Mono-Bold.ttf") if bold else FONT_PATH
    return ImageFont.truetype(path, size)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                 *, width: int = 63, line_height: int = 17,
                 color: tuple[int, int, int] = (180, 198, 205),
                 max_y: int = 704) -> int:
    for line in textwrap.wrap(text, width=width, break_long_words=False,
                              break_on_hyphens=False):
        if y + line_height > max_y:
            break
        draw.text((x, y), line, font=font(13), fill=color)
        y += line_height
    return y


def render_frame(game_frame: np.ndarray, row: dict, model_name: str) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (11, 20, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Doom: Defend the Center", font=font(25, True),
              fill=(235, 245, 242))
    draw.text((700, 20), model_name, font=font(22, True), fill=(235, 245, 242))
    draw.text((700, 51), "Choice input: labels-derived text; 3 actions", font=font(13),
              fill=(145, 174, 181))

    canvas.paste(Image.fromarray(game_frame[:480, :640]), (24, 80))
    draw.rectangle((24, 80, 663, 559), outline=(61, 83, 92), width=1)
    draw.text((24, 582), f"step {row['step']:03d}   kills {row['kills_before']:02d}   "
              f"ammo {row['ammo']:02d}   health {row['health']:03d}", font=font(17, True),
              fill=(227, 235, 232))
    draw.text((24, 613), f"chosen: {row['action']}     decision: "
              f"{row['latency_s'] * 1000:.0f} ms     input tokens: {row['input_tokens']}",
              font=font(15), fill=COLORS[row["action"]])
    draw.text((24, 648), "seed 100  |  4 game tics per action (114 ms game time)",
              font=font(13), fill=(145, 174, 181))

    draw.text((700, 86), "Action scores", font=font(17, True), fill=(227, 235, 232))
    probabilities = row["probabilities"]
    for index, action in enumerate(ACTIONS):
        y = 123 + index * 46
        score = float(probabilities[action])
        draw.text((700, y), action, font=font(15, action == row["action"]),
                  fill=COLORS[action])
        draw.rectangle((835, y + 1, 1175, y + 17), fill=(38, 57, 65))
        draw.rectangle((835, y + 1, 835 + round(340 * score), y + 17),
                       fill=COLORS[action])
        draw.text((1190, y), f"{score:.3f}", font=font(14), fill=(232, 238, 236))

    draw.line((690, 269, 1250, 269), fill=(46, 71, 81), width=1)
    draw.text((700, 281), "Model input (shared Choice request)", font=font(16, True),
              fill=(227, 235, 232))
    state = {"enemies": row["enemies"], "ammo": row["ammo"], "health": row["health"]}
    y = draw_wrapped(draw, "State: " + render_text(state), 700, 311)
    y += 6
    y = draw_wrapped(draw, "Question: Which statement about the nearest visible enemy "
                     "is most accurate? Choose its matching action.", 700, y,
                     color=(230, 212, 158))
    y += 6
    for action in ACTIONS:
        y = draw_wrapped(draw, f"{action}: {POSITION_NONE_OPTIONS[action]}", 700, y,
                         color=COLORS[action]) + 2
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True,
                        help="temporary seed-100 MP4 from eval_doom_qwen.py")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", default="api_position_none")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.open()]
    rows = [row for row in rows if row["policy"] == args.policy and row["episode"] == 0]
    if not rows:
        raise ValueError("trace contains no matching seed-100 rows")
    if any(row["step"] != i for i, row in enumerate(rows)):
        raise ValueError("trace step numbers are not contiguous")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(args.video)
    if reader.count_frames() != len(rows):
        raise ValueError("video frame count does not match the trace")
    with tempfile.TemporaryDirectory(prefix="doom-gif-") as temp:
        intermediate = Path(temp) / "annotated.mp4"
        writer = imageio.get_writer(intermediate, fps=args.fps, codec="libx264",
                                    quality=7, macro_block_size=16)
        try:
            count = math.ceil(len(rows) * args.fps / 17)
            for k in range(count):
                index = min(len(rows) - 1, round(k * 17 / args.fps))
                image = render_frame(reader.get_data(index), rows[index], args.model_name)
                writer.append_data(np.asarray(image))
        finally:
            writer.close()
            reader.close()
        filter_complex = ("[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];"
                          "[b][p]paletteuse=dither=bayer:bayer_scale=5")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                        "-i", str(intermediate), "-filter_complex", filter_complex,
                        "-loop", "0", str(args.output)], check=True)
    print(f"wrote {args.output} ({count} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
