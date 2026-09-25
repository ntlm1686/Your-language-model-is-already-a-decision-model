"""Place the Qwen and hosted Jev seed-100 Doom GIFs side by side."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def scoreboard_data(trace: Path, summary: Path, frames: int):
    rows = [json.loads(line) for line in trace.open()]
    rows = [r for r in rows if r['policy'] == 'api_position_none' and r['episode'] == 0]
    if not rows or any(r['step'] != i for i, r in enumerate(rows)):
        raise ValueError('expected contiguous seed-100 trace steps')
    if frames != math.ceil(len(rows) * 8 / 17):
        raise ValueError('GIF does not match the trace frame sampling')
    episode = json.loads(summary.read_text())['results']['api_position_none']['episodes'][0]
    if episode['seed'] != 100 or episode['steps'] != len(rows):
        raise ValueError('summary does not match the seed-100 trace')
    return rows, episode['kills'], bool(episode.get('timed_out', False))


def kills_at(data, frame: int, frames: int):
    rows, final, _ = data
    if frame >= frames - 1:
        return final, True
    row = rows[min(len(rows)-1, round(frame * 17 / 8))]
    return row['kills_before'], False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path,
                        default=Path("results/doom/qwen3_8b_seed100.gif"))
    parser.add_argument("--jev", type=Path,
                        default=Path("results/doom/jev_1_13_seed100.gif"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/doom/qwen_vs_jev_seed100.gif"))
    parser.add_argument('--qwen-trace', type=Path)
    parser.add_argument('--jev-trace', type=Path)
    parser.add_argument('--qwen-summary', type=Path)
    parser.add_argument('--jev-summary', type=Path)
    parser.add_argument('--qwen-name', default='Qwen3.5-9B')
    parser.add_argument('--jev-name', default='Jev 1.13.0')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    qwen, jev = Image.open(args.qwen), Image.open(args.jev)
    if qwen.size != (1280, 720) or jev.size != (1280, 720):
        raise ValueError("expected two 1280x720 annotated GIFs")
    sources = [args.qwen_trace, args.jev_trace, args.qwen_summary, args.jev_summary]
    with_scores = all(sources)
    if any(sources) and not with_scores:
        parser.error('scoreboard requires both traces and both summaries')
    if with_scores:
        qdata = scoreboard_data(args.qwen_trace, args.qwen_summary, qwen.n_frames)
        jdata = scoreboard_data(args.jev_trace, args.jev_summary, jev.n_frames)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-jev-gif-") as temp:
            intermediate = Path(temp) / "side_by_side.mp4"
            writer = imageio.get_writer(intermediate, fps=8, codec="libx264",
                                        quality=7, macro_block_size=16)
            try:
                for step in range(max(qwen.n_frames, jev.n_frames) + (16 if with_scores else 0)):
                    qwen.seek(min(step, qwen.n_frames - 1))
                    jev.seek(min(step, jev.n_frames - 1))
                    offset = 96 if with_scores else 0
                    frame = Image.new("RGB", (2560, 720 + offset), (11, 20, 27))
                    frame.paste(qwen.convert("RGB"), (0, offset))
                    frame.paste(jev.convert("RGB"), (1280, offset))
                    if with_scores:
                        qkills, qdone = kills_at(qdata, step, qwen.n_frames)
                        jkills, jdone = kills_at(jdata, step, jev.n_frames)
                        draw = ImageDraw.Draw(frame)
                        panel_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 22)
                        for x, name in [(700, args.qwen_name), (1980, args.jev_name)]:
                            draw.rectangle((x, offset + 18, x + 550, offset + 48), fill=(11, 20, 27))
                            draw.text((x, offset + 20), name, font=panel_font, fill=(235, 245, 242))
                        draw.text((32, 14), f'{args.qwen_name}  |  Kills: {qkills}', font=font, fill=(107, 207, 255))
                        draw.text((1430, 14), f'{args.jev_name}  |  Kills: {jkills}', font=font, fill=(255, 199, 102))
                        draw.text((1280, 33), f'{qkills} : {jkills}', anchor='mm', font=font, fill='white')
                        for base, label_x, done, data, kills in [
                                (0, 32, qdone, qdata, qkills),
                                (1280, 1430, jdone, jdata, jkills)]:
                            status = ('TIMEOUT' if data[2] else 'EPISODE ENDED') if done else 'PLAYING'
                            draw.text((label_x, 66), status, font=small, fill=(170, 190, 200))
                            if done:
                                draw.rounded_rectangle((base + 48, offset + 250, base + 640, offset + 386),
                                                       radius=12, fill=(11, 20, 27), outline=(255, 199, 102), width=3)
                                draw.text((base + 344, offset + 292), status, anchor='mm', font=font,
                                          fill=(255, 199, 102))
                                draw.text((base + 344, offset + 343), f'Final kills: {kills}  |  Replay finished',
                                          anchor='mm', font=small, fill=(235, 245, 242))
                    writer.append_data(np.asarray(frame))
            finally:
                writer.close()
            filter_complex = ("[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];"
                              "[b][p]paletteuse=dither=bayer:bayer_scale=5")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                            "-i", str(intermediate), "-filter_complex", filter_complex,
                            "-loop", "0", str(args.output)], check=True)
    finally:
        qwen.close()
        jev.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
