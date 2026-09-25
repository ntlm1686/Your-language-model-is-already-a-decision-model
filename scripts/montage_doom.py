"""Create an eight-second four-policy GIF preview from the detailed GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INPUTS = (
    ("Qwen3-8B", "qwen3_8b_seed100.gif"),
    ("CLM-v0.1-8B", "clm_seed100.gif"),
    ("Open-Jev-2B", "open_jev_2b_seed100.gif"),
    ("Open-Jev-9B", "open_jev_9b_seed100.gif"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/doom"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/doom/text_models_preview.gif"))
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()
    sources = [Image.open(args.directory / filename) for _, filename in INPUTS]
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
    frames = []
    try:
        for step in range(args.seconds * 8):
            canvas = Image.new("RGB", (720, 680), (10, 18, 24))
            draw = ImageDraw.Draw(canvas)
            for pane, source in enumerate(sources):
                source.seek(min(step, source.n_frames - 1))
                crop = source.convert("RGB").crop((24, 80, 664, 680))
                x, y = (pane % 2) * 360, (pane // 2) * 340
                canvas.paste(crop.resize((360, 338)), (x, y))
                draw.rectangle((x + 4, y + 4, x + 202, y + 26), fill=(10, 18, 24))
                draw.text((x + 10, y + 7), INPUTS[pane][0], font=label_font,
                          fill=(244, 244, 242))
                if step >= source.n_frames:
                    draw.text((x + 288, y + 7), "ENDED", font=label_font,
                              fill=(242, 181, 93))
            frames.append(canvas)
    finally:
        for source in sources:
            source.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=125, loop=0, optimize=True)
    print(f"wrote {args.output} ({len(frames)} frames at 8 fps)")


if __name__ == "__main__":
    main()
