"""Convert an upstream AlexWortega Doom replay to a readable 8-fps GIF."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("gif", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (f"[0:v]fps={args.fps},scale={args.width}:-1:flags=lanczos,"
                      "split[a][b];[a]palettegen=stats_mode=diff[p];"
                      "[b][p]paletteuse=dither=bayer:bayer_scale=5")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                    "-i", str(args.video), "-filter_complex", filter_complex,
                    "-loop", "0", str(args.gif)], check=True)
    print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
