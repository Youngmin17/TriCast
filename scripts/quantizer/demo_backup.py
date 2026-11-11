"""Backup recording of the midterm-2 demo from an automated rehearsal (scripts/quantizer/demo_rehearsal.mjs):
one frame per step with the step, what the rehearsal checked, and what to say (the "말할 것" column of
docs/DEMO.md §2), after a title frame that says how it was made. An animated GIF (no ffmpeg needed).

    python scripts/quantizer/demo_backup.py <rehearsal_dir> <out.gif> [--seconds 6]

Prints BACKUP_DONE with the frame count and size. Needs Pillow and a Korean font (macOS AppleSDGothicNeo).
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
WIDTH = 1280


def lines_to_say(demo: Path) -> dict[int, str]:
    """Step number -> the "말할 것" cell of docs/DEMO.md §2, without emphasis and disclosure brackets."""
    say = {}
    for line in demo.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            text = re.sub(r"\*\*|`", "", cells[4])
            say[int(cells[0])] = re.sub(r"\s+", " ", text)
    return say


def frame(shot: Path | None, head: str, body: list[str], font: ImageFont.FreeTypeFont,
          small: ImageFont.FreeTypeFont) -> Image.Image:
    if shot is None:
        page = Image.new("RGB", (WIDTH, 800), "white")
    else:
        page = Image.open(shot).convert("RGB")
        page = page.resize((WIDTH, round(page.height * WIDTH / page.width)))
    wrapped = [part for line in body for part in textwrap.wrap(line, 92) or [""]]
    strip = 24 + 34 + 26 * len(wrapped) + 16
    out = Image.new("RGB", (WIDTH, page.height + strip), "white")
    out.paste(page, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, page.height, WIDTH, page.height + strip), fill=(24, 32, 44))
    draw.text((24, page.height + 18), head, font=font, fill=(255, 255, 255))
    for number, line in enumerate(wrapped):
        draw.text((24, page.height + 58 + 26 * number), line, font=small, fill=(214, 222, 232))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rehearsal", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, default=6.0, help="how long each step frame shows")
    args = parser.parse_args()
    run = json.loads((args.rehearsal / "rehearsal.json").read_text(encoding="utf-8"))
    say = lines_to_say(ROOT / "docs" / "DEMO.md")
    font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 19)
    frames = [frame(None, "TriCast Auto-tune — 중간평가 2 시연 예비 녹화", [
        "자동 리허설(scripts/quantizer/demo_rehearsal.mjs)이 화면(tune.html)에서 시연 "
        f"{len(run['steps'])}단계를 화면 조작 그대로 진행한 장면이다.",
        "배낭 계산과 레시피 파일 생성은 리허설 때 실제로 돌았고, 품질 숫자(PPL)는 A100 측정 기록이다.",
        f"단계 통과 {sum(step['pass'] for step in run['steps'])}/{len(run['steps'])} · 내레이션 없음 — "
        "아래 자막이 말할 내용이다.",
    ], font, small)]
    for step in run["steps"]:
        mark = "통과" if step["pass"] else "실패"
        body = [f"확인: {step['detail']} ({mark})"]
        if step["step"] in say:
            body.append(f"말할 것: {say[step['step']]}")
        head = f"{step['step']}/{len(run['steps'])} · {step['title']}"
        frames.append(frame(Path(step["shot"]), head, body, font, small))
    height = max(item.height for item in frames)
    canvas = []
    for item in frames:
        padded = Image.new("RGB", (WIDTH, height), "white")
        padded.paste(item, (0, 0))
        canvas.append(padded.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    durations = [5000] + [round(args.seconds * 1000)] * (len(canvas) - 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas[0].save(args.out, save_all=True, append_images=canvas[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"BACKUP_DONE frames={len(canvas)} size={args.out.stat().st_size / 1e6:.1f}MB out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
