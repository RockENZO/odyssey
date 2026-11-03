"""Render the crash-resume demo's real output as a GIF.

``python docs/demo/crash_resume.py > docs/demo/crash-resume.txt`` captures the run; this script
draws that text line by line (it adds timing and colour, never lines). Needs Pillow.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
lines = (root / "crash-resume.txt").read_text(encoding="utf-8").splitlines()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17)
W, H, LH, TOP = 1280, 760, 24, 74
BG, FG = "#0e192a", "#dce8f7"


def colour(line: str) -> str:
    if line.startswith("$"):
        return "#74e0ae"
    if "kill -9" in line or "exit status" in line or "waiting to be killed" in line:
        return "#ff8794"
    if "failed" in line:
        return "#ff8794"
    if "passed" in line or "accepted" in line or "verified" in line:
        return "#74e0ae"
    if line.startswith("Scripted model"):
        return "#e5c478"
    if line.startswith("  model:"):
        return "#7faeff"
    return FG


def frame(shown: list[str]) -> Image.Image:
    im = Image.new("RGB", (W, H), "#0b1424")
    d = ImageDraw.Draw(im)
    d.text((40, 16), "Odyssey graph engine: kill -9 mid-turn, resume from the checkpoint", font=bold, fill="#eef5ff")
    d.text((40, 44), "Real tools, SQLite checkpoint, SIGKILL, pytest and ledger. The model is a scripted stand-in.", font=font, fill="#b5c8e1")
    d.rounded_rectangle((24, 68, W - 24, H - 20), 14, fill=BG, outline="#2a405b", width=2)
    window = shown[-((H - TOP - 32) // LH):]
    for i, text in enumerate(window):
        d.text((44, TOP + 8 + i * LH), text, font=font, fill=colour(text))
    return im


frames, durations, shown = [], [], []
for text in lines:
    shown.append(text)
    pause = 1200 if text.startswith("$") else 700 if "kill -9" in text or "crash point" in text else 320
    frames.append(frame(shown))
    durations.append(pause)
durations[-1] = 5000
frames[0].save(root / "crash-resume.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
frames[-1].save(root / "crash-resume-preview.png")
(root / "crash-resume-manifest.json").write_text(
    json.dumps({"frames": len(frames), "duration_ms": sum(durations), "source": "crash-resume.txt (captured output of crash_resume.py)", "model": "scripted"}, indent=2) + "\n",
    encoding="utf-8",
)
print("rendered", len(frames), "frames,", sum(durations), "ms")
