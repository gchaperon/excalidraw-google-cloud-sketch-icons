"""Approach C: LLM redraws each icon in sketch style, starting from approach B.

Each icon goes through ``rounds`` model calls:
  first prompt: the official icon (PNG), the approach B rendering (PNG) and its shapes
  next prompts: the official icon, the rendering of the last valid answer and its shapes
The first prompt is sent again until an answer is valid.
The model answers with compact shapes (see compact.py). The last answer that parses is
kept. The model is called through OpenRouter. The per-round PNGs are written to the build
directory; tokens and cost (from OpenRouter usage accounting) go to the DEBUG log.

Needs `rsvg-convert` (librsvg) and Chrome or Chromium (see render.py).
"""

import base64
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests

from . import excalidraw as ex
from .compact import from_excalidraw, to_excalidraw
from .render import render_icon as render

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_TITLE = "excalidraw-google-cloud-sketch-icons"  # shown in the OpenRouter activity log

STYLE = """You convert Google Cloud product icons into hand-drawn Excalidraw sketch icons.

Target style: Excalidraw's native hand-drawn look - simple shapes with loose, slightly uneven strokes,
as if sketched quickly on a whiteboard:
- Simple and recognizable: keep the characteristic silhouette and the main parts, drop tiny details.
- Few shapes: usually 4-15 shapes. Each Excalidraw shape is drawn with a rough hand-drawn stroke.
- Keep the original colors per part. Google palette: blue #4285f4, red #ea4335, yellow #fbbc04,
  green #34a853, grey #9aa0a6, light blue #aecbfa, mid blue #669df6.
- Strokes are lines of width 6 (do not draw outlines of thick strokes - draw the center line once).
- Use "curve": true for arcs and round parts (the line passes smoothly through the points;
  3-7 points are enough for an arc). Use "curve": false for straight segments and sharp corners.
  A curved line rounds every corner, so split a shape into separate lines at sharp corners.
- Use "ellipse" for circles and small filled dots (fill "solid").
- Use fill "hachure" only for larger filled areas, if the original has them.

Coordinate system: the icon box is 120 x 120 units, origin top-left, y grows downward.

Shape format (JSON):
  {"t":"line","c":"#4285f4","pts":[[x,y],...],"curve":true|false,"closed":false,"fill":"none"}
  {"t":"ellipse","c":"#ea4335","x":..,"y":..,"w":..,"h":..,"fill":"none|solid|hachure"}
  {"t":"rect","c":"#34a853","x":..,"y":..,"w":..,"h":..,"fill":"none","round":false}

Answer with a short note of what you changed (max 3 sentences), then ONE ```json code block
that contains {"shapes":[...]} and nothing else."""  # noqa: E501

ROUND1 = """Icon: {name}

Image 1 is the official icon. Image 2 is an automatic conversion of it (shapes below), drawn by Excalidraw.
The automatic conversion has the right proportions but it can have errors: missing parts, extra
short lines, wrong joins, rough or broken curves. Redraw the icon in the target style. Use the
conversion as a geometric guide, but fix everything that does not look like the official icon.

Automatic conversion shapes:
```json
{shapes}
```"""  # noqa: E501

ROUND2 = """Image 1 is the official icon. Image 2 is how Excalidraw draws your shapes (below).
Compare them carefully: silhouette, proportions, missing or extra parts, colors, curves, line ends.
If it already looks right, return the same shapes. Otherwise return corrected shapes.

Your shapes:
```json
{shapes}
```"""


def key():
    """Return the OpenRouter API key from the environment variable OPENROUTER_API_KEY.

    To read it from a .env file, run with ``uv run --env-file .env ...``.

    Raises:
        SystemExit: If the variable is not set or is empty.
    """
    if k := os.environ.get("OPENROUTER_API_KEY"):
        return k
    raise SystemExit(
        "OPENROUTER_API_KEY is not set (export it, or use `uv run --env-file .env ...`)"
    )


def png_uri(path):
    """Return a PNG file as a data URI."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def ask(model, key, text, images, icon):
    """Send one prompt with images to the model.

    A non-200 answer is tried again, up to 3 attempts in total.

    Args:
        model: OpenRouter model id.
        key: OpenRouter API key.
        text: The user prompt. The system prompt is always STYLE.
        images: Paths of PNG files, sent after the text in this order.
        icon: Icon name, for the log.

    Returns:
        A tuple (answer text, cost in USD). If all attempts fail, or the answer has no
        choices, (None, 0.0).
    """
    content = [
        {"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": png_uri(p)}} for p in images),
    ]
    body = {
        "model": model,
        "max_tokens": 16000,
        "usage": {"include": True},
        "reasoning": {"effort": "medium"},
        "messages": [
            {"role": "system", "content": STYLE},
            {"role": "user", "content": content},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "X-Title": APP_TITLE}
    t0 = time.time()
    for attempt in range(3):
        r = requests.post(API_URL, timeout=600, headers=headers, json=body)
        if r.status_code == 200:
            break
        log.warning("%s: HTTP %s: %s", icon, r.status_code, r.text[:300])
        time.sleep(5 * (attempt + 1))
    else:
        return None, 0.0
    d = r.json()
    if "choices" not in d:
        log.warning("%s: no choices: %s", icon, json.dumps(d)[:300])
        return None, 0.0
    msg = d["choices"][0]["message"].get("content") or ""
    u = d.get("usage", {})
    cost = float(u.get("cost") or 0)
    log.debug(
        "%s: %.0fs in=%s out=%s cost=$%.4f",
        icon,
        time.time() - t0,
        u.get("prompt_tokens"),
        u.get("completion_tokens"),
        cost,
    )
    return msg, cost


def parse(msg):
    """Extract the shapes and the note from a model answer.

    Args:
        msg: The answer text, or None.

    Returns:
        A tuple (shapes, note). ``shapes`` comes from the last fenced JSON block with a
        "shapes" key, and ``note`` is the text before the first fence. If there is no
        valid block, (None, "").
    """
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", msg or "", re.S)
    if not blocks:
        return None, ""
    try:
        shapes = json.loads(blocks[-1])["shapes"]
    except Exception:
        return None, ""
    note = msg.split("```")[0].strip()
    return shapes, note


def process(model, rounds, key, workdir, name, svg, b_els):
    """Run all rounds for one icon.

    Args:
        model: OpenRouter model id.
        rounds: Number of model calls.
        key: OpenRouter API key.
        workdir: Output directory for the PNGs and the final shapes JSON.
        name: Display name of the icon.
        svg: Path of the official SVG.
        b_els: Approach B elements of the icon.

    Returns:
        A tuple (library item or None if no round gave valid shapes, cost in USD).
    """
    workdir = Path(workdir)
    fn = re.sub(r"\W+", "_", name)  # file name stem
    orig = workdir / f"{fn}_orig.png"
    subprocess.run(
        ["rsvg-convert", "-w", "320", "-h", "320", "-b", "white", "-o", orig, svg],
        check=True,
    )
    b_png = workdir / f"{fn}_b.png"
    render(b_els, b_png)
    shapes = from_excalidraw(b_els)
    best, cur_png, cost_sum = None, None, 0.0
    for rnd in range(1, rounds + 1):
        if best is None:  # no valid answer yet: start from approach B
            text, imgs = ROUND1.format(name=name, shapes=json.dumps(shapes)), [orig, b_png]
        else:
            text, imgs = ROUND2.format(shapes=json.dumps(best)), [orig, cur_png]
        msg, cost = ask(model, key, text, imgs, name)
        cost_sum += cost
        new, note = parse(msg)
        if not new:
            log.warning("%s round %d: could not parse answer: %r", name, rnd, (msg or "")[:300])
            continue
        best = new
        cur_png = workdir / f"{fn}_r{rnd}.png"
        render(to_excalidraw(best, name), cur_png)
        log.info("%s round %d: %d shapes. note: %s", name, rnd, len(best), note[:300])
    if not best:
        return None, cost_sum
    (workdir / f"{fn}_shapes.json").write_text(json.dumps(best))
    return ex.item(f"c-{fn}", name, to_excalidraw(best, name)), cost_sum
