"""Approach C: LLM redraws each icon in sketch style, starting from approach B.

Each icon goes through ``rounds`` model calls:
  first prompt: the official icon (PNG), the approach B rendering (PNG) and its shapes
  next prompts: the official icon, the rendering of the last valid answer and its shapes
The first prompt is sent again until an answer is valid. An answer is valid if its shapes
pass validation (see compact.py) and Excalidraw can draw them. The last valid answer is
kept. The model is called through OpenRouter. The per-round PNGs are written to the build
directory; tokens and cost (from OpenRouter usage accounting) go to the DEBUG log.

Needs `rsvg-convert` (librsvg) and Chrome or Chromium (see render.py).
"""

import asyncio
import base64
import logging
import os
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import compact
from . import excalidraw as ex
from .compact import Shape, from_excalidraw, to_excalidraw
from .render import Renderer, RenderError

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1"
APP_TITLE = "excalidraw-google-cloud-sketch-icons"  # shown in the OpenRouter activity log
ATTEMPTS = 3
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

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


def key() -> str:
    """Return the OpenRouter API key from the environment variable OPENROUTER_API_KEY.

    To read it from a .env file, run with ``uv run --env-file .env ...``.

    Raises:
        SystemExit: If the variable is not set or is empty.
    """
    if k := os.environ.get("OPENROUTER_API_KEY", ""):
        return k
    raise SystemExit(
        "OPENROUTER_API_KEY is not set (export it, or use `uv run --env-file .env ...`)"
    )


def png_uri(path: Path) -> str:
    """Return a PNG file as a data URI."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


class _Response(BaseModel):
    """The parts of an OpenRouter chat completion that we use (other keys are ignored)."""

    model_config = ConfigDict(strict=True)

    class Choice(BaseModel):
        class Message(BaseModel):
            content: str | None = None

        message: Message

    class Usage(BaseModel):
        cost: float | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

    choices: list[Choice] = Field(min_length=1)
    usage: Usage = Usage()


def client(key: str) -> httpx2.AsyncClient:
    """Return an HTTP client for the OpenRouter API, authenticated with ``key``.

    The transport tries a failed connection again; ``ask`` handles HTTP errors.
    """
    return httpx2.AsyncClient(
        base_url=API_URL,
        headers={"Authorization": f"Bearer {key}", "X-Title": APP_TITLE},
        timeout=httpx2.Timeout(600, connect=30),  # long answers take minutes
        transport=httpx2.AsyncHTTPTransport(retries=2),
    )


def _retry_delay(response: httpx2.Response | None, attempt: int) -> float:
    """Seconds to wait before the next attempt; uses Retry-After if the server sends it."""
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    if retry_after.isdigit():
        return min(float(retry_after), 60.0)
    return 5.0 * (attempt + 1)


async def ask(
    http: httpx2.AsyncClient, model: str, text: str, images: Sequence[Path], icon: str
) -> tuple[str | None, float]:
    """Send one prompt with images to the model.

    Network errors, timeouts, 408, 429 and 5xx answers are tried again, up to
    ATTEMPTS attempts in total. Other HTTP errors (for example 400 or 401) are not.

    Args:
        http: Client from ``client()``.
        model: OpenRouter model id.
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
    t0 = time.monotonic()
    for attempt in range(ATTEMPTS):
        r = None
        try:
            r = await http.post("/chat/completions", json=body)
        except httpx2.TransportError as e:
            log.warning("%s: %s: %s", icon, type(e).__name__, e)
        else:
            if r.status_code == 200:
                break
            log.warning("%s: HTTP %s: %s", icon, r.status_code, r.text[:300])
            if r.status_code not in RETRY_STATUS:
                return None, 0.0
        if attempt + 1 < ATTEMPTS:
            await asyncio.sleep(_retry_delay(r, attempt))
    else:
        return None, 0.0
    try:
        response = _Response.model_validate_json(r.content)
    except ValidationError:
        log.warning("%s: unexpected response: %s", icon, r.text[:300])
        return None, 0.0
    usage = response.usage
    cost = usage.cost if usage.cost is not None else 0.0
    log.debug(
        "%s: %.0fs in=%s out=%s cost=$%.4f",
        icon,
        time.monotonic() - t0,
        usage.prompt_tokens,
        usage.completion_tokens,
        cost,
    )
    content = response.choices[0].message.content
    return content if content is not None else "", cost


class InvalidAnswer(Exception):  # noqa: N818 (an answer, not an error of our code)
    """A model answer without valid shapes. The message says why."""


def parse(msg: str | None) -> tuple[list[Shape], str]:
    """Extract and validate the shapes and the note of a model answer.

    Args:
        msg: The answer text, or None.

    Returns:
        A tuple (shapes, note). ``shapes`` comes from the last fenced JSON block, which
        must be an object with a non-empty "shapes" list of valid shapes (see
        ``compact.validate_answer``). ``note`` is the text before the first fence.

    Raises:
        InvalidAnswer: If the answer has no such block or any shape is invalid.
    """
    if msg is None or not msg.strip():
        raise InvalidAnswer("empty answer")
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", msg, re.S)
    if not blocks:
        raise InvalidAnswer("no fenced JSON block")
    try:
        shapes = compact.validate_answer(blocks[-1])
    except ValidationError as e:
        raise InvalidAnswer(_summary(e)) from None
    return shapes, msg.split("```")[0].strip()


def _summary(e: ValidationError, limit: int = 3) -> str:
    """Return the first ``limit`` validation errors on one line, like "shapes.2.c: ..."."""
    errors = e.errors(include_url=False)
    parts = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in errors[:limit]]
    more = f" (+{len(errors) - limit} more)" if len(errors) > limit else ""
    return "invalid shapes: " + "; ".join(parts) + more


async def process(
    http: httpx2.AsyncClient,
    renderer: Renderer,
    model: str,
    rounds: int,
    workdir: Path,
    name: str,
    svg: Path,
    b_els: list[ex.Element],
) -> tuple[ex.LibraryItem | None, float]:
    """Run all rounds for one icon.

    Args:
        http: Client from ``client()``.
        renderer: Started renderer for the PNGs of the drawings.
        model: OpenRouter model id.
        rounds: Number of model calls.
        workdir: Output directory for the PNGs and the final shapes JSON.
        name: Display name of the icon.
        svg: Path of the official SVG.
        b_els: Approach B elements of the icon.

    Returns:
        A tuple (library item or None if no round gave valid shapes, cost in USD).
    """
    fn = re.sub(r"\W+", "_", name)  # file name stem
    orig = workdir / f"{fn}_orig.png"
    await asyncio.to_thread(
        subprocess.run,
        ["rsvg-convert", "-w", "320", "-h", "320", "-b", "white", "-o", orig, svg],
        check=True,
    )
    b_png = workdir / f"{fn}_b.png"
    await renderer.render_icon(b_els, b_png)
    shapes = from_excalidraw(b_els)
    best: list[Shape] | None = None
    cur_png = b_png
    cost_sum = 0.0
    for rnd in range(1, rounds + 1):
        if best is None:  # no valid answer yet: start from approach B
            text, imgs = ROUND1.format(name=name, shapes=compact.dump(shapes)), [orig, b_png]
        else:
            text, imgs = ROUND2.format(shapes=compact.dump(best)), [orig, cur_png]
        msg, cost = await ask(http, model, text, imgs, name)
        cost_sum += cost
        log.debug("%s round %d answer: %s", name, rnd, msg)
        try:
            new, note = parse(msg)
        except InvalidAnswer as e:
            log.warning("%s round %d: invalid answer: %s", name, rnd, e)
            continue
        png = workdir / f"{fn}_r{rnd}.png"
        try:
            await renderer.render_icon(to_excalidraw(new, name), png)
        except RenderError as e:  # never publish a drawing that Excalidraw cannot draw
            log.warning("%s round %d: answer dropped, render failed: %s", name, rnd, e)
            continue
        best, cur_png = new, png
        log.info("%s round %d: %d shapes. note: %s", name, rnd, len(best), note[:300])
    if best is None:
        return None, cost_sum
    (workdir / f"{fn}_shapes.json").write_text(compact.dump(best))
    return ex.item(f"c-{fn}", name, to_excalidraw(best, name)), cost_sum
