"""Render Excalidraw elements to PNG with headless Chrome (Excalidraw's own exportToSvg).

The page loads Excalidraw from esm.sh, so rendering needs network access. The browser
binary comes from the CHROME_BIN environment variable (default: chromium). The `timeout`
command (coreutils) must be on PATH.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

CHROME = os.environ.get("CHROME_BIN", "chromium")

PAGE = """<!doctype html><html><head><meta charset=utf-8><style>
html,body{margin:0;background:#fff}#c svg{width:%dpx;height:%dpx;display:block}</style></head>
<body><div id=c></div><script type=module>
import * as X from "https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@18.2.0,react-dom@18.2.0";
const exportToSvg = X.exportToSvg || X.default.exportToSvg;
const els = %s;
const svg = await exportToSvg({elements: els,
  appState: {exportBackground: true, viewBackgroundColor: '#fff', exportPadding: 0}, files: {}});
svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
document.getElementById('c').appendChild(svg); document.title = 'done';
</script></body></html>"""


def render(elements, out_png, size=320):
    """Render elements to a size x size PNG, scaled to fit, on a white background.

    Chrome gets 90 seconds. Its errors are ignored: the result is only whether the PNG
    exists.

    Args:
        elements: Excalidraw elements.
        out_png: Path of the PNG to write.
        size: Width and height of the PNG in px.

    Returns:
        True if the PNG was written.
    """
    out_png = Path(out_png).resolve()
    with tempfile.TemporaryDirectory() as d:
        html = Path(d) / "r.html"
        html.write_text(PAGE % (size, size, json.dumps(elements)))
        # The window is taller than the image; the screenshot is cropped below.
        subprocess.run(
            [
                "timeout",
                "90",
                CHROME,
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={d}/profile",
                "--virtual-time-budget=20000",
                f"--window-size={size},{size + 200}",
                f"--screenshot={out_png}",
                html.as_uri(),
            ],
            capture_output=True,
            check=False,
        )
    if out_png.exists():
        Image.open(out_png).convert("RGB").crop((0, 0, size, size)).save(out_png)
        return True
    return False


def render_icon(elements, out_png, size=320, box=120):
    """Render the icon without its label, in the frame of the original icon.

    An invisible box x box rectangle at the origin is added, so the image is not
    auto-cropped to the drawing and uses the same frame as the original icon.

    Args:
        elements: Excalidraw elements of the icon. Text elements are dropped.
        out_png: Path of the PNG to write.
        size: Width and height of the PNG in px.
        box: Size of the icon box in Excalidraw px.

    Returns:
        True if the PNG was written.
    """
    els = [e for e in elements if e.get("type") != "text"]
    if els:
        frame = dict(
            els[0],
            id="frame-box",
            type="rectangle",
            x=0,
            y=0,
            width=box,
            height=box,
            strokeColor="transparent",
            backgroundColor="transparent",
            roundness=None,
            groupIds=[],
        )
        for k in (
            "points",
            "lastCommittedPoint",
            "startBinding",
            "endBinding",
            "startArrowhead",
            "endArrowhead",
        ):
            frame.pop(k, None)
        els = [frame, *els]
    return render(els, out_png, size)
