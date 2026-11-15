"""Render Excalidraw elements to PNG with headless Chrome (Excalidraw's own exportToSvg).

One browser is started for the whole build (``Renderer``); every render uses a new page in
the same browser context, so Excalidraw is downloaded from esm.sh only once and then comes
from the browser cache. Rendering needs network access for that first download.

The browser is the installed Chrome or Chromium named by the CHROME_BIN environment
variable (default: chromium), started through Playwright. Playwright's own browser
download is not needed.
"""

import json
import os
import shutil
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

CHROME = os.environ.get("CHROME_BIN", "chromium")
TIMEOUT_MS = 90_000  # per render; the first one downloads Excalidraw

PAGE = """<!doctype html><html><head><meta charset=utf-8><style>
html,body{margin:0;background:#fff}#c svg{width:%dpx;height:%dpx;display:block}</style></head>
<body><div id=c></div><script type=module>
import * as X from "https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@18.2.0,react-dom@18.2.0";
const exportToSvg = X.exportToSvg || X.default.exportToSvg;
const els = %s;
try {
  const svg = await exportToSvg({elements: els,
    appState: {exportBackground: true, viewBackgroundColor: '#fff', exportPadding: 0}, files: {}});
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  document.getElementById('c').appendChild(svg); document.title = 'done';
} catch (e) { document.title = 'error: ' + e; }
</script></body></html>"""
# The page sets its title to "done", or to "error: ..." so that a failure is reported at once.


class RenderError(Exception):
    """The page did not finish drawing: a JavaScript error or a timeout."""


class Renderer:
    """A headless browser that renders Excalidraw elements. Use it with ``async with``."""

    def __init__(self, chrome: str = CHROME) -> None:
        """Prepare a renderer for the browser binary ``chrome`` (a name on PATH or a path)."""
        self._chrome = chrome
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> Self:
        """Start the browser.

        Raises:
            SystemExit: If the browser binary is not found.
        """
        executable = shutil.which(self._chrome)
        if executable is None:
            raise SystemExit(f"browser not found: {self._chrome!r} (set CHROME_BIN)")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            executable_path=executable, args=["--disable-gpu"]
        )
        self._context = await self._browser.new_context()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the browser."""
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = self._browser = self._context = None

    async def render(self, elements: list[dict[str, Any]], out_png: Path, size: int = 320) -> None:
        """Render elements to a size x size PNG, scaled to fit, on a white background.

        Raises:
            RenderError: If the page has a JavaScript error or does not finish in time.
        """
        if self._context is None:
            raise RuntimeError("use the Renderer with `async with`")
        page = await self._context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            await page.set_viewport_size({"width": size, "height": size})
            await page.set_content(PAGE % (size, size, json.dumps(elements)))
            await page.wait_for_function("document.title !== ''", timeout=TIMEOUT_MS)
            if (title := await page.title()) != "done":
                raise RenderError(f"{out_png.name}: {title}")
            await page.screenshot(
                path=out_png, clip={"x": 0, "y": 0, "width": size, "height": size}
            )
        except PlaywrightError as e:
            detail = "; ".join(errors) or str(e).splitlines()[0]
            raise RenderError(f"{out_png.name}: {detail}") from e
        finally:
            await page.close()

    async def render_icon(
        self, elements: list[dict[str, Any]], out_png: Path, size: int = 320, box: int = 120
    ) -> None:
        """Render the icon without its label, in the frame of the original icon.

        An invisible box x box rectangle at the origin is added, so the image is not
        auto-cropped to the drawing and uses the same frame as the original icon.

        Args:
            elements: Excalidraw elements of the icon. Text elements are dropped.
            out_png: Path of the PNG to write.
            size: Width and height of the PNG in px.
            box: Size of the icon box in Excalidraw px.

        Raises:
            RenderError: If the page has a JavaScript error or does not finish in time.
        """
        await self.render(_framed(elements, box), out_png, size)


def _framed(elements: list[dict[str, Any]], box: int) -> list[dict[str, Any]]:
    """Drop the text elements and add an invisible box x box frame at the origin."""
    els = [e for e in elements if e.get("type") != "text"]
    if not els:
        return els
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
    return [frame, *els]
