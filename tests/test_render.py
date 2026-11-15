import asyncio

import numpy as np
import pytest
from PIL import Image

from pipeline import excalidraw as ex
from pipeline.render import Renderer, RenderError, _framed

from .conftest import needs_chrome


def test_framed_drops_text_and_adds_an_invisible_box():
    group = ex.new_group()
    line = ex.line([(10, 10), (50, 50)], "#4285f4", group, curved=False)

    frame, kept = _framed([line, ex.label("X", group)], box=120)

    assert kept is line
    assert (frame["type"], frame["x"], frame["y"], frame["width"]) == ("rectangle", 0, 0, 120)
    assert frame["strokeColor"] == "transparent"
    assert "points" not in frame


def test_framed_empty():
    assert _framed([ex.label("X", "g")], box=120) == []


@pytest.mark.anyio
async def test_render_needs_async_with(tmp_path):
    with pytest.raises(RuntimeError, match="async with"):
        await Renderer().render([], tmp_path / "x.png")


@pytest.mark.anyio
async def test_missing_browser_exits():
    with pytest.raises(SystemExit, match="browser not found"):
        async with Renderer("no-such-browser-binary"):
            pass


@needs_chrome
@pytest.mark.anyio
async def test_render_draws_the_icon(tmp_path):
    """Real browser and Excalidraw from esm.sh (needs network access)."""
    group = ex.new_group()
    els = [ex.element("ellipse", 30, 30, 60, 60, "#ea4335", group)]

    async with Renderer() as r:
        await asyncio.gather(
            r.render_icon(els, tmp_path / "a.png"), r.render_icon(els, tmp_path / "b.png")
        )
        with pytest.raises(RenderError, match=r"bad\.png: error: "):
            await r.render([{"type": "line", "points": None}], tmp_path / "bad.png")

    image = np.array(Image.open(tmp_path / "a.png").convert("RGB"))
    assert image.shape == (320, 320, 3)
    assert (image[5, 5] == 255).all()  # white background
    assert ((image[..., 0] > 200) & (image[..., 1] < 120)).any()  # the red circle is drawn
    assert (tmp_path / "b.png").read_bytes() == (tmp_path / "a.png").read_bytes()
