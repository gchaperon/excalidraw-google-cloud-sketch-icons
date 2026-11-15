import json
import random
import shutil
from pathlib import Path
from typing import override

import httpx2
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from pipeline.render import CHROME, Renderer, RenderError

DATA = Path(__file__).parent / "data"

needs_rsvg = pytest.mark.skipif(
    shutil.which("rsvg-convert") is None, reason="rsvg-convert (librsvg) is not installed"
)
needs_chrome = pytest.mark.skipif(
    shutil.which(CHROME) is None, reason=f"browser {CHROME!r} is not installed (CHROME_BIN)"
)


class FakeRenderer(Renderer):
    """A Renderer without a browser: it writes a placeholder PNG and records the calls.

    ``render_icon`` is inherited, so the framing code still runs.
    """

    def __init__(self, fail=()):
        super().__init__()
        self.calls = []
        self.fail = set(fail)  # PNG names whose render fails

    @override
    async def __aenter__(self):
        return self

    @override
    async def __aexit__(self, exc_type, exc, tb):
        pass

    @override
    async def render(self, elements, out_png, size=320):
        self.calls.append((out_png.name, elements))
        if out_png.name in self.fail:
            raise RenderError(f"{out_png.name}: fake failure")
        out_png.write_bytes(b"png")


@pytest.fixture
def deterministic(monkeypatch):
    """Make element ids, seeds and timestamps repeatable."""
    random.seed(1234)
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)


@pytest.fixture
def snapshot_json(snapshot):
    """Snapshot that is stored as readable JSON."""
    return snapshot.use_extension(JSONSnapshotExtension)


def as_json(value, digits=6):
    """Round-trip through JSON, as the build does when it writes the libraries.

    This also turns numpy scalars into plain Python numbers. Floats are rounded to
    ``digits`` decimals: numpy's least-squares fit can differ in the last bits on
    different CPUs, and a snapshot must not depend on the machine.
    """
    return _round_floats(json.loads(json.dumps(value)), digits)


def _round_floats(value, digits):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_floats(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, digits) for k, v in value.items()}
    return value


@pytest.fixture
def anyio_backend():
    """Run the async tests (marked with ``pytest.mark.anyio``) on asyncio only."""
    return "asyncio"


def mock_client(handler, **kwargs):
    """Return an AsyncClient whose requests go to ``handler(request) -> Response``."""
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)


def serve(files):
    """Return a request handler that serves ``files`` {url: bytes} and 404 for other URLs."""

    def handler(request):
        data = files.get(str(request.url))
        return httpx2.Response(200, content=data) if data is not None else httpx2.Response(404)

    return handler
