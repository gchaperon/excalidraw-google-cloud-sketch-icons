import json
import random
import shutil
from pathlib import Path

import httpx2
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

DATA = Path(__file__).parent / "data"

needs_rsvg = pytest.mark.skipif(
    shutil.which("rsvg-convert") is None, reason="rsvg-convert (librsvg) is not installed"
)


@pytest.fixture
def deterministic(monkeypatch):
    """Make element ids, seeds and timestamps repeatable."""
    random.seed(1234)
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)


@pytest.fixture
def snapshot_json(snapshot):
    """Snapshot that is stored as readable JSON."""
    return snapshot.use_extension(JSONSnapshotExtension)


def as_json(value):
    """Round-trip through JSON, as the build does when it writes the libraries.

    This also turns numpy scalars into plain Python numbers.
    """
    return json.loads(json.dumps(value))


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
