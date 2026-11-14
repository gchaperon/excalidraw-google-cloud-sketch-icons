import json
import random
import shutil
from pathlib import Path

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from pipeline import convert

DATA = Path(__file__).parent / "data"

needs_rsvg = pytest.mark.skipif(
    shutil.which("rsvg-convert") is None, reason="rsvg-convert (librsvg) is not installed"
)


@pytest.fixture
def deterministic(monkeypatch):
    """Make element ids, seeds and timestamps repeatable."""
    random.seed(1234)
    monkeypatch.setattr(convert, "_ids", 0)
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
