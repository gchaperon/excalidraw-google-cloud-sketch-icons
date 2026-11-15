import json

import pytest

from pipeline import config

VALID = {
    "repo": "owner/repo",
    "model": "some/model",
    "rounds": 3,
    "workers": 2,
    "libraries": [
        {"id": "a", "title": "A", "zip": "https://x/a.zip", "root": "Icons", "file": "a.lib"}
    ],
}


def write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def test_the_repository_config_is_valid():
    cfg = config.load()

    assert cfg.libraries
    assert len({lib.id for lib in cfg.libraries}) == len(cfg.libraries)


def test_load_valid(tmp_path):
    cfg = config.load(write(tmp_path, VALID))

    assert cfg.rounds == 3
    assert cfg.libraries[0].names == {}  # optional


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"rounds": "3"}, "rounds"),  # wrong type
        ({"rounds": True}, "rounds"),  # bool is not an int here
        ({"workers": 0}, "workers"),  # must be positive
        ({"modle": "x"}, "modle"),  # unknown key (a typo)
        ({"libraries": [{"id": "a"}]}, "title"),  # missing key in a library
    ],
)
def test_load_rejects_invalid_config(tmp_path, change, error):
    with pytest.raises(SystemExit, match=error):
        config.load(write(tmp_path, VALID | change))


def test_load_rejects_missing_key(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "model"}

    with pytest.raises(SystemExit, match="model"):
        config.load(write(tmp_path, data))
