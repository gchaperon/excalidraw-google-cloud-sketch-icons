import hashlib
import io
import json
import sys
import zipfile

import pytest

from pipeline import build

ZIP_URL = "https://icons.example/lib.zip"
RELEASE = "https://github.com/owner/repo/releases/latest/download/"
CONFIG = {
    "repo": "owner/repo",
    "model": "some/model",
    "rounds": 1,
    "workers": 2,
    "libraries": [
        {
            "id": "lib",
            "title": "Test library",
            "zip": ZIP_URL,
            "root": "Icons",
            "names": {"Folder C": "Nice Name"},
            "file": "lib.excalidrawlib",
        }
    ],
}
SVGS = {"A": "<svg>a</svg>", "B": "<svg>b-new</svg>", "Folder C": "<svg>c</svg>"}


def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def icons_zip():
    members = {f"Icons/{folder}/SVG/{folder}.svg": svg for folder, svg in SVGS.items()}
    members["Icons/Two SVGs/SVG/x.svg"] = members["Icons/Two SVGs/SVG/y.svg"] = "<svg/>"
    members["Icons/No SVG folder/x.svg"] = "<svg/>"
    members["Icons/readme.txt"] = "not an icon folder"
    return make_zip(members)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def item(name, origin):
    return {
        "id": f"{origin}-{name}",
        "name": name,
        "elements": [{"type": "line"}],
        "origin": origin,
    }


def library(*items):
    return json.dumps({"type": "excalidrawlib", "version": 2, "libraryItems": list(items)})


# --- fetch_icons ---------------------------------------------------------------------------


def test_fetch_icons_reads_one_svg_per_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build, "download", lambda url: icons_zip())

    icons = build.fetch_icons(CONFIG["libraries"][0], tmp_path)

    assert sorted(icons) == ["A", "B", "Nice Name"]
    assert icons["Nice Name"]["folder"] == "Folder C"
    assert icons["A"]["sha256"] == sha(SVGS["A"])
    assert "Two SVGs: expected 1 SVG, found 2; skipped" in capsys.readouterr().err


@pytest.mark.parametrize("member", ["../evil.svg", "Icons/../../evil.svg", "/abs/evil.svg"])
def test_fetch_icons_refuses_unsafe_paths(monkeypatch, tmp_path, member):
    monkeypatch.setattr(build, "download", lambda url: make_zip({member: "x"}))

    with pytest.raises(SystemExit, match="unsafe path"):
        build.fetch_icons(CONFIG["libraries"][0], tmp_path)


def test_fetch_icons_missing_zip(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "download", lambda url: None)

    with pytest.raises(SystemExit, match="not found"):
        build.fetch_icons(CONFIG["libraries"][0], tmp_path)


# --- main ----------------------------------------------------------------------------------


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run build.main() offline. The model fails for icon "B", so B uses the fallback."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    github_output = tmp_path / "github_output"
    monkeypatch.setattr(build, "CONFIG", config)
    monkeypatch.setattr(build, "DIST", tmp_path / "dist")
    monkeypatch.setattr(build, "WORK", tmp_path / "build")
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(build.redraw, "key", lambda: "k")
    monkeypatch.setattr(build.convert, "convert", lambda svg: [{"type": "line", "groupIds": ["g"]}])
    processed = []

    def fake_process(args, key, workdir, name, svg, b_elements, log):
        processed.append(name)
        if name == "B":
            raise ConnectionError("model down")
        return item(name, "model"), 0.5

    monkeypatch.setattr(build.redraw, "process", fake_process)

    def main(release, *args):
        files = {ZIP_URL: icons_zip()} | {RELEASE + n: data for n, data in release.items()}
        monkeypatch.setattr(build, "download", lambda url: files.get(url))
        monkeypatch.setattr(sys, "argv", ["build", *args])
        build.main()
        dist = tmp_path / "dist"
        outputs = {p.name: p.read_text() for p in dist.iterdir()} if dist.exists() else {}
        return outputs, github_output.read_text(), sorted(processed)

    return main


PREVIOUS = {
    build.LOCK_FILE: json.dumps(
        {
            "lib": {
                "A": {"folder": "A", "sha256": sha(SVGS["A"])},
                "B": {"folder": "B", "sha256": sha("<svg>b-old</svg>")},
                "Gone": {"folder": "Gone", "sha256": sha("<svg>gone</svg>")},
            }
        }
    ),
    "lib.excalidrawlib": library(item("A", "old"), item("B", "old"), item("Gone", "old")),
}


def test_incremental_build(run, snapshot):
    outputs, github_output, processed = run(PREVIOUS)

    assert processed == ["B", "Nice Name"]  # A did not change
    lib = json.loads(outputs["lib.excalidrawlib"])
    assert lib["source"] == "https://github.com/owner/repo"
    assert [(i["name"], i["id"]) for i in lib["libraryItems"]] == [
        ("A", "old-A"),  # unchanged: kept from the previous release
        ("B", "b-B"),  # model failed: approach B fallback with the label
        ("Nice Name", "model-Nice Name"),
    ]
    assert [e["type"] for e in lib["libraryItems"][1]["elements"]] == ["line", "text"]
    lock = json.loads(outputs[build.LOCK_FILE])
    assert lock == {
        "lib": {
            "A": {"folder": "A", "sha256": sha(SVGS["A"])},
            "B": {"folder": "B", "sha256": sha(SVGS["B"])},
            "Nice Name": {"folder": "Folder C", "sha256": sha(SVGS["Folder C"])},
        }
    }
    assert outputs["notes.md"] == snapshot
    assert github_output == "changed=true\n"


def test_check_writes_nothing(run):
    outputs, github_output, processed = run(PREVIOUS, "--check")

    assert (outputs, processed) == ({}, [])
    assert github_output == "changed=true\n"


def test_force_rebuilds_every_icon(run):
    outputs, _, processed = run(PREVIOUS, "--force")

    assert processed == ["A", "B", "Nice Name"]
    assert "- Rebuilt: A\n" in outputs["notes.md"]


def test_no_change(run):
    names = {"A": "A", "B": "B", "Folder C": "Nice Name"}
    lock = {"lib": {names[f]: {"folder": f, "sha256": sha(svg)} for f, svg in SVGS.items()}}
    previous = {
        build.LOCK_FILE: json.dumps(lock),
        "lib.excalidrawlib": library(*(item(n, "old") for n in ["A", "B", "Nice Name"])),
    }

    outputs, github_output, processed = run(previous)

    assert processed == []
    assert "- No changes\n" in outputs["notes.md"]
    assert github_output == "changed=false\n"


def test_first_build_without_release(run):
    outputs, _, processed = run({})

    assert processed == ["A", "B", "Nice Name"]
    assert "- Added: A, B, Nice Name\n" in outputs["notes.md"]
