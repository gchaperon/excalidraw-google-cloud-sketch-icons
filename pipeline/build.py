"""Download Google's icon zips, detect changed icons, rebuild only those, and write the libraries.

Usage:
    python -m pipeline.build              # incremental: process new/changed icons only
    python -m pipeline.build --check      # only report changes, do not call the model
    python -m pipeline.build --force      # reprocess every icon

The previous state (the libraries and sources.lock.json, which holds the sha256 of every
source SVG) is downloaded from the latest GitHub release of the repository in config.json.
Nothing is written to the source tree: all outputs go to dist/ (release files and notes.md)
and build/ (intermediate images and logs). The original Google files are downloaded to a
temporary directory and are never stored.

In GitHub Actions, writes `changed=true|false` to $GITHUB_OUTPUT.
"""

import argparse
import functools
import hashlib
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import convert, redraw
from . import excalidraw as ex

REPO = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "config.json"
DIST = REPO / "dist"  # release files
WORK = REPO / "build"  # intermediate images and logs
LOCK_FILE = "sources.lock.json"
UA = {"User-Agent": "excalidraw-google-cloud-sketch-icons"}

# Not __name__: under `python -m pipeline.build` that is "__main__", outside "pipeline".
log = logging.getLogger("pipeline.build")


def sha256(path):
    """Return the hex SHA-256 digest of a file."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, timeout=120):
    """Download a URL.

    Args:
        url: The URL to get.
        timeout: Timeout in seconds.

    Returns:
        The response body, or None if the server answers 404.

    Raises:
        urllib.error.HTTPError: For every HTTP error other than 404.
    """
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def previous_release(cfg):
    """Download the files of the latest release.

    Args:
        cfg: The parsed config.json.

    Returns:
        A dict {file name: bytes}, with None for each file that the release does not have
        (or for every file, if there is no release yet).
    """
    base = f"https://github.com/{cfg['repo']}/releases/latest/download/"
    names = [LOCK_FILE, *(lib["file"] for lib in cfg["libraries"])]
    return {n: download(base + n) for n in names}


def fetch_icons(lib, tmp):
    """Download and extract the icon zip of one library.

    The zip holds one folder per icon, and each folder holds an ``SVG`` folder with exactly
    one SVG file. Folders that do not match this layout are skipped with a warning.

    Args:
        lib: One entry of ``libraries`` in config.json.
        tmp: Directory for the zip and its extracted files.

    Returns:
        A dict {display name: {"folder": ..., "svg": path, "sha256": ...}}. The display
        name is ``lib["names"][folder]`` if it exists, else the folder name.

    Raises:
        SystemExit: If the zip is not found, has an unsafe member path, or has no icons.
    """
    data = download(lib["zip"])
    if data is None:
        raise SystemExit(f"not found: {lib['zip']}")
    zip_path = Path(tmp) / f"{lib['id']}.zip"
    zip_path.write_bytes(data)
    extract_dir = Path(tmp) / lib["id"]
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():  # refuse path traversal
            if Path(member).is_absolute() or ".." in member.split("/"):
                raise SystemExit(f"unsafe path in {lib['zip']}: {member}")
        z.extractall(extract_dir)
    root = extract_dir / lib["root"]
    icons = {}
    for folder in sorted(p.name for p in root.iterdir()):
        svg_dir = root / folder / "SVG"
        if not svg_dir.is_dir():
            continue
        svgs = sorted(p for p in svg_dir.iterdir() if p.suffix.lower() == ".svg")
        if len(svgs) != 1:
            log.warning("%s/%s: expected 1 SVG, found %d; skipped", lib["id"], folder, len(svgs))
            continue
        name = lib["names"].get(folder, folder)
        icons[name] = {"folder": folder, "svg": str(svgs[0]), "sha256": sha256(svgs[0])}
    if not icons:
        raise SystemExit(f"no icons found in {lib['zip']} under {lib['root']!r}")
    return icons


def fallback_item(name, b_elements):
    """Make a library item from the approach B result, with the standard label.

    Used when the model step fails for an icon.
    """
    els = list(b_elements)
    group = els[0]["groupIds"][0] if els else "g-fallback"
    return ex.item(f"b-{name}", name, [*els, ex.label(name, group)])


def process_icon(cfg, key, workdir, name, svg):
    """Build the library item of one icon: approach B, then the model rounds (approach C).

    Args:
        cfg: The parsed config.json.
        key: The OpenRouter API key.
        workdir: Directory for the intermediate images.
        name: Display name of the icon.
        svg: Path of the source SVG.

    Returns:
        A tuple (item, cost in USD, True if the model failed and the item is the fallback).
    """
    b_elements = convert.convert(Path(svg).read_text())
    try:
        item, cost = redraw.process(
            cfg["model"], cfg["rounds"], key, workdir, name, svg, b_elements
        )
    except Exception:  # network or render failure: keep going with approach B
        log.exception("%s: model step failed", name)
        item, cost = None, 0.0
    return (item or fallback_item(name, b_elements)), cost, item is None


def release_notes(notes, cfg, total_cost):
    """Return the Markdown body of the release."""
    return (
        "Rebuilt from the current [Google Cloud icons](https://cloud.google.com/icons).\n\n"
        + "\n".join(notes)
        + f"\nModel: `{cfg['model']}`, cost ${total_cost:.2f}.\n\n"
        + "`sources.lock.json` lists the SHA-256 of every source icon."
        + " The pipeline uses it to detect changes.\n"
    )


def setup_logging():
    """Log INFO to stderr and DEBUG to build/log.txt. Returns the handlers to remove later."""
    WORK.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logfile = logging.FileHandler(WORK / "log.txt")
    logfile.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s")
    )
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    for h in (console, logfile):
        logger.addHandler(h)
    return console, logfile


def main():
    """Run the build. See the module docstring for the command line."""
    handlers = setup_logging()
    try:
        build()
    finally:
        logger = logging.getLogger("pipeline")
        for h in handlers:
            logger.removeHandler(h)
            h.close()


def build():
    """Parse the command line and run the build."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="only report changes")
    g.add_argument("--force", action="store_true", help="reprocess every icon")
    a = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    prev = previous_release(cfg)
    lock = json.loads(prev[LOCK_FILE]) if prev[LOCK_FILE] else {}
    if not lock:
        log.info("no previous release state found: every icon counts as new")
    notes, outputs, any_change, total_cost = [], {}, False, 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for lib in cfg["libraries"]:
            icons = fetch_icons(lib, tmp)
            old = lock.get(lib["id"], {})
            prev_lib = prev[lib["file"]]
            items = {i["name"]: i for i in json.loads(prev_lib)["libraryItems"]} if prev_lib else {}
            # An icon also counts as added if the lock file has it but the library does not.
            added = sorted(n for n in icons if n not in old or n not in items)
            changed = sorted(
                n
                for n in icons
                if n in old and n in items and old[n]["sha256"] != icons[n]["sha256"]
            )
            removed = sorted(n for n in old if n not in icons)
            todo = sorted(icons) if a.force else added + changed
            log.info(
                "%s: %d icons, added %s, changed %s, removed %s",
                lib["id"],
                len(icons),
                added,
                changed,
                removed,
            )
            any_change |= bool(todo or removed)
            if a.check:
                continue
            lock[lib["id"]] = {
                n: {"folder": i["folder"], "sha256": i["sha256"]} for n, i in sorted(icons.items())
            }
            fallbacks = []
            if todo:
                workdir = WORK / lib["id"]
                workdir.mkdir(parents=True, exist_ok=True)
                key = redraw.key()
                run = functools.partial(process_icon, cfg, key, workdir)
                with ThreadPoolExecutor(cfg["workers"], thread_name_prefix="icon") as pool:
                    svgs = [icons[n]["svg"] for n in todo]
                    results = dict(zip(todo, pool.map(run, todo, svgs), strict=True))
                for n, (item, cost, failed) in results.items():
                    items[n] = item
                    total_cost += cost
                    if failed:
                        fallbacks.append(n)
            outputs[lib["file"]] = ex.library(
                [items[n] for n in sorted(icons, key=str.lower) if n in items],
                source=f"https://github.com/{cfg['repo']}",
            )
            sections = (
                ("Added", added),
                ("Changed", changed),
                ("Removed", removed),
                ("Rebuilt", sorted(set(todo) - set(added) - set(changed))),
                ("Fallback to deterministic conversion (model failed)", fallbacks),
            )
            lines = [f"- {title}: {', '.join(ns)}\n" for title, ns in sections if ns]
            notes.append(f"### {lib['title']}\n" + ("".join(lines) or "- No changes\n"))

    if not a.check:
        DIST.mkdir(parents=True, exist_ok=True)
        for file_name, lib_json in outputs.items():
            (DIST / file_name).write_text(json.dumps(lib_json))
        (DIST / LOCK_FILE).write_text(json.dumps(lock, indent=1, ensure_ascii=False))
        (DIST / "notes.md").write_text(release_notes(notes, cfg, total_cost))
    log.info("changed=%s cost=$%.2f", str(any_change).lower(), total_cost)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as f:
            f.write(f"changed={str(any_change).lower()}\n")


if __name__ == "__main__":
    main()
