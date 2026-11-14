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
import argparse, hashlib, json, os, sys, tempfile, time, types, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

from . import convert, redraw
from .compact import label

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG = os.path.join(os.path.dirname(__file__), 'config.json')
DIST = os.path.join(REPO, 'dist')
WORK = os.path.join(REPO, 'build')
LOCK_FILE = 'sources.lock.json'
UA = {'User-Agent': 'excalidraw-google-cloud-sketch-icons'}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def download(url, timeout=120):
    """Return the body, or None if the URL does not exist (404)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def previous_release(cfg):
    """Files of the latest release: {file name: bytes or None}."""
    base = f'https://github.com/{cfg["repo"]}/releases/latest/download/'
    names = [LOCK_FILE] + [lib['file'] for lib in cfg['libraries']]
    return {n: download(base + n) for n in names}


def fetch_icons(lib, tmp):
    """Download and extract one zip. Returns {display name: {folder, svg path, sha256}}."""
    data = download(lib['zip'])
    if data is None:
        raise SystemExit(f'not found: {lib["zip"]}')
    zpath = os.path.join(tmp, lib['id'] + '.zip')
    open(zpath, 'wb').write(data)
    xdir = os.path.join(tmp, lib['id'])
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():  # refuse path traversal
            if os.path.isabs(m) or '..' in m.split('/'):
                raise SystemExit(f'unsafe path in {lib["zip"]}: {m}')
        z.extractall(xdir)
    root = os.path.join(xdir, lib['root'])
    icons = {}
    for folder in sorted(os.listdir(root)):
        svg_dir = os.path.join(root, folder, 'SVG')
        if not os.path.isdir(svg_dir):
            continue
        svgs = sorted(f for f in os.listdir(svg_dir) if f.lower().endswith('.svg'))
        if len(svgs) != 1:
            print(f'warning: {lib["id"]}/{folder}: expected 1 SVG, found {len(svgs)}; skipped', file=sys.stderr)
            continue
        path = os.path.join(svg_dir, svgs[0])
        name = lib['names'].get(folder, folder)
        icons[name] = {'folder': folder, 'svg': path, 'sha256': sha256(path)}
    if not icons:
        raise SystemExit(f'no icons found in {lib["zip"]} under {lib["root"]!r}')
    return icons


def fallback_item(name, b_elements):
    """Approach B result with the standard label, used when the model fails for an icon."""
    els = [e for e in b_elements if e['type'] != 'text']
    group = els[0]['groupIds'][0] if els else 'g-fallback'
    return {'id': f'b-{name}', 'status': 'unpublished', 'created': int(time.time() * 1000),
            'name': name, 'elements': els + [label(name, group)]}


def process_icon(cfg, key, workdir, name, svg, log):
    b_elements = convert.convert(open(svg).read(), name)
    args = types.SimpleNamespace(model=cfg['model'], rounds=cfg['rounds'])
    try:
        item, cost, _, _ = redraw.process(args, key, workdir, name, svg, b_elements, log)
    except Exception as e:  # network or render failure: keep going with approach B
        log.write(f'{name}: model step failed: {e!r}\n'); item, cost = None, 0.0
    return (item or fallback_item(name, b_elements)), cost, item is None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--check', action='store_true'); g.add_argument('--force', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    prev = previous_release(cfg)
    lock = json.loads(prev[LOCK_FILE]) if prev[LOCK_FILE] else {}
    if not lock:
        print('no previous release state found: every icon counts as new')
    notes, outputs, any_change, total_cost = [], {}, False, 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for lib in cfg['libraries']:
            icons = fetch_icons(lib, tmp)
            old = lock.get(lib['id'], {})
            items = {i['name']: i for i in json.loads(prev[lib['file']])['libraryItems']} if prev[lib['file']] else {}
            added = sorted(n for n in icons if n not in old or n not in items)
            changed = sorted(n for n in icons if n in old and n in items and old[n]['sha256'] != icons[n]['sha256'])
            removed = sorted(n for n in old if n not in icons)
            todo = sorted(icons) if a.force else added + changed
            print(f'{lib["id"]}: {len(icons)} icons, added {added}, changed {changed}, removed {removed}')
            any_change |= bool(todo or removed)
            if a.check:
                continue
            lock[lib['id']] = {n: {'folder': i['folder'], 'sha256': i['sha256']} for n, i in sorted(icons.items())}
            fallbacks = []
            if todo:
                workdir = os.path.join(WORK, lib['id']); os.makedirs(workdir, exist_ok=True)
                key = redraw.key()
                with open(os.path.join(workdir, 'log.txt'), 'a') as log, ThreadPoolExecutor(cfg['workers']) as ex:
                    results = dict(zip(todo, ex.map(lambda n: process_icon(cfg, key, workdir, n, icons[n]['svg'], log), todo)))
                for n, (item, cost, failed) in results.items():
                    items[n] = item; total_cost += cost
                    if failed:
                        fallbacks.append(n)
            outputs[lib['file']] = {
                'type': 'excalidrawlib', 'version': 2, 'source': f'https://github.com/{cfg["repo"]}',
                'libraryItems': [items[n] for n in sorted(icons, key=str.lower) if n in items]}
            lines = [f'- {t}: {", ".join(ns)}\n' for t, ns in (
                ('Added', added), ('Changed', changed), ('Removed', removed),
                ('Rebuilt', sorted(set(todo) - set(added) - set(changed))),
                ('Fallback to deterministic conversion (model failed)', fallbacks)) if ns]
            notes.append(f'### {lib["title"]}\n' + (''.join(lines) or '- No changes\n'))

    if not a.check:
        os.makedirs(DIST, exist_ok=True)
        for f, lib_json in outputs.items():
            json.dump(lib_json, open(os.path.join(DIST, f), 'w'))
        json.dump(lock, open(os.path.join(DIST, LOCK_FILE), 'w'), indent=1, ensure_ascii=False)
        open(os.path.join(DIST, 'notes.md'), 'w').write(
            'Rebuilt from the current [Google Cloud icons](https://cloud.google.com/icons).\n\n' + '\n'.join(notes)
            + f'\nModel: `{cfg["model"]}`, cost ${total_cost:.2f}.\n\n'
            + '`sources.lock.json` lists the SHA-256 of every source icon. The pipeline uses it to detect changes.\n')
    print(f'changed={str(any_change).lower()} cost=${total_cost:.2f}')
    if os.environ.get('GITHUB_OUTPUT'):
        open(os.environ['GITHUB_OUTPUT'], 'a').write(f'changed={str(any_change).lower()}\n')


if __name__ == '__main__':
    main()
