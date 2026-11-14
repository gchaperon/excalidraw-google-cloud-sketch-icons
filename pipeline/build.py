"""Download Google's icon zips, detect changed icons, rebuild only those, and write the libraries.

Usage:
  python -m pipeline.build              # incremental: process new/changed icons only
  python -m pipeline.build --check      # only report changes, do not call the model
  python -m pipeline.build --force      # reprocess every icon
  python -m pipeline.build --adopt      # record current source hashes as already built (no processing)

State is kept in sources.lock.json (sha256 of every source SVG). The original Google files are
downloaded to a temporary directory and are never written to the repository.

In GitHub Actions, writes `changed=true|false` to $GITHUB_OUTPUT and release notes to build/notes.md.
"""
import argparse, hashlib, json, os, sys, tempfile, time, types, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

from . import convert, redraw
from .compact import label

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG = os.path.join(os.path.dirname(__file__), 'config.json')
LOCK = os.path.join(REPO, 'sources.lock.json')
WORK = os.path.join(REPO, 'build')


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def fetch_icons(lib, tmp):
    """Download and extract one zip. Returns {display name: {folder, svg path, sha256}}."""
    zpath = os.path.join(tmp, lib['id'] + '.zip')
    req = urllib.request.Request(lib['zip'], headers={'User-Agent': 'excalidraw-google-cloud-sketch-icons'})
    with urllib.request.urlopen(req, timeout=120) as r, open(zpath, 'wb') as f:
        f.write(r.read())
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


def load_items(path):
    if not os.path.exists(path):
        return {}
    return {i['name']: i for i in json.load(open(path))['libraryItems']}


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
    g.add_argument('--adopt', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    lock = json.load(open(LOCK)) if os.path.exists(LOCK) else {}
    notes, any_change, total_cost = [], False, 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for lib in cfg['libraries']:
            icons = fetch_icons(lib, tmp)
            old = lock.get(lib['id'], {})
            added = sorted(n for n in icons if n not in old)
            changed = sorted(n for n in icons if n in old and old[n]['sha256'] != icons[n]['sha256'])
            removed = sorted(n for n in old if n not in icons)
            todo = sorted(icons) if a.force else added + changed
            print(f'{lib["id"]}: {len(icons)} icons, added {added}, changed {changed}, removed {removed}')
            if a.check:
                any_change |= bool(added or changed or removed); continue
            lock[lib['id']] = {n: {'folder': i['folder'], 'sha256': i['sha256']} for n, i in sorted(icons.items())}
            if a.adopt or not (todo or removed):
                continue
            any_change = True
            out = os.path.join(REPO, lib['out'])
            items = load_items(out)
            workdir = os.path.join(WORK, lib['id']); os.makedirs(workdir, exist_ok=True)
            key = redraw.key()
            with open(os.path.join(workdir, 'log.txt'), 'a') as log, ThreadPoolExecutor(cfg['workers']) as ex:
                results = dict(zip(todo, ex.map(lambda n: process_icon(cfg, key, workdir, n, icons[n]['svg'], log), todo)))
            fallbacks = [n for n, r in results.items() if r[2]]
            for n, (item, cost, _) in results.items():
                items[n] = item; total_cost += cost
            items = {n: items[n] for n in sorted(icons, key=str.lower) if n in items}
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({'type': 'excalidrawlib', 'version': 2,
                       'source': 'https://github.com/gchaperon/excalidraw-google-cloud-sketch-icons',
                       'libraryItems': list(items.values())}, open(out, 'w'))
            notes.append(f'### {lib["title"]}\n'
                         + ''.join(f'- {label_}: {", ".join(ns)}\n' for label_, ns in
                                   (('Added', added), ('Changed', changed), ('Removed', removed),
                                    ('Rebuilt', sorted(set(todo) - set(added) - set(changed))),
                                    ('Fallback to deterministic conversion (model failed)', fallbacks)) if ns))

    if not a.check:
        json.dump(lock, open(LOCK, 'w'), indent=1, ensure_ascii=False); open(LOCK, 'a').write('\n')
    if notes:
        os.makedirs(WORK, exist_ok=True)
        open(os.path.join(WORK, 'notes.md'), 'w').write(
            'Rebuilt from the current [Google Cloud icons](https://cloud.google.com/icons).\n\n' + '\n'.join(notes)
            + f'\nModel: `{cfg["model"]}`, cost ${total_cost:.2f}.\n')
    print(f'changed={str(any_change).lower()} cost=${total_cost:.2f}')
    if os.environ.get('GITHUB_OUTPUT'):
        open(os.environ['GITHUB_OUTPUT'], 'a').write(f'changed={str(any_change).lower()}\n')


if __name__ == '__main__':
    main()
