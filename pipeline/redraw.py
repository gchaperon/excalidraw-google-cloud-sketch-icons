"""Approach C: LLM redraws each icon in sketch style, starting from approach B.

Usage: python -m pipeline.redraw MODEL OUTDIR [--icons out/icons.json] [--blib out/b.excalidrawlib] [--rounds 3] [--workers 4]
Writes OUTDIR/<model>.excalidrawlib,
per-round PNGs and a log with tokens and cost (from OpenRouter usage accounting).
"""
import argparse, base64, json, os, re, subprocess, sys, time
import requests

from .compact import to_excalidraw, from_excalidraw
from .render import render_icon as render

ROOT = os.path.join(os.path.dirname(__file__), '..')
ENV = os.path.join(ROOT, '.env')

STYLE = """You convert Google Cloud product icons into hand-drawn Excalidraw sketch icons.

Target style (like the popular hand-drawn AWS Excalidraw libraries):
- Simple and recognizable: keep the characteristic silhouette and the main parts, drop tiny details.
- Few shapes: usually 4-15 shapes. Each Excalidraw shape is drawn with a rough hand-drawn stroke.
- Keep the original colors per part. Google palette: blue #4285f4, red #ea4335, yellow #fbbc04,
  green #34a853, grey #9aa0a6, light blue #aecbfa, mid blue #669df6.
- Strokes are lines of width 6 (do not draw outlines of thick strokes - draw the center line once).
- Use "curve": true for arcs and round parts (the line passes smoothly through the points;
  3-7 points are enough for an arc). Use "curve": false for straight segments and sharp corners.
  A curved line rounds every corner, so split a shape into separate lines at sharp corners.
- Use "ellipse" for circles and small filled dots (fill "solid").
- Use fill "hachure" only for larger filled areas, if the original has them.

Coordinate system: the icon box is 120 x 120 units, origin top-left, y grows downward.

Shape format (JSON):
  {"t":"line","c":"#4285f4","pts":[[x,y],...],"curve":true|false,"closed":false,"fill":"none"}
  {"t":"ellipse","c":"#ea4335","x":..,"y":..,"w":..,"h":..,"fill":"none|solid|hachure"}
  {"t":"rect","c":"#34a853","x":..,"y":..,"w":..,"h":..,"fill":"none","round":false}

Answer with a short note of what you changed (max 3 sentences), then ONE ```json code block
that contains {"shapes":[...]} and nothing else."""

ROUND1 = """Icon: {name}

Image 1 is the official icon. Image 2 is an automatic conversion of it (shapes below), drawn by Excalidraw.
The automatic conversion has the right proportions but it can have errors: missing parts, extra
short lines, wrong joins, rough or broken curves. Redraw the icon in the target style. Use the
conversion as a geometric guide, but fix everything that does not look like the official icon.

Automatic conversion shapes:
```json
{shapes}
```"""

ROUND2 = """Image 1 is the official icon. Image 2 is how Excalidraw draws your shapes (below).
Compare them carefully: silhouette, proportions, missing or extra parts, colors, curves, line ends.
If it already looks right, return the same shapes. Otherwise return corrected shapes.

Your shapes:
```json
{shapes}
```"""


def key():
    if os.environ.get('OPENROUTER_API_KEY'):
        return os.environ['OPENROUTER_API_KEY']
    for line in open(ENV) if os.path.exists(ENV) else []:
        if line.startswith('OPENROUTER_API_KEY='):
            return line.split('=', 1)[1].strip()
    raise SystemExit('OPENROUTER_API_KEY is not set (environment or .env)')


def png_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(open(path, 'rb').read()).decode()


def ask(model, k, text, images, log):
    content = [{'type': 'text', 'text': text}] + [{'type': 'image_url', 'image_url': {'url': png_uri(p)}} for p in images]
    body = {'model': model, 'max_tokens': 16000, 'usage': {'include': True}, 'reasoning': {'effort': 'medium'},
            'messages': [{'role': 'system', 'content': STYLE}, {'role': 'user', 'content': content}]}
    t0 = time.time()
    for attempt in range(3):
        r = requests.post('https://openrouter.ai/api/v1/chat/completions', timeout=600,
                          headers={'Authorization': f'Bearer {k}', 'X-Title': 'excalidraw-google-cloud-sketch-icons'}, json=body)
        if r.status_code == 200:
            break
        log.write(f'  HTTP {r.status_code}: {r.text[:300]}\n'); log.flush()
        time.sleep(5 * (attempt + 1))
    else:
        return None, 0.0, 0
    d = r.json()
    if 'choices' not in d:
        log.write(f'  no choices: {json.dumps(d)[:300]}\n'); return None, 0.0, 0
    msg = d['choices'][0]['message'].get('content') or ''
    u = d.get('usage', {})
    cost = float(u.get('cost') or 0)
    log.write(f'  {time.time() - t0:.0f}s in={u.get("prompt_tokens")} out={u.get("completion_tokens")} cost=${cost:.4f}\n')
    return msg, cost, time.time() - t0


def parse(msg):
    m = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', msg or '', re.S)
    if not m:
        return None, ''
    try:
        shapes = json.loads(m[-1])['shapes']
    except Exception:
        return None, ''
    note = (msg.split('```')[0]).strip()
    return shapes, note


def process(a, k, od, name, svg, b_els, log):
    """Run all rounds for one icon. Returns (item or None, cost, seconds, notes)."""
    fn = re.sub(r'\W+', '_', name)
    orig = os.path.join(od, f'{fn}_orig.png')
    subprocess.run(['rsvg-convert', '-w', '320', '-h', '320', '-b', 'white', '-o', orig, os.path.join(ROOT, svg)], check=True)
    b_png = os.path.join(od, f'{fn}_b.png')
    render(b_els, b_png)
    shapes = from_excalidraw(b_els)
    best, cur_png, cost_sum, secs, notes = None, None, 0.0, 0.0, []
    for rnd in range(1, a.rounds + 1):
        if rnd == 1:
            text, imgs = ROUND1.format(name=name, shapes=json.dumps(shapes)), [orig, b_png]
        else:
            text, imgs = ROUND2.format(shapes=json.dumps(best)), [orig, cur_png]
        buf = []
        class L:  # per-icon buffered log
            write = staticmethod(buf.append); flush = staticmethod(lambda: None)
        msg, cost, dt = ask(a.model, k, text, imgs, L)
        cost_sum += cost; secs += dt
        new, note = parse(msg)
        log.write(f'{name} round {rnd}\n' + ''.join(buf))
        if not new:
            log.write(f'  could not parse answer: {(msg or "")[:300]!r}\n'); log.flush(); continue
        best = new; notes.append(note)
        cur_png = os.path.join(od, f'{fn}_r{rnd}.png')
        render(to_excalidraw(best, name), cur_png)
        log.write(f'  {len(best)} shapes. note: {note[:300]}\n'); log.flush()
    item = None
    if best:
        item = {'id': f'c-{fn}', 'status': 'unpublished', 'created': int(time.time() * 1000),
                'name': name, 'elements': to_excalidraw(best, name)}
        json.dump(best, open(os.path.join(od, f'{fn}_shapes.json'), 'w'))
    return item, cost_sum, secs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('outdir')
    ap.add_argument('--icons', default='out/icons.json', help='JSON list of {name, svg}, relative to scratchpad')
    ap.add_argument('--blib', default='out/b.excalidrawlib', help='approach B library, relative to scratchpad')
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    k = key()
    od = os.path.join(a.outdir, a.model.replace('/', '_')); os.makedirs(od, exist_ok=True)
    log = open(os.path.join(od, 'log.txt'), 'w')
    icons = json.load(open(os.path.join(ROOT, a.icons)))
    blib = {i['name']: i['elements'] for i in json.load(open(os.path.join(ROOT, a.blib)))['libraryItems']}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda ic: process(a, k, od, ic['name'], ic['svg'], blib[ic['name']], log), icons))
    items = [r[0] for r in results if r[0]]  # keeps the order of the icons file
    total, secs = sum(r[1] for r in results), sum(r[2] for r in results)
    notes = {ic['name']: r[3] for ic, r in zip(icons, results)}
    json.dump({'type': 'excalidrawlib', 'version': 2, 'source': f'gcp-sketch-c:{a.model}', 'libraryItems': items},
              open(os.path.join(od, 'lib.excalidrawlib'), 'w'))
    json.dump({'model': a.model, 'cost': total, 'seconds': secs, 'icons': len(items), 'notes': notes},
              open(os.path.join(od, 'summary.json'), 'w'), indent=1)
    log.write(f'TOTAL cost ${total:.4f}, model time {secs:.0f}s, {len(items)}/{len(icons)} icons\n'); log.close()
    print(f'{a.model}: ${total:.4f}, {secs:.0f}s, {len(items)}/{len(icons)} icons')


if __name__ == '__main__':
    main()
