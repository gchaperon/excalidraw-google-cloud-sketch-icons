"""Approach B: deterministic SVG -> sketch-style Excalidraw library item.

For every <path>/<circle>/<rect> in the icon's art group:
  1. rasterize that shape alone (rsvg-convert) at RES x RES
  2. if it is a thin stroke (outlined stroke): skeletonize -> trace polylines
     -> prune spurs -> split at corners -> simplify -> Excalidraw `line`s
     if it is a solid blob: emit an ellipse (if round) or a filled closed line
  3. group everything, add a Virgil label below.

Usage: python -m pipeline.convert OUT.excalidrawlib NAME=path.svg [NAME=path.svg ...]
"""
import json, math, random, re, subprocess, sys, tempfile, os, time
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import approximate_polygon, label, regionprops, find_contours

RES = 512          # raster resolution (icons are 512 viewBox)
ICON = 120         # output icon size in Excalidraw px
STROKE = 6         # Excalidraw stroke width
ROUGH = 1          # Excalidraw roughness (0 architect, 1 artist, 2 cartoonist)
SPUR = 34          # px at RES: drop skeleton branches shorter than this
TOL = 4.0          # px at RES: RDP tolerance
CORNER = 50        # degrees: split polylines at turns sharper than this

SHAPE_RE = re.compile(r'<(path|circle|rect|ellipse|polygon|polyline|line)\b[^>]*?/?>', re.S)


def rasterize(svg_text):
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, 'a.svg'), os.path.join(d, 'a.png')
        open(src, 'w').write(svg_text)
        subprocess.run(['rsvg-convert', '-w', str(RES), '-h', str(RES), '-o', out, src], check=True)
        im = np.array(Image.open(out).convert('RGBA'))
    return im


def shapes(svg):
    """Yield (svg_with_only_this_shape, color) for each drawable shape of the art."""
    # strip the invisible bounding box group used by Google icons
    body = re.sub(r'<g id="bounding_box">.*?</g>', '', svg, flags=re.S)
    hidden = [m.span() for m in re.finditer(r'<(defs|mask|clipPath)\b.*?</\1>', body, re.S)]
    inside = lambda m: any(a <= m.start() < b for a, b in hidden)
    matches = [m for m in SHAPE_RE.finditer(body) if not inside(m)]
    for i, m in enumerate(matches):
        others = body
        # blank out every other drawable shape; defs/masks stay intact
        for j, n in reversed(list(enumerate(matches))):
            if j != i:
                others = others[:n.start()] + others[n.end():]
        yield others


def neighbors(p, pts):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and (y + dy, x + dx) in pts:
                yield (y + dy, x + dx)


def trace(skel):
    """Skeleton bitmap -> list of polylines [(y,x),...]."""
    pts = set(zip(*np.nonzero(skel)))
    deg = {p: sum(1 for _ in neighbors(p, pts)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    seen_edges, lines = set(), []

    def walk(a, b):
        line = [a, b]
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in neighbors(cur, pts) if q != prev and (cur, q) not in seen_edges]
            # prefer 4-connected step to avoid diagonal shortcuts duplicating
            if not nxt:
                break
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            seen_edges.add((cur, nxt[0])); seen_edges.add((nxt[0], cur))
            prev, cur = cur, nxt[0]
            line.append(cur)
            if cur == a:
                break
        return line

    for n in nodes:
        for q in neighbors(n, pts):
            if (n, q) not in seen_edges:
                seen_edges.add((n, q)); seen_edges.add((q, n))
                lines.append(walk(n, q))
    # pure loops (no node) e.g. circles
    covered = {p for l in lines for p in l}
    rest = pts - covered
    while rest:
        start = next(iter(rest))
        nb = list(neighbors(start, pts))
        if not nb:
            rest.discard(start); continue
        nodes.add(start)
        seen_edges.add((start, nb[0])); seen_edges.add((nb[0], start))
        l = walk(start, nb[0])
        lines.append(l)
        rest -= set(l)
    return lines, deg


def prune(lines, deg):
    out = []
    for l in lines:
        length = sum(math.dist(l[k], l[k + 1]) for k in range(len(l) - 1))
        endpoint = deg.get(l[0], 2) == 1 or deg.get(l[-1], 2) == 1
        junction = deg.get(l[0], 2) >= 3 or deg.get(l[-1], 2) >= 3
        if length < SPUR and endpoint and junction:
            continue  # spur
        if length < 4:
            continue
        out.append(l)
    return out


def split_corners(poly):
    pts = [tuple(map(float, p)) for p in poly]
    if len(pts) < 3:
        return [pts]
    parts, cur = [], [pts[0]]
    for k in range(1, len(pts) - 1):
        cur.append(pts[k])
        a, b, c = pts[k - 1], pts[k], pts[k + 1]
        v1, v2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 and n2:
            cos = max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            if math.degrees(math.acos(cos)) > CORNER:
                parts.append(cur); cur = [pts[k]]
    cur.append(pts[-1]); parts.append(cur)
    return parts


def fit_circle(pts):
    p = np.array(pts, float)
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = (p ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(c + cx * cx + cy * cy, 0))
    err = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r).max()
    return cx, cy, r, err


_ids = 0


def base(kind, x, y, w, h, color, group, **kw):
    global _ids
    _ids += 1
    e = dict(id=f'b{_ids}-{random.randrange(1 << 30):x}', type=kind, x=x, y=y, width=w, height=h,
             angle=0, strokeColor=color, backgroundColor='transparent', fillStyle='solid',
             strokeWidth=STROKE, strokeStyle='solid', roughness=ROUGH, opacity=100,
             groupIds=[group], frameId=None, roundness=None, seed=random.randrange(1 << 31),
             version=1, versionNonce=random.randrange(1 << 31), isDeleted=False,
             boundElements=None, updated=int(time.time() * 1000), link=None, locked=False)
    e.update(kw)
    return e


def line_el(pts, color, group, curved, closed=False, fill=None):
    s = ICON / RES
    xy = [(p[1] * s, p[0] * s) for p in pts]  # (y,x) -> (x,y)
    if closed and xy[0] != xy[-1]:
        xy.append(xy[0])
    x0, y0 = xy[0]
    rel = [[px - x0, py - y0] for px, py in xy]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    kw = dict(points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
              startArrowhead=None, endArrowhead=None,
              roundness={'type': 2} if curved else None)
    if fill:
        kw.update(backgroundColor=fill, fillStyle='hachure')
    return base('line', x0, y0, max(xs) - min(xs), max(ys) - min(ys), color, group, **kw)


def color_of(im):
    a = im[..., 3] > 128
    if not a.any():
        return None
    rgb = np.median(im[a][:, :3], axis=0).astype(int)
    return '#%02x%02x%02x' % tuple(rgb)


def convert(svg_text, name):
    group = f'g-{random.randrange(1 << 30):x}'
    els = []
    for shape_svg in shapes(svg_text):
        im = rasterize(shape_svg)
        mask = im[..., 3] > 100
        if mask.sum() < 30:
            continue
        color = color_of(im)
        lab = label(mask)
        for reg in regionprops(lab):
            m = lab == reg.label
            dist = ndimage.distance_transform_edt(m)
            half = dist.max()
            minr, minc, maxr, maxc = reg.bbox
            if reg.area < 2600 and reg.solidity > 0.85 and reg.eccentricity < 0.8:
                # small dot: solid filled ellipse
                s = ICON / RES
                h, w = maxr - minr, maxc - minc
                els.append(base('ellipse', minc * s, minr * s, w * s, h * s, color, group,
                                backgroundColor=color, fillStyle='solid', strokeWidth=1))
                continue
            if half > 28:  # solid blob, not a stroke
                h, w = maxr - minr, maxc - minc
                if reg.eccentricity < 0.5 and reg.solidity > 0.9:
                    s = ICON / RES
                    els.append(base('ellipse', minc * s, minr * s, w * s, h * s, color, group,
                                    backgroundColor=color, fillStyle='hachure'))
                else:
                    c = max(find_contours(m.astype(float), 0.5), key=len)
                    c = approximate_polygon(c, TOL)
                    els.append(line_el(c, color, group, False, closed=True, fill=color))
                continue
            sk = skeletonize(m)
            lines, deg = trace(sk)
            for l in prune(lines, deg):
                closed = l[0] == l[-1]
                if not closed and len(l) > 6:
                    l = list(l)
                    for end, ref in ((0, 6), (-1, -7)):
                        if deg.get(l[end], 2) == 1:
                            (y0, x0), (y1, x1) = l[end], l[ref]
                            d = math.hypot(y0 - y1, x0 - x1) or 1
                            ext = (y0 + (y0 - y1) / d * half * 0.8, x0 + (x0 - x1) / d * half * 0.8)
                            if end == 0: l.insert(0, ext)
                            else: l.append(ext)
                if closed and len(l) > 20:
                    cx, cy, r, err = fit_circle(l)
                    if err < 3:
                        s = ICON / RES
                        els.append(base('ellipse', (cy - r) * s, (cx - r) * s, 2 * r * s, 2 * r * s,
                                        color, group))
                        continue
                simp = approximate_polygon(np.array(l, float), TOL)
                for part in split_corners(simp):
                    if len(part) < 2:
                        continue
                    curved = len(part) > 2
                    if curved:  # resample curved part more finely for smooth roundness
                        part = [tuple(p) for p in approximate_polygon(np.array(part), TOL / 2)]
                    els.append(line_el(part, color, group, curved))
    # label
    els.append(base('text', 0, ICON + 8, ICON, 25, '#1e1e1e', group, text=name, originalText=name,
                    fontSize=20, fontFamily=1, textAlign='center', verticalAlign='top',
                    containerId=None, lineHeight=1.25, baseline=18, roughness=1, strokeWidth=1))
    return els


def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    items = []
    for pr in pairs:
        name, path = pr.split('=', 1)
        els = convert(open(path).read(), name)
        print(f'{name}: {len(els)} elements', file=sys.stderr)
        items.append(dict(id=f'item-{len(items)}', status='unpublished', created=int(time.time() * 1000),
                          name=name, elements=els))
    json.dump(dict(type='excalidrawlib', version=2, source='gcp-sketch-b', libraryItems=items),
              open(out, 'w'))


if __name__ == '__main__':
    main()
