"""Compact shape format <-> Excalidraw elements.

Compact shape (coordinates in a 120x120 icon box, origin top-left):
  {"t": "line", "c": "#4285f4", "pts": [[x, y], ...], "curve": true|false, "closed": false, "fill": "none|solid|hachure"}
  {"t": "ellipse", "c": "#ea4335", "x": .., "y": .., "w": .., "h": .., "fill": "none|solid|hachure"}
  {"t": "rect", "c": "#34a853", "x": .., "y": .., "w": .., "h": .., "fill": "none|solid|hachure", "round": false}
Optional on any shape: "sw" (stroke width, default 6).
"""
import random, time

ICON = 120


def _base(kind, x, y, w, h, color, group, sw, rough):
    return dict(id=f'{kind[0]}{random.randrange(1 << 40):x}', type=kind, x=x, y=y, width=w, height=h,
                angle=0, strokeColor=color, backgroundColor='transparent', fillStyle='solid',
                strokeWidth=sw, strokeStyle='solid', roughness=rough, opacity=100,
                groupIds=[group], frameId=None, roundness=None, seed=random.randrange(1 << 31),
                version=1, versionNonce=random.randrange(1 << 31), isDeleted=False,
                boundElements=None, updated=int(time.time() * 1000), link=None, locked=False)


def to_excalidraw(shapes, name, rough=1):
    group = f'g{random.randrange(1 << 40):x}'
    els = []
    for s in shapes:
        t, c, sw = s.get('t'), s.get('c', '#1e1e1e'), float(s.get('sw', 6))
        fill = s.get('fill', 'none')
        if t == 'line':
            pts = [[float(p[0]), float(p[1])] for p in s.get('pts', [])]
            if len(pts) < 2:
                continue
            if s.get('closed') and pts[0] != pts[-1]:
                pts.append(pts[0])
            x0, y0 = pts[0]
            rel = [[px - x0, py - y0] for px, py in pts]
            xs, ys = [p[0] for p in rel], [p[1] for p in rel]
            e = _base('line', x0, y0, max(xs) - min(xs), max(ys) - min(ys), c, group, sw, rough)
            e.update(points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
                     startArrowhead=None, endArrowhead=None,
                     roundness={'type': 2} if s.get('curve') else None)
        elif t in ('ellipse', 'rect'):
            kind = 'ellipse' if t == 'ellipse' else 'rectangle'
            e = _base(kind, float(s['x']), float(s['y']), float(s['w']), float(s['h']), c, group, sw, rough)
            if t == 'rect' and s.get('round'):
                e['roundness'] = {'type': 3}
        else:
            continue
        if fill in ('solid', 'hachure', 'cross-hatch'):
            e.update(backgroundColor=c, fillStyle=fill)
            if t == 'ellipse' and fill == 'solid' and max(e['width'], e['height']) <= 16:
                e['strokeWidth'] = 1  # small dot: thin outline keeps its original size
        els.append(e)
    els.append(label(name, group))
    return els


FONT_SIZE, LINE_HEIGHT, CHAR_W = 20, 1.25, 0.55  # Excalifont, approximate average glyph width


def label(name, group):
    """Excalifont label under the icon; names longer than 14 characters wrap to two lines."""
    words, lines = name.split(), [name]
    if len(name) > 14 and len(words) > 1:
        cut = min(range(1, len(words)), key=lambda i: abs(len(' '.join(words[:i])) - len(' '.join(words[i:]))))
        lines = [' '.join(words[:cut]), ' '.join(words[cut:])]
    text = '\n'.join(lines)
    w = max(len(l) for l in lines) * FONT_SIZE * CHAR_W
    h = len(lines) * FONT_SIZE * LINE_HEIGHT
    return dict(_base('text', (ICON - w) / 2, ICON + 8, w, h, '#1e1e1e', group, 1, 1),
                text=text, originalText=text, fontSize=FONT_SIZE, fontFamily=5, textAlign='center',
                verticalAlign='top', containerId=None, lineHeight=LINE_HEIGHT, autoResize=True)


def from_excalidraw(els):
    """Excalidraw elements (icon box ICON) -> compact shapes (text dropped)."""
    out = []
    r = lambda v: round(float(v), 1)
    for e in els:
        fill = 'none' if e.get('backgroundColor') in (None, 'transparent') else e.get('fillStyle', 'solid')
        if e['type'] == 'line':
            out.append({'t': 'line', 'c': e['strokeColor'], 'curve': bool(e.get('roundness')),
                        'pts': [[r(e['x'] + p[0]), r(e['y'] + p[1])] for p in e['points']],
                        **({'fill': fill} if fill != 'none' else {})})
        elif e['type'] in ('ellipse', 'rectangle'):
            out.append({'t': 'ellipse' if e['type'] == 'ellipse' else 'rect', 'c': e['strokeColor'],
                        'x': r(e['x']), 'y': r(e['y']), 'w': r(e['width']), 'h': r(e['height']),
                        **({'fill': fill} if fill != 'none' else {})})
    return out
