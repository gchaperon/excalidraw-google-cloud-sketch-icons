"""Compact shape format <-> Excalidraw elements.

The LLM reads and writes this format instead of full Excalidraw elements, because it is much
shorter. Coordinates are in a 120x120 icon box, with the origin at the top left:

    {"t": "line", "c": "#4285f4", "pts": [[x, y], ...], "curve": true|false, "closed": false,
     "fill": "none|solid|hachure"}
    {"t": "ellipse", "c": "#ea4335", "x": .., "y": .., "w": .., "h": ..,
     "fill": "none|solid|hachure"}
    {"t": "rect", "c": "#34a853", "x": .., "y": .., "w": .., "h": .., "fill": "none|solid|hachure",
     "round": false}

Optional on any shape: "sw" (stroke width, default 6).

This module also makes the text label that goes under every icon.
"""

import random
import time

ICON = 120  # icon box size in Excalidraw px
FONT_SIZE = 20
LINE_HEIGHT = 1.25
CHAR_W = 0.55  # approximate average Excalifont glyph width, as a fraction of FONT_SIZE
FONT_EXCALIFONT = 5  # Excalidraw fontFamily id


def _base(kind, x, y, w, h, color, group, sw, rough):
    """Return an Excalidraw element of type ``kind`` with the common properties set."""
    return {
        "id": f"{kind[0]}{random.randrange(1 << 40):x}",
        "type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": sw,
        "strokeStyle": "solid",
        "roughness": rough,
        "opacity": 100,
        "groupIds": [group],
        "frameId": None,
        "roundness": None,
        "seed": random.randrange(1 << 31),
        "version": 1,
        "versionNonce": random.randrange(1 << 31),
        "isDeleted": False,
        "boundElements": None,
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
    }


def to_excalidraw(shapes, name, rough=1):
    """Convert compact shapes to Excalidraw elements, and add the label.

    Shapes with an unknown type and lines with fewer than 2 points are skipped.

    Args:
        shapes: List of compact shapes (see the module docstring).
        name: Text of the label.
        rough: Excalidraw roughness (0 architect, 1 artist, 2 cartoonist).

    Returns:
        The elements, all in one new group. The label is the last element.
    """
    group = f"g{random.randrange(1 << 40):x}"
    els = []
    for s in shapes:
        t, c, sw = s.get("t"), s.get("c", "#1e1e1e"), float(s.get("sw", 6))
        fill = s.get("fill", "none")
        if t == "line":
            pts = [[float(p[0]), float(p[1])] for p in s.get("pts", [])]
            if len(pts) < 2:
                continue
            if s.get("closed") and pts[0] != pts[-1]:
                pts.append(pts[0])
            x0, y0 = pts[0]
            rel = [[px - x0, py - y0] for px, py in pts]  # Excalidraw points are relative
            xs, ys = [p[0] for p in rel], [p[1] for p in rel]
            e = _base("line", x0, y0, max(xs) - min(xs), max(ys) - min(ys), c, group, sw, rough)
            e.update(
                points=rel,
                lastCommittedPoint=None,
                startBinding=None,
                endBinding=None,
                startArrowhead=None,
                endArrowhead=None,
                roundness={"type": 2} if s.get("curve") else None,
            )
        elif t in ("ellipse", "rect"):
            kind = "ellipse" if t == "ellipse" else "rectangle"
            x, y, w, h = (float(s[k]) for k in ("x", "y", "w", "h"))
            e = _base(kind, x, y, w, h, c, group, sw, rough)
            if t == "rect" and s.get("round"):
                e["roundness"] = {"type": 3}
        else:
            continue
        if fill in ("solid", "hachure", "cross-hatch"):
            e.update(backgroundColor=c, fillStyle=fill)
            if t == "ellipse" and fill == "solid" and max(e["width"], e["height"]) <= 16:
                e["strokeWidth"] = 1  # small dot: thin outline keeps its original size
        els.append(e)
    els.append(label(name, group))
    return els


def label(name, group):
    """Return the Excalifont label that goes under the icon.

    Names longer than 14 characters wrap to two lines, at the word break that gives the
    most equal line lengths.

    Args:
        name: Text of the label.
        group: Excalidraw group id, so that the label moves with the icon.
    """
    words, lines = name.split(), [name]
    if len(name) > 14 and len(words) > 1:

        def imbalance(i):
            return abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))

        cut = min(range(1, len(words)), key=imbalance)
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    text = "\n".join(lines)
    w = max(len(line) for line in lines) * FONT_SIZE * CHAR_W
    h = len(lines) * FONT_SIZE * LINE_HEIGHT
    return {
        **_base("text", (ICON - w) / 2, ICON + 8, w, h, "#1e1e1e", group, 1, 1),
        "text": text,
        "originalText": text,
        "fontSize": FONT_SIZE,
        "fontFamily": FONT_EXCALIFONT,
        "textAlign": "center",
        "verticalAlign": "top",
        "containerId": None,
        "lineHeight": LINE_HEIGHT,
        "autoResize": True,
    }


def _round1(v):
    return round(float(v), 1)


def from_excalidraw(els):
    """Convert Excalidraw elements to compact shapes.

    Only lines, ellipses and rectangles are kept; text and other elements are dropped.
    Coordinates are rounded to 0.1 to keep the prompt short.

    Args:
        els: Excalidraw elements, drawn in the ICON x ICON box.

    Returns:
        List of compact shapes (see the module docstring).
    """
    out = []
    for e in els:
        has_fill = e.get("backgroundColor") not in (None, "transparent")
        fill = {"fill": e.get("fillStyle", "solid")} if has_fill else {}
        if e["type"] == "line":
            pts = [[_round1(e["x"] + p[0]), _round1(e["y"] + p[1])] for p in e["points"]]
            out.append(
                {"t": "line", "c": e["strokeColor"], "curve": bool(e.get("roundness")), "pts": pts}
                | fill
            )
        elif e["type"] in ("ellipse", "rectangle"):
            out.append(
                {
                    "t": "ellipse" if e["type"] == "ellipse" else "rect",
                    "c": e["strokeColor"],
                    "x": _round1(e["x"]),
                    "y": _round1(e["y"]),
                    "w": _round1(e["width"]),
                    "h": _round1(e["height"]),
                }
                | fill
            )
    return out
