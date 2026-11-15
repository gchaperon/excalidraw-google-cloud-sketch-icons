"""Builders for Excalidraw elements, library items and library files.

All icons use the same look: stroke width 6, roughness 1 ("artist"), and an Excalifont
label under the icon. Coordinates are Excalidraw px; an icon fills an ICON x ICON box at
the origin.
"""

import random
import time

ICON = 120  # icon box size in Excalidraw px
STROKE_WIDTH = 6
ROUGHNESS = 1  # 0 architect, 1 artist, 2 cartoonist

FONT_SIZE = 20
LINE_HEIGHT = 1.25
CHAR_W = 0.55  # approximate average Excalifont glyph width, as a fraction of FONT_SIZE
FONT_EXCALIFONT = 5  # Excalidraw fontFamily id
LABEL_COLOR = "#1e1e1e"


def _now_ms():
    return int(time.time() * 1000)


def new_group():
    """Return a new random Excalidraw group id."""
    return f"g{random.randrange(1 << 40):x}"


def element(kind, x, y, w, h, color, group, **props):
    """Return an Excalidraw element of type ``kind`` with the common properties set.

    Args:
        kind: Excalidraw element type, for example "line", "ellipse" or "text".
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.
        color: Stroke color.
        group: Group id. All elements of one icon share a group, so they move together.
        **props: Properties that override the defaults or add type-specific ones.
    """
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
        "strokeWidth": STROKE_WIDTH,
        "strokeStyle": "solid",
        "roughness": ROUGHNESS,
        "opacity": 100,
        "groupIds": [group],
        "frameId": None,
        "roundness": None,
        "seed": random.randrange(1 << 31),
        "version": 1,
        "versionNonce": random.randrange(1 << 31),
        "isDeleted": False,
        "boundElements": None,
        "updated": _now_ms(),
        "link": None,
        "locked": False,
    } | props


def line(points, color, group, curved, closed=False, **props):
    """Return an Excalidraw line through absolute ``points`` = [(x, y), ...].

    Args:
        points: At least 2 points.
        color: Stroke color.
        group: Group id.
        curved: If True, Excalidraw draws a smooth curve through the points.
        closed: If True, the line ends at its first point.
        **props: Other element properties, for example a fill.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if closed and pts[0] != pts[-1]:
        pts.append(pts[0])
    x0, y0 = pts[0]
    rel = [[x - x0, y - y0] for x, y in pts]  # Excalidraw points are relative to (x, y)
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    return element(
        "line",
        x0,
        y0,
        max(xs) - min(xs),
        max(ys) - min(ys),
        color,
        group,
        points=rel,
        lastCommittedPoint=None,
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead=None,
        roundness={"type": 2} if curved else None,
        **props,
    )


def label(name, group):
    """Return the Excalifont label that goes under the icon.

    Names longer than 14 characters wrap to two lines, at the word break that gives the
    most equal line lengths.

    Args:
        name: Text of the label.
        group: Group id of the icon, so that the label moves with it.
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
    return element(
        "text",
        (ICON - w) / 2,
        ICON + 8,
        w,
        h,
        LABEL_COLOR,
        group,
        strokeWidth=1,
        text=text,
        originalText=text,
        fontSize=FONT_SIZE,
        fontFamily=FONT_EXCALIFONT,
        textAlign="center",
        verticalAlign="top",
        containerId=None,
        lineHeight=LINE_HEIGHT,
        autoResize=True,
    )


def item(item_id, name, elements):
    """Return a library item."""
    return {
        "id": item_id,
        "status": "unpublished",
        "created": _now_ms(),
        "name": name,
        "elements": elements,
    }


def library(items, source):
    """Return an .excalidrawlib document (version 2)."""
    return {"type": "excalidrawlib", "version": 2, "source": source, "libraryItems": items}
