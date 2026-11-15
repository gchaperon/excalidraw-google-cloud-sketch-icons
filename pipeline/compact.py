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
"""

from typing import Any

from . import excalidraw as ex

type Shape = dict[str, Any]
"""A compact shape, as described above. It comes from the model, so it is not trusted."""

DEFAULT_COLOR = "#1e1e1e"

_FILL = {"type": "string", "enum": ["none", "solid", "hachure"]}
_BOX = {k: {"type": "number"} for k in ("x", "y", "w", "h")}
_COMMON = {"c": {"type": "string", "description": "color, #rrggbb"}, "fill": _FILL}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "description": "what you changed, max 3 sentences"},
        "shapes": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "t": {"type": "string", "enum": ["line"]},
                            "pts": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "number"}},
                            },
                            "curve": {"type": "boolean"},
                            "closed": {"type": "boolean"},
                            **_COMMON,
                        },
                        "required": ["t", "c", "pts", "curve"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "t": {"type": "string", "enum": ["ellipse"]},
                            **_BOX,
                            **_COMMON,
                        },
                        "required": ["t", "c", "x", "y", "w", "h"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "t": {"type": "string", "enum": ["rect"]},
                            **_BOX,
                            "round": {"type": "boolean"},
                            **_COMMON,
                        },
                        "required": ["t", "c", "x", "y", "w", "h"],
                    },
                ]
            },
        },
    },
    "required": ["note", "shapes"],
}
"""JSON schema of a model answer: {"note": ..., "shapes": [compact shapes]}."""
FILLS = ("solid", "hachure", "cross-hatch")


def to_excalidraw(shapes: list[Shape], name: str) -> list[ex.Element]:
    """Convert compact shapes to Excalidraw elements, and add the label.

    Shapes with an unknown type and lines with fewer than 2 points are skipped.

    Args:
        shapes: List of compact shapes (see the module docstring).
        name: Text of the label.

    Returns:
        The elements, all in one new group. The label is the last element.
    """
    group = ex.new_group()
    els: list[ex.Element] = []
    for s in shapes:
        t, c = s.get("t"), s.get("c", DEFAULT_COLOR)
        props: dict[str, Any] = {"strokeWidth": float(s.get("sw", ex.STROKE_WIDTH))}
        fill = s.get("fill", "none")
        if fill in FILLS:
            props |= {"backgroundColor": c, "fillStyle": fill}
        if t == "line":
            pts = s.get("pts", [])
            if len(pts) < 2:
                continue
            e = ex.line(pts, c, group, bool(s.get("curve")), bool(s.get("closed")), **props)
        elif t in ("ellipse", "rect"):
            kind = "ellipse" if t == "ellipse" else "rectangle"
            x, y, w, h = (float(s[k]) for k in ("x", "y", "w", "h"))
            e = ex.element(kind, x, y, w, h, c, group, **props)
            if t == "rect" and s.get("round"):
                e["roundness"] = {"type": 3}
        else:
            continue
        if t == "ellipse" and fill == "solid" and max(e["width"], e["height"]) <= 16:
            e["strokeWidth"] = 1  # small dot: thin outline keeps its original size
        els.append(e)
    els.append(ex.label(name, group))
    return els


def _round1(v: float) -> float:
    return round(float(v), 1)


def from_excalidraw(els: list[ex.Element]) -> list[Shape]:
    """Convert Excalidraw elements to compact shapes.

    Only lines, ellipses and rectangles are kept; text and other elements are dropped.
    Coordinates are rounded to 0.1 to keep the prompt short.

    Args:
        els: Excalidraw elements, drawn in the ICON x ICON box.

    Returns:
        List of compact shapes (see the module docstring).
    """
    out: list[Shape] = []
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
