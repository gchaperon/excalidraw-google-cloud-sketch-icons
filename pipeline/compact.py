"""Compact shape format <-> Excalidraw elements.

The LLM reads and writes this format instead of full Excalidraw elements, because it is much
shorter. Coordinates are in a 120x120 icon box, with the origin at the top left:

    {"t": "line", "c": "#4285f4", "curve": true|false, "pts": [[x, y], ...], "closed": false,
     "fill": "none|solid|hachure"}
    {"t": "ellipse", "c": "#ea4335", "x": .., "y": .., "w": .., "h": ..,
     "fill": "none|solid|hachure"}
    {"t": "rect", "c": "#34a853", "x": .., "y": .., "w": .., "h": .., "round": false,
     "fill": "none|solid|hachure"}

Optional on any shape: "sw" (stroke width, default 6).

The model's answers are external data, so they are validated with the pydantic models
below (``validate``): an answer with an unknown key, a wrong type, a bad color or an
impossible size is rejected as a whole, with a message that says why.
"""

import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    PositiveFloat,
)

from . import excalidraw as ex

Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
Fill = Literal["none", "solid", "hachure", "cross-hatch"]


class _Shape(BaseModel):
    # Field order in the subclasses is the key order of the JSON in the prompts.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Line(_Shape):
    """A polyline or a smooth curve through the points."""

    t: Literal["line"]
    c: Color
    curve: bool = False
    pts: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=2)
    closed: bool = False
    fill: Fill = "none"
    sw: PositiveFloat | None = None


class Ellipse(_Shape):
    """An ellipse in the box (x, y, w, h)."""

    t: Literal["ellipse"]
    c: Color
    x: FiniteFloat
    y: FiniteFloat
    w: NonNegativeFloat
    h: NonNegativeFloat
    fill: Fill = "none"
    sw: PositiveFloat | None = None


class Rect(_Shape):
    """A rectangle (x, y, w, h), optionally with round corners."""

    t: Literal["rect"]
    c: Color
    x: FiniteFloat
    y: FiniteFloat
    w: NonNegativeFloat
    h: NonNegativeFloat
    round: bool = False
    fill: Fill = "none"
    sw: PositiveFloat | None = None


type Shape = Line | Ellipse | Rect


class _Answer(BaseModel):
    # Other keys next to "shapes" are ignored; the shapes themselves are strict.
    model_config = ConfigDict(extra="ignore", strict=True)

    shapes: list[Annotated[Line | Ellipse | Rect, Field(discriminator="t")]] = Field(min_length=1)


def validate_answer(json_text: str) -> list[Shape]:
    """Validate the JSON block of a model answer, ``{"shapes": [...]}``, in strict mode.

    Strict mode rejects values of the wrong JSON type, for example ``"x": "10"``.

    Raises:
        pydantic.ValidationError: If the text is not valid JSON, or not an object with a
            non-empty list of valid shapes.
    """
    return list(_Answer.model_validate_json(json_text).shapes)


def dump(shapes: Sequence[Shape]) -> str:
    """Return shapes as compact JSON for a prompt. Fields that were not set are left out."""
    return json.dumps([s.model_dump(exclude_unset=True) for s in shapes])


def to_excalidraw(shapes: Sequence[Shape], name: str) -> list[ex.Element]:
    """Convert shapes to Excalidraw elements, and add the label.

    Args:
        shapes: Validated shapes.
        name: Text of the label.

    Returns:
        The elements, all in one new group. The label is the last element.
    """
    group = ex.new_group()
    els: list[ex.Element] = []
    for s in shapes:
        props: dict[str, Any] = {"strokeWidth": float(s.sw or ex.STROKE_WIDTH)}
        if s.fill != "none":
            props |= {"backgroundColor": s.c, "fillStyle": s.fill}
        match s:
            case Line():
                e = ex.line(s.pts, s.c, group, s.curve, s.closed, **props)
            case Ellipse():
                e = ex.element("ellipse", s.x, s.y, s.w, s.h, s.c, group, **props)
                if s.fill == "solid" and max(s.w, s.h) <= 16:
                    e["strokeWidth"] = 1  # small dot: thin outline keeps its original size
            case Rect():
                e = ex.element("rectangle", s.x, s.y, s.w, s.h, s.c, group, **props)
                if s.round:
                    e["roundness"] = {"type": 3}
        els.append(e)
    els.append(ex.label(name, group))
    return els


def _round1(v: float) -> float:
    return round(float(v), 1)


def from_excalidraw(els: list[ex.Element]) -> list[Shape]:
    """Convert Excalidraw elements to shapes.

    Only lines, ellipses and rectangles are kept; text and other elements are dropped.
    Coordinates are rounded to 0.1 to keep the prompt short.

    Args:
        els: Excalidraw elements, drawn in the ICON x ICON box.
    """
    out: list[Shape] = []
    for e in els:
        has_fill = e.get("backgroundColor") not in (None, "transparent")
        fill = {"fill": e.get("fillStyle", "solid")} if has_fill else {}
        color = e["strokeColor"]
        if e["type"] == "line":
            pts = [(_round1(e["x"] + p[0]), _round1(e["y"] + p[1])) for p in e["points"]]
            curve = bool(e.get("roundness"))
            out.append(Line(t="line", c=color, curve=curve, pts=pts, **fill))
        elif e["type"] in ("ellipse", "rectangle"):
            x, y, w, h = (_round1(e[k]) for k in ("x", "y", "width", "height"))
            if e["type"] == "ellipse":
                out.append(Ellipse(t="ellipse", c=color, x=x, y=y, w=w, h=h, **fill))
            else:
                out.append(Rect(t="rect", c=color, x=x, y=y, w=w, h=h, **fill))
    return out
