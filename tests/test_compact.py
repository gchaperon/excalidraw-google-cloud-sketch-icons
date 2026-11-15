import json

import pytest
from pydantic import ValidationError

from pipeline import compact, excalidraw

from .conftest import as_json

RAW = [
    {"t": "line", "c": "#4285f4", "curve": True, "pts": [[10.0, 20.0], [30.5, 40.0], [50.0, 25.0]]},
    {
        "t": "line",
        "c": "#ea4335",
        "curve": False,
        "pts": [[0.0, 0.0], [10.0, 0.0]],
        "fill": "hachure",
    },
    {"t": "ellipse", "c": "#34a853", "x": 5.0, "y": 6.0, "w": 7.0, "h": 8.0, "fill": "solid"},
    {"t": "rect", "c": "#fbbc04", "x": 60.0, "y": 70.0, "w": 40.0, "h": 30.0},
]


def validate(shapes):
    return compact.validate_answer(json.dumps({"shapes": shapes}))


SHAPES = validate(RAW)


# --- validation ----------------------------------------------------------------------------


def test_validate_accepts_valid_shapes():
    assert [type(s).__name__ for s in SHAPES] == ["Line", "Line", "Ellipse", "Rect"]
    assert SHAPES[0].pts[1] == (30.5, 40.0)
    assert SHAPES[3].fill == "none"  # default


def test_validate_accepts_ints_and_ignores_keys_next_to_shapes():
    ellipse = {"t": "ellipse", "c": "#000000", "x": 1, "y": 2, "w": 3, "h": 4}
    text = json.dumps({"note": "x", "shapes": [ellipse]})

    (shape,) = compact.validate_answer(text)

    assert isinstance(shape, compact.Ellipse)
    assert (shape.x, shape.w) == (1.0, 3.0)


LINE = {"t": "line", "c": "#4285f4", "pts": [[0, 0], [10, 10]]}


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ({"t": "triangle", "c": "#4285f4"}, "does not match any of the expected tags"),
        (LINE | {"pts": [[0, 0]]}, "at least 2 items"),
        (LINE | {"pts": [[0, 0, 0], [1, 1]]}, "at most 2 items"),
        (LINE | {"c": "blue"}, "should match pattern"),
        (LINE | {"c": "#4285f"}, "should match pattern"),
        (LINE | {"curve": "yes"}, "valid boolean"),  # strict: no string for a bool
        (LINE | {"pts": [["0", 0], [1, 1]]}, "valid number"),  # strict: no string for a number
        (LINE | {"fill": "dots"}, "'none', 'solid', 'hachure' or 'cross-hatch'"),
        (LINE | {"sw": 0}, "greater than 0"),
        (LINE | {"colour": "#4285f4"}, "Extra inputs are not permitted"),
        (
            {"t": "ellipse", "c": "#000000", "x": 0, "y": 0, "w": -1, "h": 5},
            "greater than or equal to 0",
        ),
        ({"t": "rect", "c": "#000000", "x": 0, "y": 0, "w": 5}, "Field required"),
    ],
)
def test_validate_rejects_invalid_shapes(shape, error):
    with pytest.raises(ValidationError, match=error):
        validate([LINE, shape])


@pytest.mark.parametrize(
    "text",
    [
        '{"shapes": []}',  # no shapes
        "[]",  # not an object
        '{"other": 1}',  # no "shapes"
        "{not json",
        '{"shapes": [{"t": "line", "c": "#000000", "pts": [[0, NaN], [1, 1]]}]}',
    ],
)
def test_validate_rejects_invalid_answers(text):
    with pytest.raises(ValidationError):
        compact.validate_answer(text)


# --- conversion ----------------------------------------------------------------------------


def test_round_trip_keeps_the_shapes():
    els = compact.to_excalidraw(SHAPES, "BigQuery")

    assert compact.from_excalidraw(els) == SHAPES


def test_dump_leaves_out_unset_fields():
    assert json.loads(compact.dump(SHAPES)) == RAW


def test_to_excalidraw_adds_label_last_in_the_same_group():
    els = compact.to_excalidraw(SHAPES[:1], "BigQuery")

    assert [e["type"] for e in els] == ["line", "text"]
    assert els[-1]["text"] == "BigQuery"
    assert els[0]["groupIds"] == els[1]["groupIds"]


def test_closed_line_ends_at_its_first_point():
    (shape,) = validate(
        [{"t": "line", "c": "#000000", "pts": [[0, 0], [10, 0], [10, 10]], "closed": True}]
    )

    (line, _label) = compact.to_excalidraw([shape], "X")

    assert line["points"][-1] == line["points"][0] == [0.0, 0.0]


def test_small_solid_dot_gets_a_thin_outline():
    dot = {"t": "ellipse", "c": "#000000", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "solid"}

    small_el, big_el, _ = compact.to_excalidraw(validate([dot, dot | {"w": 30, "h": 30}]), "X")

    assert small_el["strokeWidth"] == 1
    assert big_el["strokeWidth"] == 6


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("BigQuery", "BigQuery"),
        ("Cloud Run Functions", "Cloud Run\nFunctions"),
        ("Superlongnamewithoutspaces", "Superlongnamewithoutspaces"),
        ("AI Applications & Agents", "AI Applications\n& Agents"),
    ],
)
def test_label_wraps_long_names_on_two_balanced_lines(name, text):
    assert excalidraw.label(name, "g")["text"] == text


def test_to_excalidraw_snapshot(deterministic, snapshot_json):
    assert as_json(compact.to_excalidraw(SHAPES, "Cloud Run Functions")) == snapshot_json
