import pytest

from pipeline import compact

from .conftest import as_json

SHAPES = [
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


def test_round_trip_keeps_the_shapes():
    els = compact.to_excalidraw(SHAPES, "BigQuery")

    assert compact.from_excalidraw(els) == SHAPES


def test_to_excalidraw_skips_invalid_shapes_and_adds_label_last():
    shapes = [{"t": "triangle"}, {"t": "line", "pts": [[1, 1]]}, *SHAPES[:1]]

    els = compact.to_excalidraw(shapes, "BigQuery")

    assert [e["type"] for e in els] == ["line", "text"]
    assert els[-1]["text"] == "BigQuery"
    assert els[0]["groupIds"] == els[1]["groupIds"]


def test_closed_line_ends_at_its_first_point():
    shape = {"t": "line", "pts": [[0, 0], [10, 0], [10, 10]], "closed": True}

    (line, _label) = compact.to_excalidraw([shape], "X")

    assert line["points"][-1] == line["points"][0] == [0.0, 0.0]


def test_small_solid_dot_gets_a_thin_outline():
    dot = {"t": "ellipse", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "solid"}
    big = {**dot, "w": 30, "h": 30}

    small_el, big_el, _ = compact.to_excalidraw([dot, big], "X")

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
    assert compact.label(name, "g")["text"] == text


def test_to_excalidraw_snapshot(deterministic, snapshot_json):
    assert as_json(compact.to_excalidraw(SHAPES, "Cloud Run Functions")) == snapshot_json
