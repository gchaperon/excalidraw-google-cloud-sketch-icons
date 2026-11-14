import pytest

from pipeline import convert

from .conftest import DATA, as_json, needs_rsvg


def test_shapes_keeps_one_drawable_shape_per_document():
    svg = (DATA / "strokes.svg").read_text()
    docs = list(convert.shapes(svg))

    assert len(docs) == 3  # ring, L-shape, dot; not the bounding box or the <defs> circle
    for doc, kept in zip(docs, ["<path", "<polygon", '<circle fill="#34a853"'], strict=True):
        assert kept in doc
        assert 'id="bounding_box"' not in doc
        assert '<circle id="unused"' in doc  # defs stay, so references still work
    assert "<polygon" not in docs[0]
    assert "<path" not in docs[1]


def test_split_corners_splits_at_sharp_turns_only():
    right_angle = [(0, 0), (0, 10), (10, 10)]
    gentle = [(0, 0), (0, 10), (2, 20)]

    assert convert.split_corners(right_angle) == [[(0, 0), (0, 10)], [(0, 10), (10, 10)]]
    assert convert.split_corners(gentle) == [[(0.0, 0.0), (0.0, 10.0), (2.0, 20.0)]]


def test_fit_circle_finds_center_and_radius():
    pts = [(10 + 5 * c, 20 + 5 * s) for c, s in [(1, 0), (0, 1), (-1, 0), (0, -1)]]

    ca, cb, r, err = convert.fit_circle(pts)

    assert (ca, cb, r) == pytest.approx((10, 20, 5))
    assert err == pytest.approx(0, abs=1e-9)


@needs_rsvg
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # ring -> circle fit; L-shaped stroke -> split at the corner; small dot
        (
            "strokes",
            [("ellipse", "none"), ("line", "none"), ("line", "none"), ("ellipse", "solid")],
        ),
        # round blob -> ellipse; other blob -> filled closed line
        ("blobs", [("ellipse", "hachure"), ("line", "hachure")]),
    ],
)
def test_convert_kinds(name, expected):
    els = convert.convert((DATA / f"{name}.svg").read_text())

    kinds = [
        (e["type"], e["fillStyle"] if e["backgroundColor"] != "transparent" else "none")
        for e in els
    ]
    assert kinds == expected
    assert len({g for e in els for g in e["groupIds"]}) == 1


@needs_rsvg
@pytest.mark.parametrize("name", ["strokes", "blobs"])
def test_convert_snapshot(name, deterministic, snapshot_json):
    assert as_json(convert.convert((DATA / f"{name}.svg").read_text())) == snapshot_json
