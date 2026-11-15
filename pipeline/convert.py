"""Approach B: deterministic SVG -> sketch-style Excalidraw elements.

For every <path>/<circle>/<rect>/... in the icon's art:
  1. rasterize that shape alone (rsvg-convert) at RES x RES
  2. split the raster into connected regions, and for each region:
     - small round dot: a solid filled ellipse
     - solid blob (thick region): an ellipse if it is round, else a filled closed line
     - thin stroke (outlined stroke): skeletonize -> trace polylines -> prune spurs
       -> extend line ends -> fit circles -> simplify -> split at corners -> Excalidraw `line`s
  3. put everything in one group.

The result has the right proportions but often has small errors. Approach C (redraw.py)
uses it as a geometric guide for the LLM.

Coordinates: the rasters and polylines use (row, column) = (y, x) at RES px; the output
elements use (x, y) in an ICON x ICON box.

Needs `rsvg-convert` (librsvg) on PATH.
"""

import math
import re
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours, label, regionprops
from skimage.morphology import skeletonize

from . import excalidraw as ex

RES = 512  # raster resolution (icons are 512 viewBox)
SCALE = ex.ICON / RES  # raster px -> Excalidraw px
SPUR = 34  # px at RES: drop skeleton branches shorter than this
TOL = 4.0  # px at RES: RDP tolerance
CORNER = 50  # degrees: split polylines at turns sharper than this

type Pixel = tuple[int, int]
"""A raster pixel, (row, column) = (y, x)."""
type RasterPoint = tuple[float, float]
"""A point in raster px, (y, x)."""
type Polyline = Sequence[RasterPoint]

SHAPE_RE = re.compile(r"<(path|circle|rect|ellipse|polygon|polyline|line)\b[^>]*?/?>", re.S)


def rasterize(svg_text: str) -> npt.NDArray[np.uint8]:
    """Rasterize an SVG document with rsvg-convert.

    Returns:
        An RGBA uint8 array of shape (RES, RES, 4).
    """
    png = subprocess.run(
        ["rsvg-convert", "-w", str(RES), "-h", str(RES)],
        input=svg_text.encode(),
        capture_output=True,
        check=True,
    ).stdout
    return np.array(Image.open(BytesIO(png)).convert("RGBA"))


def shapes(svg: str) -> Iterator[str]:
    """Yield one SVG document per drawable shape of the art, with only that shape in it.

    The invisible ``bounding_box`` group of Google icons is removed. Shapes inside
    <defs>, <mask> and <clipPath> are not counted as drawable, and they stay in every
    document, so that references to them still work.

    This works on the SVG text with regular expressions, which is sufficient for the
    simple, flat SVGs of the Google icons.
    """
    body = re.sub(r'<g id="bounding_box">.*?</g>', "", svg, flags=re.S)
    hidden = [m.span() for m in re.finditer(r"<(defs|mask|clipPath)\b.*?</\1>", body, re.S)]

    def is_hidden(m: re.Match[str]) -> bool:
        return any(a <= m.start() < b for a, b in hidden)

    matches = [m for m in SHAPE_RE.finditer(body) if not is_hidden(m)]
    for i in range(len(matches)):
        only = body
        # Remove every other drawable shape. Go backwards so that the offsets stay valid.
        for j, n in reversed(list(enumerate(matches))):
            if j != i:
                only = only[: n.start()] + only[n.end() :]
        yield only


def neighbors(p: Pixel, pts: set[Pixel]) -> Iterator[Pixel]:
    """Yield the 8-connected neighbors of pixel ``p`` = (y, x) that are in the set ``pts``."""
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and (y + dy, x + dx) in pts:
                yield (y + dy, x + dx)


def trace(skel: npt.NDArray[np.bool_]) -> tuple[list[list[Pixel]], dict[Pixel, int]]:
    """Trace a skeleton bitmap into polylines.

    Every pixel with other than 2 neighbors is a node (an end point or a junction). Each
    polyline goes from a node to the next node. Closed loops without a node (for example
    circles) become polylines that start and end at the same pixel.

    Args:
        skel: Boolean skeleton bitmap, one pixel wide.

    Returns:
        A tuple (polylines, degree). Each polyline is a list of (y, x) pixels. ``degree``
        maps each skeleton pixel to its number of neighbors.
    """
    pts: set[Pixel] = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel), strict=True)}
    deg = {p: sum(1 for _ in neighbors(p, pts)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    seen_edges: set[tuple[Pixel, Pixel]] = set()
    lines: list[list[Pixel]] = []

    def walk(a: Pixel, b: Pixel) -> list[Pixel]:
        """Follow the skeleton from node ``a`` through ``b`` to the next node."""
        line = [a, b]
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in neighbors(cur, pts) if q != prev and (cur, q) not in seen_edges]
            if not nxt:
                break
            # prefer 4-connected step to avoid diagonal shortcuts duplicating
            nxt.sort(key=lambda q, cur=cur: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            seen_edges.add((cur, nxt[0]))
            seen_edges.add((nxt[0], cur))
            prev, cur = cur, nxt[0]
            line.append(cur)
            if cur == a:
                break
        return line

    for n in nodes:
        for q in neighbors(n, pts):
            if (n, q) not in seen_edges:
                seen_edges.add((n, q))
                seen_edges.add((q, n))
                lines.append(walk(n, q))
    # pure loops (no node) e.g. circles
    covered = {p for line in lines for p in line}
    rest = pts - covered
    while rest:
        start = next(iter(rest))
        nb = list(neighbors(start, pts))
        if not nb:
            rest.discard(start)
            continue
        nodes.add(start)
        seen_edges.add((start, nb[0]))
        seen_edges.add((nb[0], start))
        line = walk(start, nb[0])
        lines.append(line)
        rest -= set(line)
    return lines, deg


def prune(lines: list[list[Pixel]], deg: dict[Pixel, int]) -> list[list[Pixel]]:
    """Remove spurs and very short polylines.

    A spur is a short branch from a junction to a free end. Skeletonization makes them at
    the corners and ends of thick strokes.

    Args:
        lines: Polylines from ``trace``.
        deg: Degree map from ``trace``.

    Returns:
        The polylines that are kept.
    """
    out = []
    for line in lines:
        length = sum(math.dist(line[k], line[k + 1]) for k in range(len(line) - 1))
        endpoint = deg.get(line[0], 2) == 1 or deg.get(line[-1], 2) == 1
        junction = deg.get(line[0], 2) >= 3 or deg.get(line[-1], 2) >= 3
        if length < SPUR and endpoint and junction:
            continue  # spur
        if length < 4:
            continue
        out.append(line)
    return out


def split_corners(poly: Iterable[Any]) -> list[list[RasterPoint]]:
    """Split a polyline at each turn sharper than CORNER degrees.

    Excalidraw's curved lines round every corner, so sharp corners must be the ends of
    separate lines.

    Returns:
        List of polylines, as lists of float tuples. Adjacent parts share their end point.
    """
    pts = [(float(p[0]), float(p[1])) for p in poly]
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
                parts.append(cur)
                cur = [pts[k]]
    cur.append(pts[-1])
    parts.append(cur)
    return parts


def fit_circle(pts: Polyline) -> tuple[float, float, float, float]:
    """Fit a circle to points with linear least squares.

    Args:
        pts: Points as (a, b) pairs. The result uses the same axis order.

    Returns:
        A tuple (center a, center b, radius, maximum distance of a point from the circle).
    """
    p = np.array(pts, float)
    # (a - ca)^2 + (b - cb)^2 = r^2  <=>  2a*ca + 2b*cb + c = a^2 + b^2,  c = r^2 - ca^2 - cb^2
    coeffs = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    rhs = (p**2).sum(1)
    ca, cb, c = np.linalg.lstsq(coeffs, rhs, rcond=None)[0]
    r = math.sqrt(max(c + ca * ca + cb * cb, 0))
    err = np.abs(np.hypot(p[:, 0] - ca, p[:, 1] - cb) - r).max()
    return float(ca), float(cb), r, float(err)


def to_xy(pts: Iterable[Any]) -> list[ex.Point]:
    """Convert raster points [(y, x), ...] to Excalidraw points [(x, y), ...]."""
    return [(float(p[1]) * SCALE, float(p[0]) * SCALE) for p in pts]


def color_of(im: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> str:
    """Return the median color of the opaque pixels as "#rrggbb".

    Args:
        im: RGBA raster.
        mask: Pixels of the shape. Used if no pixel is more than half opaque.
    """
    opaque = im[..., 3] > 128
    pixels = im[opaque if opaque.any() else mask]
    r, g, b = np.median(pixels[:, :3], axis=0).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def extend_ends(line: Polyline, deg: dict[Pixel, int], half: float) -> list[RasterPoint]:
    """Extend the free ends of a polyline by 0.8 x the stroke half-width.

    The skeleton stops about one half-width before the end of the stroke, so without this
    the lines look too short.

    Args:
        line: Polyline in (y, x) raster px, with more than 6 points.
        deg: Degree map from ``trace``. Only ends with degree 1 are free ends.
        half: Stroke half-width in raster px.

    Returns:
        A new polyline.
    """
    line = list(line)
    for end, ref in ((0, 6), (-1, -7)):  # direction from the 7th point to the end point
        if deg.get(line[end], 2) == 1:
            (y0, x0), (y1, x1) = line[end], line[ref]
            d = math.hypot(y0 - y1, x0 - x1) or 1
            ext = (y0 + (y0 - y1) / d * half * 0.8, x0 + (x0 - x1) / d * half * 0.8)
            if end == 0:
                line.insert(0, ext)
            else:
                line.append(ext)
    return line


def convert(svg_text: str) -> list[ex.Element]:
    """Convert one icon SVG to Excalidraw elements (approach B).

    Args:
        svg_text: The SVG document.

    Returns:
        The elements, all in one group, without a label.
    """
    group = ex.new_group()
    els = []
    for shape_svg in shapes(svg_text):
        im = rasterize(shape_svg)
        mask = im[..., 3] > 100
        if mask.sum() < 30:
            continue  # empty or nearly invisible
        color = color_of(im, mask)
        lab = label(mask)
        for reg in regionprops(lab):
            m = lab == reg.label
            half = ndimage.distance_transform_edt(m).max()  # half of the thickest part
            minr, minc, maxr, maxc = reg.bbox
            h, w = maxr - minr, maxc - minc
            box = (minc * SCALE, minr * SCALE, w * SCALE, h * SCALE)
            if reg.area < 2600 and reg.solidity > 0.85 and reg.eccentricity < 0.8:
                # small dot: solid filled ellipse
                els.append(
                    ex.element(
                        "ellipse",
                        *box,
                        color,
                        group,
                        backgroundColor=color,
                        fillStyle="solid",
                        strokeWidth=1,
                    )
                )
                continue
            if half > 28:  # solid blob, not a stroke
                if reg.eccentricity < 0.5 and reg.solidity > 0.9:
                    els.append(
                        ex.element(
                            "ellipse",
                            *box,
                            color,
                            group,
                            backgroundColor=color,
                            fillStyle="hachure",
                        )
                    )
                else:
                    outline = max(find_contours(m.astype(float), 0.5), key=len)
                    outline = approximate_polygon(outline, TOL)
                    els.append(
                        ex.line(
                            to_xy(outline),
                            color,
                            group,
                            False,
                            closed=True,
                            backgroundColor=color,
                            fillStyle="hachure",
                        )
                    )
                continue
            lines, deg = trace(skeletonize(m))
            for line in prune(lines, deg):
                closed = line[0] == line[-1]
                if not closed and len(line) > 6:
                    line = extend_ends(line, deg, half)
                if closed and len(line) > 20:
                    cy, cx, r, err = fit_circle(line)
                    if err < 3:  # a circle: draw it as an ellipse
                        d = 2 * r * SCALE
                        els.append(
                            ex.element(
                                "ellipse", (cx - r) * SCALE, (cy - r) * SCALE, d, d, color, group
                            )
                        )
                        continue
                simplified = approximate_polygon(np.array(line, float), TOL)
                for part in split_corners(simplified):
                    if len(part) < 2:
                        continue
                    curved = len(part) > 2
                    if curved:  # resample curved part more finely for smooth roundness
                        part = [tuple(p) for p in approximate_polygon(np.array(part), TOL / 2)]
                    els.append(ex.line(to_xy(part), color, group, curved))
    return els
