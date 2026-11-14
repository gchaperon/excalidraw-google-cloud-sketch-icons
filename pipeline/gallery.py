"""Build a comparison gallery HTML: one row per icon, columns = original + each library.

Usage: python -m pipeline.gallery OUT.html ICONS.json COL=lib.excalidrawlib [COL=...]

ICONS.json: [{"name": "BigQuery", "svg": "path/to.svg"}, ...]

Library items are matched to icons by item name. The page draws the items with
Excalidraw's own exportToSvg, loaded from esm.sh, so open it in a browser (or a headless
Chromium) with network access. The page title changes to "done" when it is ready.

Development tool: the build does not use it. Needs `rsvg-convert` (librsvg) on PATH.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset=utf-8><style>
body{margin:0;background:#fff;font:13px sans-serif;color:#222}
table{border-collapse:collapse;margin:8px}
td,th{border:1px solid #ddd;width:190px;height:200px;text-align:center;vertical-align:middle;padding:4px}
th{height:auto;background:#f4f4f4}td svg{max-width:180px;max-height:190px}td.n{width:90px;font-weight:bold}
.k{font-size:11px;color:#777}</style></head><body><table id=t></table>
<script type=module>
import * as X from "https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@18.2.0,react-dom@18.2.0";
const exportToSvg = X.exportToSvg || X.default.exportToSvg;
const rows = ROWS, titles = TITLES; const t = document.getElementById('t');
t.insertAdjacentHTML('beforeend',
  '<tr><th>icon</th><th>original</th>' + titles.map(x => `<th>${x}</th>`).join('') + '</tr>');
for (const r of rows) {
  const tr = document.createElement('tr'); t.appendChild(tr);
  tr.insertAdjacentHTML('beforeend',
    `<td class=n>${r.name}</td><td><img src="${r.png}" width=150></td>`);
  for (const els of r.cells) {
    const td = document.createElement('td'); tr.appendChild(td);
    if (!els) { td.textContent = '—'; continue; }
    try {
      td.appendChild(await exportToSvg({elements: els,
        appState: {exportBackground: true, viewBackgroundColor: '#fff'}, files: {}}));
      td.insertAdjacentHTML('beforeend', `<div class=k>${els.length} elements</div>`);
    } catch (e) { td.textContent = 'ERR ' + e; }
  }
}
document.title = 'done';
</script></body></html>"""  # noqa: E501


def png_data_uri(svg_path, size=160):
    """Rasterize an SVG with rsvg-convert and return it as a PNG data URI."""
    png = subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), svg_path],
        capture_output=True,
        check=True,
    ).stdout
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    """Build the gallery from the command line arguments."""
    out, icons_path, cols = sys.argv[1], sys.argv[2], sys.argv[3:]
    icons = json.loads(Path(icons_path).read_text())
    libs = []
    for col in cols:
        title, path = col.split("=", 1)
        lib = json.loads(Path(path).read_text())
        libs.append(
            {"title": title, "items": {i["name"]: i["elements"] for i in lib["libraryItems"]}}
        )
    rows = [
        {
            "name": icon["name"],
            "png": png_data_uri(icon["svg"]),
            "cells": [lib["items"].get(icon["name"]) for lib in libs],
        }
        for icon in icons
    ]
    titles = [lib["title"] for lib in libs]
    html = PAGE.replace("ROWS", json.dumps(rows)).replace("TITLES", json.dumps(titles))
    Path(out).write_text(html)


if __name__ == "__main__":
    main()
