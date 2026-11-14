"""Build a comparison gallery HTML: one row per icon, columns = original + each library.

Usage: gallery.py OUT.html ICONS.json COL=lib.excalidrawlib [COL=lib.excalidrawlib ...]
ICONS.json: [{"name": "BigQuery", "svg": "path/to.svg"}, ...]
Library items are matched to icons by item name. Render with headless chromium.
"""
import base64, json, subprocess, sys

out, icons_path, cols = sys.argv[1], sys.argv[2], sys.argv[3:]
icons = json.load(open(icons_path))
libs = []
for c in cols:
    title, path = c.split('=', 1)
    d = json.load(open(path))
    libs.append({'title': title, 'items': {i['name']: i['elements'] for i in d['libraryItems']}})
rows = []
for ic in icons:
    png = subprocess.run(['rsvg-convert', '-w', '160', '-h', '160', ic['svg']], capture_output=True, check=True).stdout
    rows.append({'name': ic['name'], 'png': 'data:image/png;base64,' + base64.b64encode(png).decode(),
                 'cells': [lib['items'].get(ic['name']) for lib in libs]})

html = '''<!doctype html><html><head><meta charset=utf-8><style>
body{margin:0;background:#fff;font:13px sans-serif;color:#222}
table{border-collapse:collapse;margin:8px}td,th{border:1px solid #ddd;width:190px;height:200px;text-align:center;vertical-align:middle;padding:4px}
th{height:auto;background:#f4f4f4}td svg{max-width:180px;max-height:190px}td.n{width:90px;font-weight:bold}
.k{font-size:11px;color:#777}</style></head><body><table id=t></table>
<script type=module>
import * as X from "https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@18.2.0,react-dom@18.2.0";
const exportToSvg = X.exportToSvg || X.default.exportToSvg;
const rows = ROWS, titles = TITLES; const t = document.getElementById('t');
t.insertAdjacentHTML('beforeend', '<tr><th>icon</th><th>original</th>' + titles.map(x => `<th>${x}</th>`).join('') + '</tr>');
for (const r of rows) {
  const tr = document.createElement('tr'); t.appendChild(tr);
  tr.insertAdjacentHTML('beforeend', `<td class=n>${r.name}</td><td><img src="${r.png}" width=150></td>`);
  for (const els of r.cells) {
    const td = document.createElement('td'); tr.appendChild(td);
    if (!els) { td.textContent = '—'; continue; }
    try { td.appendChild(await exportToSvg({elements: els, appState: {exportBackground: true, viewBackgroundColor: '#fff'}, files: {}}));
          td.insertAdjacentHTML('beforeend', `<div class=k>${els.length} elements</div>`); }
    catch (e) { td.textContent = 'ERR ' + e; }
  }
}
document.title = 'done';
</script></body></html>'''
html = html.replace('ROWS', json.dumps(rows)).replace('TITLES', json.dumps([l['title'] for l in libs]))
open(out, 'w').write(html)
