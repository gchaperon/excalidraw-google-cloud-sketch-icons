"""Render Excalidraw elements to PNG with headless chromium (Excalidraw's own exportToSvg)."""
import json, os, subprocess, tempfile

CHROME = os.environ.get('CHROME_BIN', 'chromium')

PAGE = '''<!doctype html><html><head><meta charset=utf-8><style>
html,body{margin:0;background:#fff}#c svg{width:%dpx;height:%dpx;display:block}</style></head>
<body><div id=c></div><script type=module>
import * as X from "https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@18.2.0,react-dom@18.2.0";
const exportToSvg = X.exportToSvg || X.default.exportToSvg;
const els = %s;
const svg = await exportToSvg({elements: els, appState: {exportBackground: true, viewBackgroundColor: '#fff', exportPadding: 0}, files: {}});
svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
document.getElementById('c').appendChild(svg); document.title = 'done';
</script></body></html>'''


def render(elements, out_png, size=320):
    with tempfile.TemporaryDirectory() as d:
        html = os.path.join(d, 'r.html')
        open(html, 'w').write(PAGE % (size, size, json.dumps(elements)))
        subprocess.run(['timeout', '90', CHROME, '--headless=new', '--disable-gpu',
                        f'--user-data-dir={d}/profile', '--virtual-time-budget=20000',
                        f'--window-size={size},{size + 200}', f'--screenshot={os.path.abspath(out_png)}',
                        'file://' + html], capture_output=True)
    if os.path.exists(out_png):
        from PIL import Image
        Image.open(out_png).convert('RGB').crop((0, 0, size, size)).save(out_png)
        return True
    return False


def render_icon(elements, out_png, size=320, box=120):
    """Render without the label, inside an invisible box x box frame, so the image
    uses the same frame as the original icon (no auto-crop to the drawing)."""
    els = [e for e in elements if e.get('type') != 'text']
    if els:
        frame = dict(els[0], id='frame-box', type='rectangle', x=0, y=0, width=box, height=box,
                     strokeColor='transparent', backgroundColor='transparent', roundness=None,
                     groupIds=[])
        for k in ('points', 'lastCommittedPoint', 'startBinding', 'endBinding', 'startArrowhead', 'endArrowhead'):
            frame.pop(k, None)
        els = [frame] + els
    return render(els, out_png, size)
