"""Pipeline that builds hand-drawn Excalidraw libraries from Google Cloud's official icons.

Modules:
    build: Entry point. Downloads the icons, detects changes, and writes the libraries.
    convert: Approach B. Deterministic SVG to Excalidraw conversion.
    redraw: Approach C. An LLM redraws each icon, starting from the approach B result.
    compact: Compact shape format used in the LLM prompts, and the shared text label.
    render: Renders Excalidraw elements to PNG with headless Chrome.
"""
