# excalidraw-google-cloud-sketch-icons

Hand-drawn style [Excalidraw](https://excalidraw.com) icon libraries (`.excalidrawlib` files) for **Google Cloud**, generated automatically from Google's official product icons.

> [!IMPORTANT]
> **This is an unofficial, community project.** It is not affiliated with, sponsored by, or endorsed by Google LLC. The icons in these libraries are hand-drawn style reinterpretations of Google's official icons, made for use in architecture diagrams.

## Why this project

I really liked the hand-drawn style of the [AWS Architecture Icons](https://github.com/excalidraw/excalidraw-libraries/tree/main/libraries/childishgirl) library by ChildishGirl in the [Excalidraw libraries catalog](https://libraries.excalidraw.com), and I could not find anything similar for Google Cloud. So this project makes one, and keeps it up to date automatically.

## Libraries

| Library | Contents |
| --- | --- |
| Sketch icons for Google Cloud – Products | The 19 core product icons (four-color), for example BigQuery, Cloud Run, GKE, Vertex AI |
| Sketch icons for Google Cloud – Categories | The 26 product category icons (two-color), for example Databases, Networking, Data Analytics |

Google assigns every product that does not have its own core icon to one of the category icons. For example, Pub/Sub and Dataflow use the **Data Analytics** icon. See the [Google Cloud product icons overview (PDF)](https://services.google.com/fh/files/misc/google-cloud-product-icons.pdf) for the full product-to-category list.

The legacy two-tone blue console icons are not included.

## Installation

1. Download one or both libraries (these links always point to the latest release):
   - [**google-cloud-products.excalidrawlib**](https://github.com/gchaperon/excalidraw-google-cloud-sketch-icons/releases/latest/download/google-cloud-products.excalidrawlib)
   - [**google-cloud-categories.excalidrawlib**](https://github.com/gchaperon/excalidraw-google-cloud-sketch-icons/releases/latest/download/google-cloud-categories.excalidrawlib)
2. Drag the downloaded file onto the [Excalidraw](https://excalidraw.com) canvas. The icons are added to your library panel.

   Alternatively, open the library panel, open its `⋯` menu, choose **Open**, and select the file.

Each icon is one group with its name as a label. Ungroup it (`Ctrl+Shift+G`) if you want to edit or delete the label.

Release notes and older versions are on the [releases page](https://github.com/gchaperon/excalidraw-google-cloud-sketch-icons/releases).

## How the icons are made

1. The official icon packs are downloaded from [cloud.google.com/icons](https://cloud.google.com/icons).
2. Each SVG is converted to Excalidraw elements: the center line of every stroke is traced and turned into Excalidraw lines (stroke width 6, roughness 1 "Artist" style, original colors).
3. A vision language model (`google/gemini-4-argon` via [OpenRouter](https://openrouter.ai)) compares the conversion with the official icon and redraws it as a clean, simple sketch. It checks its own result against the original for up to three rounds.
4. The result is packaged as an `.excalidrawlib` file, with the product name as a label under each icon.

The original Google icon files are **not** stored in or redistributed by this repository. They are downloaded from Google at build time.

### Automatic updates

The [Update libraries](.github/workflows/update-libraries.yml) workflow runs every day. It downloads Google's icon packs and compares the SHA-256 of every SVG with `sources.lock.json` from the latest release. If an icon was added, changed, or removed, only those icons are rebuilt, and a new release (`vYYYY.MM.DD`) is published with the two libraries and the updated `sources.lock.json`.

Generated files are never committed to this repository. Releases are the only place where the libraries are published.

### Running the pipeline locally

Requirements: [uv](https://docs.astral.sh/uv/), `rsvg-convert` (librsvg), Chromium or Chrome (set `CHROME_BIN` if it is not called `chromium`), and an OpenRouter API key in `OPENROUTER_API_KEY` or in a `.env` file.

```sh
uv run python -m pipeline.build --check   # report changed icons, no model calls
uv run python -m pipeline.build           # rebuild new/changed icons only
uv run python -m pipeline.build --force   # rebuild every icon
```

The previous state is downloaded from the latest release. Outputs are written to `dist/` (libraries, `sources.lock.json`, release notes) and intermediate images and logs to `build/`. Both folders are git-ignored.

Settings (model, rounds, icon names) are in [`pipeline/config.json`](pipeline/config.json).

## Use of Google's icons and trademarks

- Google states that the product icons in its icon library "may be used freely and without permission to accurately reference Google's technology and tools, for instance in books or architecture diagrams" (Google Cloud Official Icons deck). These libraries are intended for exactly that purpose: architecture diagrams that refer to Google Cloud products.
- Google's [trademark guidelines](https://partnermarketinghub.withgoogle.com/brands/google/trademarks-and-terms/trademark-guidelines-for-proper-usage/) also ask third parties not to alter Google Brand Features and not to imply a relationship with Google. These icons are stylistic reinterpretations, so if you need pixel-exact official artwork (for example in marketing material or co-branding), use the [official icons](https://cloud.google.com/icons) instead.
- If you represent Google and have concerns about this project, please open an issue and it will be addressed promptly.

### Trademark attribution

Google, Google Cloud, AlloyDB, Anthos, Apigee, BigQuery, Cloud Run, Cloud SQL, Cloud Storage, Compute Engine, GKE, Google Kubernetes Engine, Hyperdisk, Looker, Mandiant, Spanner, Vertex AI, and related marks and logos are trademarks of Google LLC. All other product names are trademarks of their respective owners. Their use here is for identification and reference only.

## License

The code and tooling in this repository are licensed under the [MIT License](LICENSE).

The license does not grant any rights to Google's trademarks or to the official icon designs.
