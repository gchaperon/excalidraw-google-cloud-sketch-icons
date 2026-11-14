# excalidraw-google-cloud-sketch-icons

Hand-drawn style [Excalidraw](https://excalidraw.com) icon libraries (`.excalidrawlib` files) for **Google Cloud**, generated automatically from Google's official product icons.

> [!IMPORTANT]
> **This is an unofficial, community project.** It is not affiliated with, sponsored by, or endorsed by Google LLC. The icons in these libraries are hand-drawn style reinterpretations of Google's official icons, made for use in architecture diagrams.

## Libraries

| Library | Contents |
| --- | --- |
| Sketch icons for Google Cloud – Products | The 19 core product icons (four-color), for example BigQuery, Cloud Run, GKE, Vertex AI |
| Sketch icons for Google Cloud – Categories | The 26 product category icons (two-color), for example Databases, Networking, Data Analytics |

Google assigns every product that does not have its own core icon to one of the category icons. For example, Pub/Sub and Dataflow use the **Data Analytics** icon. See the [Google Cloud product icons overview (PDF)](https://services.google.com/fh/files/misc/google-cloud-product-icons.pdf) for the full product-to-category list.

The legacy two-tone blue console icons are not included.

## Installation

1. Download `google-cloud-products.excalidrawlib` and/or `google-cloud-categories.excalidrawlib` from the [latest release](https://github.com/gchaperon/excalidraw-google-cloud-sketch-icons/releases/latest).
2. In Excalidraw, open the library panel (press `9`), open its menu, and choose **Load**. Select the downloaded file.

Each icon is one group with its name as a label. Ungroup it if you want to edit or delete the label.

## How the icons are made

1. The official icon packs are downloaded from [cloud.google.com/icons](https://cloud.google.com/icons).
2. Each SVG is converted to Excalidraw elements: the center line of every stroke is traced and turned into Excalidraw lines (stroke width 6, roughness 1 "Artist" style, original colors).
3. A vision language model (`google/gemini-4-argon` via [OpenRouter](https://openrouter.ai)) compares the conversion with the official icon and redraws it as a clean, simple sketch. It checks its own result against the original for up to three rounds.
4. The result is packaged as an `.excalidrawlib` file, with the product name as a label under each icon.

The original Google icon files are **not** stored in or redistributed by this repository. They are downloaded from Google at build time.

### Automatic updates

The [Update libraries](.github/workflows/update-libraries.yml) workflow runs every day. It downloads Google's icon packs and compares the SHA-256 of every SVG with [`sources.lock.json`](sources.lock.json). If an icon was added, changed, or removed, only those icons are rebuilt, the libraries and the lock file are committed, and a new release (`vYYYY.MM.DD`) is published with the two library files.

### Running the pipeline locally

Requirements: [uv](https://docs.astral.sh/uv/), `rsvg-convert` (librsvg), Chromium or Chrome (set `CHROME_BIN` if it is not called `chromium`), and an OpenRouter API key in `OPENROUTER_API_KEY` or in a `.env` file.

```sh
uv run python -m pipeline.build --check   # report changed icons, no model calls
uv run python -m pipeline.build           # rebuild new/changed icons only
uv run python -m pipeline.build --force   # rebuild every icon
```

Settings (model, rounds, icon names) are in [`pipeline/config.json`](pipeline/config.json). Intermediate images and logs are written to `build/`.

## Use of Google's icons and trademarks

- Google states that the product icons in its icon library "may be used freely and without permission to accurately reference Google's technology and tools, for instance in books or architecture diagrams" (Google Cloud Official Icons deck). These libraries are intended for exactly that purpose: architecture diagrams that refer to Google Cloud products.
- Google's [trademark guidelines](https://partnermarketinghub.withgoogle.com/brands/google/trademarks-and-terms/trademark-guidelines-for-proper-usage/) also ask third parties not to alter Google Brand Features and not to imply a relationship with Google. These icons are stylistic reinterpretations, so if you need pixel-exact official artwork (for example in marketing material or co-branding), use the [official icons](https://cloud.google.com/icons) instead.
- If you represent Google and have concerns about this project, please open an issue and it will be addressed promptly.

### Trademark attribution

Google, Google Cloud, AlloyDB, Anthos, Apigee, BigQuery, Cloud Run, Cloud SQL, Cloud Storage, Compute Engine, GKE, Google Kubernetes Engine, Hyperdisk, Looker, Mandiant, Spanner, Vertex AI, and related marks and logos are trademarks of Google LLC. All other product names are trademarks of their respective owners. Their use here is for identification and reference only.

## License

The code and tooling in this repository are licensed under the [MIT License](LICENSE).

The license does not grant any rights to Google's trademarks or to the official icon designs.
