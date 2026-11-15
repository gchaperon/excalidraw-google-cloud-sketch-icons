import io
import json
import types

import pytest

from pipeline import redraw

from .conftest import DATA, needs_rsvg

SHAPES = [{"t": "ellipse", "c": "#4285f4", "x": 10, "y": 10, "w": 50, "h": 50}]


def answer(shapes, note="Made it simpler."):
    return f"{note}\n```json\n{json.dumps({'shapes': shapes})}\n```"


# --- parse ---------------------------------------------------------------------------------


def test_parse_returns_shapes_and_note():
    assert redraw.parse(answer(SHAPES)) == (SHAPES, "Made it simpler.")


def test_parse_uses_the_last_json_block():
    msg = (
        f'Note.\n```json\n{{"shapes": []}}\n```\nFixed:\n```\n{json.dumps({"shapes": SHAPES})}\n```'
    )

    assert redraw.parse(msg) == (SHAPES, "Note.")


@pytest.mark.parametrize(
    "msg", [None, "", "no code block", "```json\n{not json}\n```", '```json\n{"other": 1}\n```']
)
def test_parse_rejects_invalid_answers(msg):
    assert redraw.parse(msg) == (None, "")


# --- key -----------------------------------------------------------------------------------


def test_key_reads_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")

    assert redraw.key() == "from-env"


@pytest.mark.parametrize("value", [None, ""])
def test_key_missing_exits(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        redraw.key()


# --- ask -----------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body, self.text = status_code, body, json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def post(monkeypatch, tmp_path):
    """Replace requests.post with a queue of responses; record the requests."""
    responses, requests = [], []

    def fake_post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses.pop(0)

    monkeypatch.setattr(redraw.requests, "post", fake_post)
    monkeypatch.setattr(redraw.time, "sleep", lambda s: None)
    image = tmp_path / "a.png"
    image.write_bytes(b"png")
    return types.SimpleNamespace(responses=responses, requests=requests, image=image)


def ok(content, cost=0.25):
    return FakeResponse(
        200, {"choices": [{"message": {"content": content}}], "usage": {"cost": cost}}
    )


def test_ask_sends_text_then_images_and_returns_cost(post):
    post.responses.append(ok("hello"))

    assert redraw.ask("some/model", "k", "prompt", [post.image], io.StringIO()) == ("hello", 0.25)
    (body,) = post.requests
    assert body["model"] == "some/model"
    assert body["messages"][0] == {"role": "system", "content": redraw.STYLE}
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "prompt"}
    assert user[1]["image_url"]["url"] == "data:image/png;base64,cG5n"


def test_ask_retries_errors(post):
    post.responses += [FakeResponse(429, {}), FakeResponse(500, {}), ok("hello")]
    log = io.StringIO()

    assert redraw.ask("m", "k", "prompt", [], log) == ("hello", 0.25)
    assert "HTTP 429" in log.getvalue()
    assert "HTTP 500" in log.getvalue()


def test_ask_gives_up_after_three_attempts(post):
    post.responses += [FakeResponse(500, {})] * 3

    assert redraw.ask("m", "k", "prompt", [], io.StringIO()) == (None, 0.0)


def test_ask_without_choices(post):
    post.responses.append(FakeResponse(200, {"error": "x"}))

    assert redraw.ask("m", "k", "prompt", [], io.StringIO()) == (None, 0.0)


# --- process -------------------------------------------------------------------------------


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model and the Chrome renderer. Answers are taken from a queue."""
    answers, calls, renders = [], [], []

    def fake_ask(model, k, text, images, log):
        calls.append((text, [p.name for p in images]))
        log.write("  (fake call)\n")
        return answers.pop(0), 0.5

    def fake_render(elements, out_png):
        renders.append(out_png.name)
        out_png.write_bytes(b"png")
        return True

    monkeypatch.setattr(redraw, "ask", fake_ask)
    monkeypatch.setattr(redraw, "render", fake_render)
    return types.SimpleNamespace(answers=answers, calls=calls, renders=renders)


B_ELEMENTS = [
    {
        "type": "ellipse",
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
        "strokeColor": "#000000",
        "backgroundColor": "transparent",
    }
]


@needs_rsvg
def test_process_keeps_the_last_valid_answer(fake_model, tmp_path):
    better = [{**SHAPES[0], "x": 20}]
    fake_model.answers += [answer(SHAPES), "garbage", answer(better)]
    log = io.StringIO()
    opts = types.SimpleNamespace(model="m", rounds=3)

    item, cost = redraw.process(
        opts, "k", tmp_path, "Big Query", DATA / "blobs.svg", B_ELEMENTS, log
    )

    assert cost == 1.5
    assert item["id"] == "c-Big_Query"
    assert item["name"] == "Big Query"
    assert [e["type"] for e in item["elements"]] == ["ellipse", "text"]
    assert item["elements"][0]["x"] == 20
    assert json.loads((tmp_path / "Big_Query_shapes.json").read_text()) == better
    # Round 1 sees the approach B rendering; later rounds see the last valid answer.
    assert [images for _, images in fake_model.calls] == [
        ["Big_Query_orig.png", "Big_Query_b.png"],
        ["Big_Query_orig.png", "Big_Query_r1.png"],
        ["Big_Query_orig.png", "Big_Query_r1.png"],
    ]
    assert "Icon: Big Query" in fake_model.calls[0][0]
    assert '"x": 1.0' in fake_model.calls[0][0]  # approach B shapes are in the prompt
    assert "could not parse answer" in log.getvalue()


@needs_rsvg
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="known bug: after an invalid round 1 answer, round 2 has no shapes and no image "
    "(cur_png is None), so the real ask() fails and the icon falls back to approach B",
)
def test_process_without_any_valid_answer(fake_model, tmp_path):
    fake_model.answers += ["garbage", "garbage"]
    opts = types.SimpleNamespace(model="m", rounds=2)

    item, cost = redraw.process(
        opts, "k", tmp_path, "X", DATA / "blobs.svg", B_ELEMENTS, io.StringIO()
    )

    assert (item, cost) == (None, 1.0)
