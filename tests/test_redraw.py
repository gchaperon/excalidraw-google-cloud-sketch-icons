import json
import types

import httpx2
import pytest

from pipeline import redraw

from .conftest import DATA, FakeRenderer, mock_client, needs_rsvg

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


@pytest.fixture
def model(monkeypatch, tmp_path):
    """An OpenRouter client that answers from a queue; it records the request bodies.

    A queue entry is a Response, or an exception to raise (a network error).
    """
    responses, bodies = [], []

    def handler(request):
        bodies.append(json.loads(request.content))
        answer = responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(redraw, "_retry_delay", lambda response, attempt: 0)
    image = tmp_path / "a.png"
    image.write_bytes(b"png")
    return types.SimpleNamespace(
        http=mock_client(handler, base_url=redraw.API_URL),
        responses=responses,
        bodies=bodies,
        image=image,
    )


def ok(content, cost=0.25):
    body = {"choices": [{"message": {"content": content}}], "usage": {"cost": cost}}
    return httpx2.Response(200, json=body)


@pytest.mark.anyio
async def test_ask_sends_text_then_images_and_returns_cost(model):
    model.responses.append(ok("hello"))

    answer = await redraw.ask(model.http, "some/model", "prompt", [model.image], "X")

    assert answer == ("hello", 0.25)
    (body,) = model.bodies
    assert body["model"] == "some/model"
    assert body["messages"][0] == {"role": "system", "content": redraw.STYLE}
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "prompt"}
    assert user[1]["image_url"]["url"] == "data:image/png;base64,cG5n"


@pytest.mark.anyio
async def test_ask_retries_server_and_network_errors(model, caplog):
    model.responses += [httpx2.Response(429), httpx2.ConnectError("down"), ok("hello")]

    assert await redraw.ask(model.http, "m", "prompt", [], "X") == ("hello", 0.25)
    assert "X: HTTP 429" in caplog.text
    assert "X: ConnectError: down" in caplog.text


@pytest.mark.anyio
async def test_ask_gives_up_after_three_attempts(model):
    model.responses += [httpx2.Response(500)] * 3

    assert await redraw.ask(model.http, "m", "prompt", [], "X") == (None, 0.0)


@pytest.mark.anyio
async def test_ask_does_not_retry_client_errors(model):
    model.responses += [httpx2.Response(401), ok("never sent")]

    assert await redraw.ask(model.http, "m", "prompt", [], "X") == (None, 0.0)
    assert len(model.bodies) == 1


@pytest.mark.anyio
async def test_ask_without_choices(model):
    model.responses.append(httpx2.Response(200, json={"error": "x"}))

    assert await redraw.ask(model.http, "m", "prompt", [], "X") == (None, 0.0)


@pytest.mark.parametrize(
    ("headers", "attempt", "delay"),
    [
        ({}, 0, 5.0),
        ({}, 1, 10.0),
        ({"Retry-After": "7"}, 0, 7.0),
        ({"Retry-After": "600"}, 0, 60.0),
    ],
)
def test_retry_delay(headers, attempt, delay):
    assert redraw._retry_delay(httpx2.Response(429, headers=headers), attempt) == delay


# --- process -------------------------------------------------------------------------------


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model and the Chrome renderer. Answers are taken from a queue."""
    answers, calls = [], []

    async def fake_ask(http, model, text, images, icon):
        calls.append((text, [p.name for p in images]))
        return answers.pop(0), 0.5

    monkeypatch.setattr(redraw, "ask", fake_ask)
    return types.SimpleNamespace(answers=answers, calls=calls, renderer=FakeRenderer())


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
@pytest.mark.anyio
async def test_process_keeps_the_last_valid_answer(fake_model, tmp_path, caplog):
    better = [{**SHAPES[0], "x": 20}]
    fake_model.answers += [answer(SHAPES), "garbage", answer(better)]
    item, cost = await redraw.process(
        httpx2.AsyncClient(),
        fake_model.renderer,
        "m",
        3,
        tmp_path,
        "Big Query",
        DATA / "blobs.svg",
        B_ELEMENTS,
    )

    assert cost == 1.5
    assert item is not None
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
    assert "Big Query round 2: could not parse answer" in caplog.text
    # The approach B drawing and each valid answer are rendered, without the label.
    rendered = fake_model.renderer.calls
    assert [png for png, _ in rendered] == [
        "Big_Query_b.png",
        "Big_Query_r1.png",
        "Big_Query_r3.png",
    ]
    assert all(e["type"] != "text" for _, els in rendered for e in els)


@needs_rsvg
@pytest.mark.anyio
async def test_process_without_any_valid_answer(fake_model, tmp_path):
    fake_model.answers += ["garbage", "garbage"]
    item, cost = await redraw.process(
        httpx2.AsyncClient(),
        fake_model.renderer,
        "m",
        2,
        tmp_path,
        "X",
        DATA / "blobs.svg",
        B_ELEMENTS,
    )

    assert (item, cost) == (None, 1.0)


@needs_rsvg
@pytest.mark.anyio
async def test_process_repeats_the_first_prompt_until_an_answer_is_valid(fake_model, tmp_path):
    fake_model.answers += [None, "garbage", answer(SHAPES)]
    item, cost = await redraw.process(
        httpx2.AsyncClient(),
        fake_model.renderer,
        "m",
        3,
        tmp_path,
        "X",
        DATA / "blobs.svg",
        B_ELEMENTS,
    )

    assert cost == 1.5
    assert item is not None
    assert item["elements"][0]["x"] == 10
    assert [images for _, images in fake_model.calls] == [["X_orig.png", "X_b.png"]] * 3
    assert all("Icon: X" in text for text, _ in fake_model.calls)


@needs_rsvg
@pytest.mark.anyio
async def test_process_drops_answers_that_do_not_render(fake_model, tmp_path, caplog):
    second, third = [{**SHAPES[0], "x": 20}], [{**SHAPES[0], "x": 30}]
    fake_model.answers += [answer(SHAPES), answer(second), answer(third)]
    renderer = FakeRenderer(fail={"X_r2.png"})

    item, cost = await redraw.process(
        httpx2.AsyncClient(), renderer, "m", 3, tmp_path, "X", DATA / "blobs.svg", B_ELEMENTS
    )

    assert cost == 1.5  # the cost of every round is counted
    assert item is not None
    assert item["elements"][0]["x"] == 30
    # Round 3 continues from the last answer that rendered (round 1).
    assert [images for _, images in fake_model.calls][2] == ["X_orig.png", "X_r1.png"]
    assert "X round 2: answer dropped, render failed: X_r2.png: fake failure" in caplog.text


@needs_rsvg
@pytest.mark.anyio
async def test_process_logs_every_answer(fake_model, tmp_path, caplog):
    caplog.set_level("DEBUG", logger="pipeline")
    fake_model.answers += ["garbage"]

    await redraw.process(
        httpx2.AsyncClient(), FakeRenderer(), "m", 1, tmp_path, "X", DATA / "blobs.svg", B_ELEMENTS
    )

    assert "X round 1 answer: garbage" in caplog.text
