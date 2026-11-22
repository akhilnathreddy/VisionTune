"""Demo backend (FastAPI) and frontend (Streamlit) tests, all on CPU."""

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend.main import create_app  # noqa: E402
from conftest import make_image  # noqa: E402


def png_bytes(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    make_image(1, size=size).save(buf, format="PNG")
    return buf.getvalue()


class FakeEngine:
    variants = ["base", "qlora"]

    def __init__(self):
        self.calls = []

    def info(self):
        return {"base_model": "fake", "variants": self.variants}

    def predict(self, image, question, choices, context, variant):
        self.calls.append((image.size, question, choices, context, variant))
        return {"model": variant, "answer": "A. whale" if choices else "whale", "answer_index": 0 if choices else None,
                "explanation": "Whales breathe air.", "extraction": "strict" if choices else "open",
                "raw_output": "Answer: A. whale\nExplanation: Whales breathe air.", "latency_s": 0.01, "new_tokens": 9}


@pytest.fixture
def serve_cfg(tmp_path):
    return {"model": {"name_or_path": "fake"}, "serve": {
        "max_upload_mb": 1, "max_image_side": 512, "max_question_chars": 200, "max_choices": 5,
        "rate_limit_per_minute": 0, "feedback_path": str(tmp_path / "fb.jsonl")}}


@pytest.fixture
def client(serve_cfg):
    engine = FakeEngine()
    with TestClient(create_app(serve_cfg, engine_factory=lambda cfg: engine)) as c:
        c.engine = engine
        yield c


def post(client, path="/predict", data=None, content=None, ctype="image/png"):
    files = {"image": ("x.png", content if content is not None else png_bytes(), ctype)}
    return client.post(path, files=files, data={"question": "Which is a mammal?", **(data or {})})


def test_health_models_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/models").json() == {"variants": ["base", "qlora"], "default": "qlora"}
    assert client.get("/model-info").json()["base_model"] == "fake"


def test_predict_multiple_choice_and_open(client):
    r = post(client, data={"choices": json.dumps(["whale", "shark"]), "model": "qlora"})
    assert r.status_code == 200 and r.json()["answer"] == "A. whale"
    assert client.engine.calls[-1][2] == ["whale", "shark"]
    r = post(client, data={"choices": "whale\nshark", "model": "base"})   # newline-separated also accepted
    assert r.status_code == 200 and client.engine.calls[-1][2] == ["whale", "shark"]
    r = post(client, data={"model": "base"})                              # no choices: open question
    assert r.status_code == 200 and r.json()["extraction"] == "open"


@pytest.mark.parametrize("kwargs,status", [
    ({"ctype": "application/pdf"}, 415),
    ({"content": b"x" * (1024 * 1024 + 10)}, 413),
    ({"content": b"not an image"}, 422),
    ({"content": png_bytes((600, 40))}, 422),                               # side > max_image_side
    ({"data": {"choices": json.dumps(["only one"])}}, 422),
    ({"data": {"choices": json.dumps([str(i) for i in range(6)])}}, 422),
    ({"data": {"model": "gpt"}}, 422),
    ({"data": {"question": "   "}}, 422),
])
def test_predict_validation(client, kwargs, status):
    assert post(client, **kwargs).status_code == status


def test_compare_runs_every_variant(client):
    r = post(client, "/compare", data={"choices": json.dumps(["whale", "shark"])})
    assert [p["model"] for p in r.json()["predictions"]] == ["base", "qlora"]


def test_feedback_is_appended(client, serve_cfg):
    r = client.post("/feedback", json={"model": "qlora", "question": "q", "answer": "A", "correct": False})
    assert r.json() == {"stored": True}
    line = json.loads(open(serve_cfg["serve"]["feedback_path"]).read().strip())
    assert line["correct"] is False and "time" in line


def test_rate_limit(serve_cfg):
    serve_cfg["serve"]["rate_limit_per_minute"] = 2
    with TestClient(create_app(serve_cfg, engine_factory=lambda cfg: FakeEngine())) as c:
        codes = [post(c).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_metrics_reads_committed_results(client):
    m = client.get("/metrics").json()
    assert m["scienceqa_test"]["n"] == 2017 and set(m["scienceqa_test"]["models"]) == {"base", "qlora", "lora"}
    assert m["ai2d_test"]["n"] == 3088
    assert any(r["n_train"] == 6218 for r in m["scienceqa_validation_scaling"])


def test_real_engine_switches_adapters(tiny_model_dir, tmp_path):
    """Tiny random Qwen2.5-VL + two adapters with non-zero LoRA-B: base and adapter outputs
    come from the same loaded model and switching must not leak state between requests."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5_VLForConditionalGeneration

    from visiontune.serving.engine import VQAEngine

    paths = {}
    for i, name in enumerate(["qlora", "lora"]):
        torch.manual_seed(10 + i)
        m = get_peft_model(Qwen2_5_VLForConditionalGeneration.from_pretrained(tiny_model_dir),
                           LoraConfig(r=4, lora_alpha=64, target_modules=["q_proj", "v_proj"]))
        for n, p in m.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.5)
        paths[name] = tmp_path / name
        m.save_pretrained(paths[name])
    cfg = {"model": {"name_or_path": str(tiny_model_dir), "dtype": "fp32"},
           "processor": {"min_pixels": 16 * 28 * 28, "max_pixels": 32 * 28 * 28},
           "prompt": {"system": "You are a science tutor."}, "generation": {"max_new_tokens": 6}}
    engine = VQAEngine(cfg, adapters=paths, quantized=False)
    assert engine.variants == ["base", "qlora", "lora"]
    img = make_image(3)
    outs = {v: engine.predict(img, "Which is a mammal?", ["whale", "shark"], None, v)["raw_output"]
            for v in ["base", "qlora", "base", "lora"]}
    first_base = engine.predict(img, "Which is a mammal?", ["whale", "shark"], None, "base")["raw_output"]
    assert first_base == outs["base"]                       # base is unaffected by earlier adapter calls
    assert len({outs["base"], outs["qlora"], outs["lora"]}) > 1
    open_res = engine.predict(img, "What animal is this?", None, None, "qlora")
    assert open_res["extraction"] == "open" and open_res["answer_index"] is None
    with pytest.raises(ValueError):
        engine.predict(img, "q", None, None, "nope")


def test_streamlit_pages_render_without_backend(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:9")    # nothing listens here
    for page in ["Playground", "Compare models", "Evaluation", "About"]:
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"), default_timeout=60)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
    assert any("ScienceQA test" in s.value for s in at.subheader) or page == "About"
