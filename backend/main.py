"""VisionTune inference API.

    VISIONTUNE_CONFIG=configs/serve.yaml uvicorn --factory backend.main:create_app --host 0.0.0.0 --port 8000

The model and adapters load once at startup (lifespan). Endpoints follow blueprint §10:
/health /models /model-info /predict /compare /metrics /feedback.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from backend.schemas import CompareResponse, Feedback, Prediction
from visiontune.config import load_config
from visiontune.serving.results import summary as results_summary

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
DISCLAIMER = "Experimental research demo. Answers are model outputs, not verified scientific facts."


def default_engine_factory(cfg: dict[str, Any]):
    from visiontune.env import pin_single_gpu

    pin_single_gpu()
    from visiontune.serving.engine import VQAEngine

    s = cfg["serve"]
    adapters = {k: v for k, v in (s.get("adapters") or {}).items() if Path(v).exists()}
    return VQAEngine(cfg, adapters=adapters, quantized=s.get("quantized_base", True))


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Rate limit exceeded; try again in a minute.")
        q.append(now)


def create_app(cfg: dict[str, Any] | None = None, engine_factory: Callable[[dict], Any] = default_engine_factory) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("VISIONTUNE_CONFIG", "configs/serve.yaml"))
    s = cfg["serve"]
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["engine"] = engine_factory(cfg)
        yield
        state.clear()

    app = FastAPI(title="VisionTune AI", version="0.1.0", description=DISCLAIMER, lifespan=lifespan)
    limiter = RateLimiter(int(s.get("rate_limit_per_minute", 20)))

    def engine():
        if "engine" not in state:
            raise HTTPException(503, "Model is still loading.")
        return state["engine"]

    def rate_limit(request: Request):
        limiter.check(request.client.host if request.client else "unknown")

    async def read_image(upload: UploadFile) -> Image.Image:
        if upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported file type {upload.content_type!r}; use PNG, JPEG or WebP.")
        data = await upload.read(int(s["max_upload_mb"] * 1024 * 1024) + 1)
        if len(data) > s["max_upload_mb"] * 1024 * 1024:
            raise HTTPException(413, f"Image larger than {s['max_upload_mb']} MB.")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(422, "File is not a readable image.")
        if max(image.size) > s["max_image_side"]:
            raise HTTPException(422, f"Image side exceeds {s['max_image_side']} px.")
        return image

    def parse_inputs(question: str, choices: str | None) -> list[str] | None:
        question = question.strip()
        if not question or len(question) > s["max_question_chars"]:
            raise HTTPException(422, f"Question must be 1-{s['max_question_chars']} characters.")
        if not choices:
            return None
        try:
            parsed = json.loads(choices)
        except json.JSONDecodeError:
            parsed = [c for c in choices.split("\n")]
        parsed = [str(c).strip() for c in parsed if str(c).strip()]
        if parsed and not 2 <= len(parsed) <= s["max_choices"]:
            raise HTTPException(422, f"Provide between 2 and {s['max_choices']} choices, or none.")
        return parsed or None

    @app.get("/health")
    def health():
        return {"status": "ok" if "engine" in state else "loading", "disclaimer": DISCLAIMER}

    @app.get("/models")
    def models(eng=Depends(engine)):
        return {"variants": eng.variants, "default": "qlora" if "qlora" in eng.variants else eng.variants[0]}

    @app.get("/model-info")
    def model_info(eng=Depends(engine)):
        return eng.info()

    @app.get("/metrics")
    def metrics():
        return results_summary()

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(rate_limit)])
    async def predict(image: UploadFile = File(...), question: str = Form(...), choices: str | None = Form(None),
                      context: str | None = Form(None), model: str = Form("qlora"), eng=Depends(engine)):
        img = await read_image(image)
        opts = parse_inputs(question, choices)
        if model not in eng.variants:
            raise HTTPException(422, f"Unknown model {model!r}; available: {eng.variants}")
        return await run_in_threadpool(eng.predict, img, question, opts, context, model)

    @app.post("/compare", response_model=CompareResponse, dependencies=[Depends(rate_limit)])
    async def compare(image: UploadFile = File(...), question: str = Form(...), choices: str | None = Form(None),
                      context: str | None = Form(None), eng=Depends(engine)):
        img = await read_image(image)
        opts = parse_inputs(question, choices)
        preds = [await run_in_threadpool(eng.predict, img, question, opts, context, v) for v in eng.variants]
        return {"predictions": preds}

    @app.post("/feedback", dependencies=[Depends(rate_limit)])
    def feedback(item: Feedback):
        path = Path(s.get("feedback_path", "data/feedback/feedback.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({**item.model_dump(), "time": datetime.now(timezone.utc).isoformat()}) + "\n")
        return {"stored": True}

    return app
