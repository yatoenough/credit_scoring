import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request

from credit_scoring.models import ClientData


@asynccontextmanager
async def lifespan(app: FastAPI):
    target_path = Path(os.environ.get("TARGET_PATH", "target/model.pkl"))
    if not target_path.exists():
        raise RuntimeError(
            f"Model not found at {target_path}. Run `uv run train` first."
        )
    app.state.model = joblib.load(target_path)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/score")
def score(data: ClientData, request: Request):
    bundle = request.app.state.model
    X = pd.DataFrame([data.model_dump()])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return {"approved": p < bundle["threshold"], "default_probability": p}


@app.get("/health")
def health():
    return {"status": "ok"}


def serve():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "3333"))
    uvicorn.run(app, host=host, port=port)
