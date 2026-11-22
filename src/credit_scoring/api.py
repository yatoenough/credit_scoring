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
    model_path = Path(os.environ.get("MODEL_PATH", "models/model.pkl"))
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run `uv run train` first."
        )
    app.state.model = joblib.load(model_path)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/score")
def score(data: ClientData, request: Request):
    bundle = request.app.state.model
    X = pd.DataFrame([data.model_dump()])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return {"approved": p < bundle["threshold"], "default_probability": p}


def serve():
    uvicorn.run(app, host="127.0.0.1", port=3333)
