import joblib
import uvicorn
from fastapi import FastAPI

from credit_scoring.models import ClientData

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/score")
def score(data: ClientData):
    features = [data.age, data.income, data.education, data.work, data.car]
    approved = not model.predict([features])[0].item()
    return {"approved": approved}

def serve():
    uvicorn.run(app, host="127.0.0.1", port=3333)
