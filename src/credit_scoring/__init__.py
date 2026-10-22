import random

import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/score")
def score():
    approved = random.choice([False, True])
    return {"approved": approved}

def serve():
    uvicorn.run(app, host="127.0.0.1", port=3333)
