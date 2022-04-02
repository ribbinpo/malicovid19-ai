# uvicorn main:app --reload
from typing import Optional
from fastapi import FastAPI
from components.algorithm.prediction import predict
from components.algorithm.train import train
app = FastAPI()

#CORS 
# origins = [
#     "http://localhost:3000",
#     "*"
# ]


@app.get("/")
def read_root():
    return {"data": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.get("/api/predict")
async def prediction():
    result = await predict()
    return result
@app.get("/api/train")
def training():
    train()
    return "success"

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host="0.0.0.0",port=8080)
