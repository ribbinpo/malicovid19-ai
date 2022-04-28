# uvicorn main:app --reload
from typing import Optional
from fastapi import FastAPI
from components.algorithm.prediction import predictV1,predictV2
from components.algorithm.train import train
from components.algorithm.analysis import getInformation

app = FastAPI()
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

#CORS 
# origins = [
#     "http://localhost:3000",
#     "*"
# ]


# @app.get("/")
# def read_root():
#     return {"data": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.get("/api/predict")
async def prediction():
    result = await predictV1()
    result_json = jsonable_encoder(result)
    return JSONResponse(content=result_json)
    # return json.dumps(result)

@app.get("/api/covid19lstm")
async def LSTM():
    result = await predictV2()
    result_json = jsonable_encoder(result)
    return JSONResponse(content = result_json)

@app.get("/api/covid19-information")
def information():
    result = getInformation()
    result_json = jsonable_encoder(result)
    return JSONResponse(content = result_json)

@app.get("/api/train")
def training():
    train()
    return "success"

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host="0.0.0.0",port=8080)
