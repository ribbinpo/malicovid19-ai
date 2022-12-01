from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from components.algorithm.prediction import predictV1,predictV2
from components.algorithm.train import train
from components.algorithm.analysis import getInformation

#APIRouter creates path operations for item module
router = APIRouter(
    prefix="/api/covid19/v1",
    tags=["graph"],
    responses={404: {"description": "Not found"}},
)


@router.get("/predict")
async def prediction():
    result = await predictV1()
    result_json = jsonable_encoder(result)
    return JSONResponse(content=result_json)
    # return json.dumps(result)

@router.get("/covid19lstm")
async def LSTM():
    result = await predictV2()
    result_json = jsonable_encoder(result)
    return JSONResponse(content = result_json)

@router.get("/covid19-information")
def information():
    result = getInformation()
    result_json = jsonable_encoder(result)
    return JSONResponse(content = result_json)

# @router.get("/api/train")
# def training():
#     train()
#     return "success"