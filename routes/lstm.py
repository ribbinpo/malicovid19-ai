from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd

from utils.transform import seird_params, accumulativeToNon
from services.read_datasets import df_wave
from components.train.seird import process
from components.evaluate.seird import accurate

from assets.models.seird.model import eq

#APIRouter creates path operations for item module
router = APIRouter(
    prefix="/api/lstm",
    tags=["graph"],
    responses={404: {"description": "Not found"}},
)

@router.get('/train')
async def lstm():
  return 0
