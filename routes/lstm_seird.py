from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import asyncio

from utils.transform import seird_params, accumulativeToNon
from services.read_datasets import df_wave
from components.train.seird import process
from components.evaluate.seird import accurate

from assets.models.seird.model import eq
from assets.datasets.lstm import lstm_seird

#APIRouter creates path operations for item module
router = APIRouter(
    prefix="/api/lstm_seird",
    tags=["graph"],
    responses={404: {"description": "Not found"}},
)

@router.get('/result')
async def lstm():
  await asyncio.sleep(3)
  return lstm_seird
