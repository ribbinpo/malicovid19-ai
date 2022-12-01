from fastapi import APIRouter
import pandas as pd

from utils.transform import seird_params, accumulativeToNon
from services.read_datasets import df_wave
from components.train.seird import process
from components.evaluate.seird import accurate

from assets.models.seird.model import eq

#APIRouter creates path operations for item module
router = APIRouter(
    prefix="/api/seir",
    tags=["graph"],
    responses={404: {"description": "Not found"}},
)

@router.get("/train")
async def SEIR(wave: int):
  df_params = pd.read_csv('assets/models/seird/seird_params.csv')
  params = seird_params(df_params, wave)
  result, ranges = await process(eq, params)
  S, E, I, R, D = result
  new_R = accumulativeToNon(R)
  new_D = accumulativeToNon(D)
  acc = accurate(I, new_R, new_D, df_wave(wave))
  acc = {
    'I': {
      'rmse': acc['rmse']['I'],
      'mae': acc['mae']['I'],
      'r2': acc['r2']['I'],
      'nrmse': acc['nrmse']['I'],
      'mape': acc['mape']['I'],
    },
    'R': {
      'rmse': acc['rmse']['R'],
      'mae': acc['mae']['R'],
      'r2': acc['r2']['R'],
      'nrmse': acc['nrmse']['R'],
      'mape': acc['mape']['R'],
    },
    'D': {
      'rmse': acc['rmse']['D'],
      'mae': acc['mae']['D'],
      'r2': acc['r2']['D'],
      'nrmse': acc['nrmse']['D'],
      'mape': acc['mape']['D'],
    }
  }
  print(acc)
  return {
    'acc': acc,
    'data': {
      'S': list(S),
      'E': list(E),
      'I': list(I),
      'R': list(new_R),
      'D': list(new_D)
    },
  }
