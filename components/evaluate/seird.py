import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from utils.metrics import smape

def accurate(I, R, D, df):
  #I
  rmse_I = math.sqrt(mean_squared_error(I, df['confirmed']))
  mae_I = mean_absolute_error(I, df['confirmed'])
  r2_I = r2_score(I, df['confirmed'])
  nrmse_I = rmse_I / (df['confirmed'].max() - df['confirmed'].min())
  mape_I = mean_absolute_percentage_error(I, df['confirmed'])
  # smape_I = smape(I, df['confirmed'])
  i = [rmse_I, mae_I, r2_I, nrmse_I, mape_I]
  #R
  rmse_R = math.sqrt(mean_squared_error(R, df['recovered']))
  mae_R = mean_absolute_error(R, df['recovered'])
  r2_R = r2_score(R, df['recovered'])
  nrmse_R = rmse_R / (df['recovered'].max() - df['recovered'].min())
  mape_R = mean_absolute_percentage_error(R, df['recovered'])
  # smape_R = smape(R, df['recovered'])
  r = [rmse_R, mae_R, r2_R, nrmse_R, mape_R]
  #D
  rmse_D = math.sqrt(mean_squared_error(D, df['death']))
  mae_D = mean_absolute_error(D, df['death'])
  r2_D = r2_score(D, df['death'])
  nrmse_D = rmse_D / (df['death'].max() - df['death'].min())
  mape_D =mean_absolute_percentage_error(D, df['death'])
  # smape_D = smape(D, df['death'])
  d = [rmse_D, mae_D, r2_D, nrmse_D, mape_D]
  data = [i, r, d]
  acc = pd.DataFrame(data, columns=["rmse", "mae", "r2", "nrmse", "mape"], index=["I", "R", "D"])
  return acc