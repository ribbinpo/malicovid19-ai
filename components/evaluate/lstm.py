import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from utils.metrics import smape

def accurate(trainY_, trainPredict, testPredict, testY_):
  # ------ Train ------
  #I
  rmse_I = math.sqrt(mean_squared_error(trainY_[:,0], trainPredict[:,0]))
  mae_I = mean_absolute_error(trainY_[:,0], trainPredict[:,0])
  r2_I = r2_score(trainY_[:,0], trainPredict[:,0])
  nrmse_I = rmse_I / (trainY_[:,0].max() - trainY_[:,0].min()) #
  mape_I = mean_absolute_percentage_error(trainY_[:,0], trainPredict[:,0])
  smape_I = smape(trainY_[:,0], trainPredict[:,0])
  i_train = [rmse_I, mae_I, r2_I, nrmse_I, mape_I, smape_I]
  #R
  rmse_R = math.sqrt(mean_squared_error(trainY_[:,2], trainPredict[:,2]))
  mae_R = mean_absolute_error(trainY_[:,2], trainPredict[:,2])
  r2_R = r2_score(trainY_[:,2], trainPredict[:,2])
  nrmse_R = rmse_R / (trainY_[:,2].max() - trainY_[:,2].min()) #
  mape_R = mean_absolute_percentage_error(trainY_[:,2], trainPredict[:,2])
  smape_R = smape(trainY_[:,2], trainPredict[:,2])
  r_train = [rmse_R, mae_R, r2_R, nrmse_R, mape_R, smape_R]
  #D
  rmse_D = math.sqrt(mean_squared_error(trainY_[:,1], trainPredict[:,1]))
  mae_D = mean_absolute_error(trainY_[:,1], trainPredict[:,1])
  r2_D = r2_score(trainY_[:,1], trainPredict[:,1])
  nrmse_D = rmse_D / (trainY_[:,1].max() - trainY_[:,1].min()) #
  mape_D =mean_absolute_percentage_error(trainY_[:,1], trainPredict[:,1])
  smape_D = smape(trainY_[:,1], trainPredict[:,1])
  d_train = [rmse_D, mae_D, r2_D, nrmse_D, mape_D, smape_D]
  # ------ Test ------
  #I
  rmse_I = math.sqrt(mean_squared_error(testY_[:,0], testPredict[:,0]))
  mae_I = mean_absolute_error(testY_[:,0], testPredict[:,0])
  r2_I = r2_score(testY_[:,0], testPredict[:,0])
  nrmse_I = rmse_I / (testY_[:,0].max() - testY_[:,0].min()) #
  mape_I =mean_absolute_percentage_error(testY_[:,0], testPredict[:,0])
  smape_I = smape(testY_[:,0], testPredict[:,0])
  i_test = [rmse_I, mae_I, r2_I, nrmse_I, mape_I, smape_I]
  #R
  rmse_R = math.sqrt(mean_squared_error(testY_[:,2], testPredict[:,2]))
  mae_R = mean_absolute_error(testY_[:,2], testPredict[:,2])
  r2_R = r2_score(testY_[:,2], testPredict[:,2])
  nrmse_R = rmse_R / (testY_[:,2].max() - testY_[:,2].min()) #
  mape_R = mean_absolute_percentage_error(testY_[:,2], testPredict[:,2])
  smape_R = smape(testY_[:,2], testPredict[:,2])
  r_test = [rmse_R, mae_R, r2_R, nrmse_R, mape_R, smape_R]
  #D
  rmse_D = math.sqrt(mean_squared_error(testY_[:,1], testPredict[:,1]))
  mae_D = mean_absolute_error(testY_[:,1], testPredict[:,1])
  r2_D = r2_score(testY_[:,1], testPredict[:,1])
  nrmse_D = rmse_D / (testY_[:,1].max() - testY_[:,1].min()) #
  mape_D =mean_absolute_percentage_error(testY_[:,1], testPredict[:,1])
  smape_D = smape(testY_[:,1], testPredict[:,1])
  d_test = [rmse_D, mae_D, r2_D, nrmse_D, mape_D, smape_D]

  data_I = [i_train, i_test]
  data_R = [r_train, r_test]
  data_D = [d_train, d_test]
  df_I = pd.DataFrame(data_I, columns=["rmse", "mae", "r2", "nrmse", "mape", "smape"], index=["train", "test"])
  df_R = pd.DataFrame(data_R, columns=["rmse", "mae", "r2", "nrmse", "mape", "smape"], index=["train", "test"])
  df_D = pd.DataFrame(data_D, columns=["rmse", "mae", "r2", "nrmse", "mape", "smape"], index=["train", "test"])
  return df_I, df_R, df_D