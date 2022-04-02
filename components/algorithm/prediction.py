import numpy as np 
import pandas as pd
from tensorflow.keras.models import load_model
import datetime

#joblib for read file scaler
import joblib

model = None
# Setup impotent var
async def configModel():
    #load model .h5
    model = load_model("assets/models/LSTM/model1.h5")
    #load scaler .save
    scaler = joblib.load("assets/models/LSTM/scaler1.save")
    #load local dataset or api(request)
    df = pd.read_csv("assets/datasets/dataset1.csv")
    return model,scaler,df
# Prepare data for prediction or train model
async def preprocess_data(df):
    data = scaler.transform(df.reshape(-1,1))
    data_scaled = data[-shift_data:]
    return data_scaled
# Predict data from model -get accuracy/RMSE MAE ->Error
async def get_predict(data, next):
    prediction_list = data[-shift_data:]
    index = []
    dates = []
    date = datetime.datetime.strptime(df["txn_date"].values[-1] , '%Y-%m-%d').date()
    for i in range(next):
        x = prediction_list[-shift_data:]
        x = x.reshape((1, shift_data, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
        dates.append(date + datetime.timedelta(days=i+1))
    prediction_list = prediction_list[shift_data-1:]
    prediction_list =scaler.inverse_transform(np.array(prediction_list).reshape(-1,1)).reshape(-1).tolist()
    for i in range(len(prediction_list)):
        prediction_list[i] = int(prediction_list[i])
    return dates,prediction_list
# Final Predict and setting data format
async def predict():
    global model,scaler,df, shift_data
    shift_data = 7
    preds = []
    if (model is None) or (scaler is None) or (df is None):
        model, scaler, df = await configModel()
    df_pre = await preprocess_data(df["new_case"].values)
    y_pred = await get_predict(df_pre,7)
    date, data = y_pred

    for i in range(len(date)):
        pred = {}
        pred["date"] = datetime.datetime.strptime(str(date[i]),'%Y-%m-%d')
        pred["case"] = data[i]
        preds.append(pred)
    return preds
# Train model
async def trainModel():
    model, scaler, df = await configModel()
    return 0