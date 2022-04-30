import numpy as np 
import pandas as pd
from keras.models import load_model
import datetime
import os
#joblib for read file scaler
import joblib

model = None

# Setup impotent var
async def configModel():
    file = os.listdir("assets/models/LSTM")
    for i in range(len(file)):
        if not (file[i].startswith("model")):
            modelPath = file[i-1]
            break
    scalerPath = file[-1]
    #load model .h5
    model = load_model(os.path.join("assets/models/LSTM/",modelPath))
    #load scaler .save
    scaler = joblib.load(os.path.join("assets/models/LSTM/",scalerPath))
    #load local dataset or api(request)
    df = pd.read_csv(os.path.join("assets/datasets/",os.listdir("assets/datasets/")[-1]))
    return model,scaler,df

# Prepare data for prediction or train model
async def preprocess_data(df):
    data = scaler.fit_transform(df.reshape(-1,1))
    # data_scaled = data[-look_back:]
    return data

#Function for create dataset for each batch
async def createDataset(cols,lookback=1):
    dataX = []
    dataY = []
    for i in range(len(cols)-lookback-1):
        col = cols[i:(i+lookback)]
        dataX.append(col)
        dataY.append(cols[i+lookback])
    return np.array(dataX), np.array(dataY)

async def get_predict(X,df_pre,look_back,batch_size):
    pred = model.predict(X, batch_size=batch_size)
    pred = scaler.inverse_transform(pred)
    predPlot = np.empty_like(df_pre)
    predPlot[:, :] = np.nan
    predPlot[look_back:len(pred)+look_back,:] = pred
    return predPlot

# Reformat
async def reformat(predictData,predictDataFuture):
    predicts = []
    raws = []
    dates = []
    dataSums = []
    #Predicts
    predictData = predictData.tolist()
    #Raws
    # df = df.reset_index()
    raw_case = df["new_case"].values.tolist()
    date = df["txn_date"].values.tolist()
    #Old case
    for i in range(len(predictData)):
        # dataSum = {}
        if str(predictData[i][0]) == "nan":
            predictData[i][0] = None
        else:
            predictData[i][0] = round(predictData[i][0]) 
        predicts.append((predictData[i][0]))
        raws.append(raw_case[i])
        dates.append(date[i])
        # dataSum["date"] = date[i]
        # dataSum["real"] = raw_case[i]
        # dataSum["forecast"] = predictData[i][0]
        # dataSums.append(dataSum) 
    predicts[-1] = raws[-1]
    # dataSums[-1]["forecast"] = int(dataSum["real"])+5
    #TotalCase
    totalCase=0
    newDeath=df["new_death"].values.tolist()[-10:]
    for i in range(10):
        totalCase = totalCase+raw_case[-10:][i]-newDeath[i]
    #New case
    # pre = scaler.inverse_transform(np.array(predictDataFuture).reshape(-1,1)).reshape(-1).tolist()
    pre = predictDataFuture
    dto = datetime.datetime.strptime(date[-1], '%Y-%m-%d').date()
    for i in range(len(pre)):
        # dataSum = {}
        dates.append(str(dto + datetime.timedelta(days=i+1)))
        predicts.append(round(pre[i]))
        raws.append(None)
        # dataSum["date"] = str(dto + datetime.timedelta(days=i+1))
        # dataSum["forecast"] = round(pre[i])
        # dataSum["real"] = None
        # dataSums.append(dataSum)
    newData = {}
    # data
    dataSums = {"date":dates,"real":raws,"forecast":predicts}
    newData["date"] = date[-1]
    newData["data"] = dataSums
    newData["accuracy"] = round(100-(0.3682260946322276*100))
    # newData["accuracy"] = round(100-trainScore)
    newData["totalCase"] = totalCase
    newData["predictTomorrow"] = round(pre[0])
    newData["todayCase"] = raw_case[-1]
    return newData

# Final Predict and setting data format
async def predictV1():
    global model,scaler,df,look_back
    look_back = 7
    newData = 0
    if (model is None) or (scaler is None) or (df is None):
        model, scaler, df = await configModel()
    df_pre = await preprocess_data(df["new_case"].values)
    # create x and y
    X, y = await createDataset(df_pre,look_back)
    X = X.reshape(X.shape[0],X.shape[1],1)
    #Predict compare
    predictData = await get_predict(X,df_pre,look_back,1)
    #Predict future
    predictDataFuture = []
    for i in range(look_back):
        if i == 0:
            x_pred = df_pre[-look_back:]
        else:
            x_pred = np.append(x_pred[0][:-1],[[y_pred]],axis=0)
        x_pred = np.array([x_pred.tolist()])
        y_pred = model.predict(x_pred,batch_size=1)[0][0]
        predictDataFuture.append(y_pred)
    predictDataFuture = scaler.inverse_transform(np.array(predictDataFuture).reshape(-1,1)).reshape(-1).tolist()
    # print(predictDataFuture)
    newData = await reformat(predictData,predictDataFuture)
    return newData

async def predictV2():
    newData = await predictV1()
    data = {}
    # Forcast: Date:
    data["date"] = newData["data"]["date"][-7:]
    data["cases"] = newData["data"]["forecast"][-7:]
    return data

# Train model
async def trainModel():
    model, scaler, df = await configModel()
    return 0