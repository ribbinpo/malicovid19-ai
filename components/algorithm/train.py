from re import L
import numpy as np 
import pandas as pd
import datetime
import os
import requests
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
import tensorflow as tf

def checkModel():
    file = os.listdir("assets/models/LSTM")
    for i in range(len(file)):
        if not (file[i].startswith("model")):
            model = file[i-1]
            break
    scaler = file[-1]
    return model,scaler

async def readData():
    data = requests.get("https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-by-provinces")
    # df.to_csv("case")
    df = pd.read_json(data.text)
    # current_total_case = phuket["total_case"]
    # Start from 07/04/2021 - present
    phuketAll=df[df["province"] == "ภูเก็ต"].reset_index()
    phuket = phuketAll
    phuket=phuketAll[phuketAll["new_case"] != 0]
    phuketBefore = phuketAll.loc[phuket.index[0]-1]
    #Reset value before
    phuket["total_case"]=phuket["total_case"] - phuketBefore["total_case"]
    phuket["total_case_excludeabroad"] -= phuketBefore["total_case_excludeabroad"]
    phuket["total_death"] -= phuketBefore["total_death"]
    phuket.reset_index()
    # #14days recovery 
    return phuket

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i+look_back])
	return np.array(dataX), np.array(dataY)

def createModel(look_back):
    #Create Model
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(6, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(6, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

async def train():
    model, scaler = checkModel()
    modelId = int(model[5:model.index(".")])+1
    scalerId = int(scaler[6:scaler.index(".")])+1
    phuket =  await readData()
    # fix random seed for reproducibility
    np.random.seed(7)
    dataset = phuket[["new_case"]].values
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1,1))
    # reshape into X=t and Y=t+1
    look_back = 7
    trainX, trainY = create_dataset(dataset, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    model = createModel(look_back)
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX, batch_size=1)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY_ = scaler.inverse_transform(trainY)

    return 0