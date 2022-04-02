from re import L
import numpy as np 
import pandas as pd
from tensorflow.keras.models import load_model
import datetime
import os

def checkModel():
    file = os.listdir("assets/models/LSTM")
    for i in range(len(file)):
        if not (file[i].startswith("model")):
            model = file[i-1]
            break
    scaler = file[-1]
    return model,scaler

def train():
    model, scaler = checkModel()
    print(model)
    print(scaler)
    return 0