import json
import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.metrics import mean_squared_error

global drop_avg
global drop_var

def lambda_handler(event, context):
    df=pd.read_csv("/var/task/MNC-lambda-container-example/tun1.csv")
    df.head(5)

    training_set = df.iloc[:1400].values
    test_set = df.iloc[1400:].values

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    model = load_model("/var/task/MNC-lambda-container-example/model1.h5")

    dataset_train = df.iloc[:1400]
    dataset_test = df.iloc[1400:]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_test.values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)


    X_test = []

    for i in range(0,200):

       X_test.append(inputs[i:i+200, 0])


    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_drop = model.predict(X_test)
    predicted_drop = sc.inverse_transform(predicted_drop)

    drop_avg = np.mean(predicted_drop)
    drop_var = np.var(predicted_drop)

    real_drop = df.iloc[1600:1800].values

    RMSE = mean_squared_error(real_drop, predicted_drop)**0.5

    avg = str(drop_avg)
    var = str(drop_var) 
    packetdrop = "(Average PacketDrop: " + avg + ", PacketDropVariance: " + var +")"

    return {
        'packetdrop': packetdrop
    }
