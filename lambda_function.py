import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import decode_predictions
import numpy as np



def lambda_handler(event, context):
    model = MobileNet(weights='imagenet')
    img_array = np.load("/var/task/MNC-lambda-container-example/preprocessed_image.npy")
    preds = model.predict(img_array)
    prediction = decode_predictions(preds, top=1)[0]

    prediction_result = {
        'class': prediction[0][1],
        'confidence': float(prediction[0][2])
    }
    
    return {
        'Predicted': prediction_result
    }
