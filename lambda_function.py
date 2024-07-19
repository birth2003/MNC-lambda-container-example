import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import decode_predictions
import numpy as np
import time

def lambda_handler(event, context):
    model = MobileNet(weights='imagenet')
    img_array = np.load("/var/task/lambda-container-example/preprocessed_image.npy")
    preds = model.predict(img_array)

    return {
        'Predicted:', decode_predictions(preds, top=1)[0]
    }
    
