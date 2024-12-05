import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import decode_predictions
import numpy as np
import base64
import io

def lambda_handler(event, context):
    # 모델 로드
    model = MobileNet(weights='imagenet')
    
    # 입력 데이터 (Base64로 인코딩된 데이터) 디코딩
    image_base64 = event['body']  # API Gateway로부터 받은 Base64 데이터
    image_bytes = base64.b64decode(image_base64)
    
    # NumPy 배열 로드 (예: preprocessed_image.npy 형식)
    img_array = np.load(io.BytesIO(image_bytes))
    
    # 모델 추론
    preds = model.predict(img_array)
    prediction = decode_predictions(preds, top=1)[0]

    # 결과 반환
    prediction_result = {
        'class': prediction[0][1],
        'confidence': float(prediction[0][2])
    }
    
    return {
        'statusCode': 200,
        'body': prediction_result
    }
