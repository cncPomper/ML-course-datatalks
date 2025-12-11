import onnxruntime as rt
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import os

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

onnx_model = os.getenv("ONNX_MODEL_PATH", "hair_classifier_empty.onnx")

session = rt.InferenceSession(
    onnx_model,
    providers=["CPUExecutionProvider"]
)

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

target_size = (200, 200)

def lambda_handler(event, context):
    url = event['url']

    img = download_image(url)
    prepared_img = prepare_image(img, target_size)
    
    x = np.array(prepared_img, dtype='float32')

    x = x / 127.5 - 1

    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)

    session_run = session.run([output_name], {input_name: x})
    float_prediction = session_run[0][0].tolist()
    result = {
        'predictions': float_prediction
    }
    return result