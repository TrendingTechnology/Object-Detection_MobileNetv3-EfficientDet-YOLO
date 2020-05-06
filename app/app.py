"""
Flask endpoint for prediction
"""

from pathlib import Path
import sys
import os
path = Path(sys.path[0])
sys.path.append(str(path.parent))

from ml_models_api import Yolov3, MobileNetv3, EfficientDet
from flask import Flask, request, jsonify
import numpy as np
import requests
import cv2

def url_to_image(url, colors='BGR'):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    # BGR by default
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # To convert to RGB - In case of EfficientDet model
    if colors=='RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return image

# initialize flask application
app = Flask(__name__)

# Load pre-trained models
models_conf = {
    'yolo': {'config': "../artifacts/yolo_files/yolov3.cfg",
             'weights': "../artifacts/yolo_files/yolov3.weights"
             },
    'mobilenet': {'graph': '../artifacts/mobilenet_v3_files/graph.pbtxt',
                  'weights': '../artifacts/mobilenet_v3_files/frozen_inference_graph.pb'
                  },
    'efficientdet': {'frozen_model': "../artifacts/efficientdet_files_frozen"}
}

# Load model
models_types = {
    'yolo': Yolov3(**models_conf['yolo']),
    'mobilenet': MobileNetv3(**models_conf['mobilenet']),
    'efficientdet': EfficientDet(**models_conf['efficientdet'])
}

# Define API
HOST = "0.0.0.0"
PORT = os.environ.get("PORT", 8080)

@app.route('/predict/v1', methods=['GET'])
def predict():
    # Request should have
    # - 'model' : in ['yolo', 'mobilenet', 'efficientdet']
    # - 'image_url' : direct url to jpg or png images like https://imadelhanafi.com/data/draft/random/img1.jpg
    if 'model' in request.args:
        model = str(request.args['model'])
    else:
        return "Error: No model field provided. Please specify a model : yolo, mobilenet or efficientdet."

    if 'image_url' in request.args:
        image_url = str(request.args['image_url'])
    else:
        return "Error: No image_url field provided. Please specify a direct url i.e https://imadelhanafi.com/data/draft/random/img1.jpg"

    # Load the provided image from url
    if model == 'efficientdet':
        colors='RGB'
    else:
        colors= 'BGR'
    input_img = url_to_image(image_url, colors)

    # Run inference using the specified model and return results
    results = models_types[model].predict(input_img)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)