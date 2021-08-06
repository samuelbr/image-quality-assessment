import sys
import requests
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image, ImageOps
sys.path.append('src')


from utils.utils import calc_mean_score
from evaluater.predict import main
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator

base_model_name = 'MobileNet'

weights_file_aesthetic = 'models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5'
weights_file_technical = 'models/MobileNet/weights_mobilenet_technical_0.11.hdf5'

def build_model(base_model_name, weights_file):
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)
    return nima

model_aesthetic = build_model(base_model_name, weights_file_aesthetic)
model_technical = build_model(base_model_name, weights_file_technical)

app = Flask(__name__)

def resize_image(img):
    target_size = 224
    img.thumbnail((target_size, target_size))

    delta_w = target_size - img.width
    delta_h = target_size - img.height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(img, padding)

@app.route('/', methods=['POST'])
def predict():
    req = request.json
    result = []
    for image in req['images']:
        stream = requests.get(image, stream=True).raw
        img = Image.open(stream)

        preprocess_fn = model_aesthetic.preprocessing_function()
        img = resize_image(img)
        img = preprocess_fn(np.array(img))

        img = img[np.newaxis,:,:,:]


        aesthetic_pred = model_aesthetic.nima_model.predict(img)
        technical_pred = model_technical.nima_model.predict(img)
        
        aesthetic_pred = calc_mean_score(aesthetic_pred[0])
        technical_pred = calc_mean_score(technical_pred[0])

        result.append({
            'image': image,
            'aesthetic': aesthetic_pred,
            'technical': technical_pred
        })

    return jsonify(result)

    

if __name__ == "__main__":
    app.run()