from flask import Blueprint, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import os

main = Blueprint('main', __name__)

# Load the .h5 model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))

# Labels and descriptions
foods = [
    {"name": "bakso", "desc": "Deskripsi Bakso", "carbo": "30g", "protein": "10g", "calory": "250 kkal", "lemak": "15g", "bahan": ["daging sapi", "tepung", "bawang putih"]},
    {"name": "bebek_betutu", "desc": "Deskripsi Bebek Betutu", "carbo": "20g", "protein": "30g", "calory": "400 kkal", "lemak": "25g", "bahan": ["bebek", "bumbu kuning", "daun pisang"]},
    {"name": "gado_gado", "desc": "Deskripsi Gado-Gado", "carbo": "35g", "protein": "15g", "calory": "300 kkal", "lemak": "10g", "bahan": ["sayur-sayuran", "tahu", "tempe", "saus kacang"]},
    {"name": "nasi_goreng", "desc": "Deskripsi Nasi Goreng", "carbo": "50g", "protein": "20g", "calory": "500 kkal", "lemak": "20g", "bahan": ["nasi", "kecap", "telur", "ayam"]},
    {"name": "pempek", "desc": "Deskripsi Pempek", "carbo": "40g", "protein": "15g", "calory": "350 kkal", "lemak": "15g", "bahan": ["ikan", "tepung", "cuka"]},
    {"name": "rawon", "desc": "Deskripsi Rawon", "carbo": "25g", "protein": "20g", "calory": "300 kkal", "lemak": "15g", "bahan": ["daging sapi", "kluwek", "bawang merah"]},
    {"name": "rendang", "desc": "Deskripsi Rendang", "carbo": "15g", "protein": "25g", "calory": "400 kkal", "lemak": "30g", "bahan": ["daging sapi", "santan", "rempah-rempah"]},
    {"name": "sate", "desc": "Deskripsi Sate", "carbo": "20g", "protein": "25g", "calory": "350 kkal", "lemak": "20g", "bahan": ["daging ayam", "bumbu kacang"]},
    {"name": "soto", "desc": "Deskripsi Soto", "carbo": "25g", "protein": "20g", "calory": "300 kkal", "lemak": "15g", "bahan": ["ayam", "sayuran", "bumbu kuning"]}
]

labels = [food['name'] for food in foods]
descriptions = {food['name']: food for food in foods}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

@main.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"status": 400, "message": "No image data provided", "data": None}), 400

    try:
        image_data = data['image']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        input_data = preprocess_image(image)
        
        predictions = model.predict(input_data)
        predicted_label = labels[np.argmax(predictions)]
        description = descriptions[predicted_label]
        
        response = {
            "status": 200,
            "message": "success",
            "data": [description]
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": 500, "message": str(e), "data": None}), 500

@main.route('/list', methods=['GET'])
def list_foods():
    try:
        response = {
            "status": 200,
            "message": "success",
            "data": foods
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": 500, "message": str(e), "data": None}), 500
