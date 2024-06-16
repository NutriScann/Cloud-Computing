from flask import Blueprint, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import base64
import io
import os

main = Blueprint('main', __name__)

# Load the .h5 model
# Daftarkan KerasLayer sebagai objek khusus
with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))

# Labels and descriptions
foods = [
    {"name": "bakso", "display_name": "Bakso", "desc": "Bola daging yang berasal dari Indonesia, biasanya terbuat dari campuran daging sapi atau ayam dan tepung tapioka.", "carbo": "30", "protein": "10", "calory": "250", "lemak": "15", "bahan": ["Daging Sapi", "Tepung", "Bawang Putih"]},
    {"name": "bibimbap", "display_name": "Bibimbap", "desc": "Hidangan khas Korea yang terdiri dari nasi dengan berbagai macam sayuran, daging, telur, dan saus gochujang.", "carbo": "85", "protein": "14", "calory": "500", "lemak": "12", "bahan": ["Nasi", "Sayuran", "Gochujang","Daging Sapi","Telur"]},
    {"name": "burger", "display_name": "Burger", "desc": "Makanan cepat saji yang berasal dari Amerika Serikat, terdiri dari daging patty yang disajikan di dalam roti bulat dengan tambahan sayuran dan saus.", "carbo": "40", "protein": "25", "calory": "500", "lemak": "25", "bahan": ["Patty", "Roti", "Keju", "Selada", "Tomat", "Acar", "Saus"]},
    {"name": "donat", "display_name": "Donat", "desc": "Kue goreng berbentuk cincin atau bola yang berasal dari Amerika Serikat, biasanya ditaburi gula atau dilapisi glasir.", "carbo": "40", "protein": "5", "calory": "300", "lemak": "15", "bahan": ["Tepung", "Ragi", "Gula", "Susu", "Telur", "Mentega", "Topping"]},
    {"name": "eskrim", "display_name": "Es Krim", "desc": "Hidangan penutup beku yang terbuat dari susu, krim, dan gula, berasal dari Persia kuno namun populer di seluruh dunia.", "carbo": "30", "protein": "4", "calory": "250", "lemak": "15", "bahan": ["Susu", "Krim", "Gula","Perisa Rasa"]},
    {"name": "gado", "display_name": "Gado-Gado", "desc": "Salad sayuran rebus khas Indonesia yang disajikan dengan bumbu kacang.", "carbo": "35", "protein": "15", "calory": "300", "lemak": "10", "bahan": ["Sayur-sayuran", "Tahu", "Tempe", "Saus Kacang"]},
    {"name": "kentang goreng", "display_name": "Kentang Goreng", "desc": "Irisan kentang yang digoreng hingga renyah, berasal dari Belgia.", "carbo": "45", "protein": "4", "calory": "350", "lemak": "20", "bahan": ["Kentang", "Garam"]},
    {"name": "nasi_goreng", "display_name": "Nasi Goreng", "desc": "Hidangan nasi yang digoreng dengan bumbu dan tambahan seperti telur, daging, dan sayuran, berasal dari Indonesia.", "carbo": "50", "protein": "20", "calory": "500", "lemak": "20", "bahan": ["Nasi", "Kecap", "Telur", "Ayam"]},
    {"name": "pangsit", "display_name": "Pangsit", "desc": "Kulit adonan tipis yang diisi dengan daging atau sayuran, lalu direbus atau digoreng, berasal dari Tiongkok.", "carbo": "6", "protein": "3", "calory": "60", "lemak": "3", "bahan": ["Kulit Pangsit", "Daging Cincang", "Bawang Putih", "Garam", "Daun Bawang"]},
    {"name": "pizza", "display_name": "Pizza", "desc": "Hidangan Italia yang terdiri dari roti pipih yang dipanggang dan diberi topping seperti saus tomat, keju, dan daging.", "carbo": "35", "protein": "12", "calory": "300", "lemak": "15", "bahan": ["Adonan Pizza", "Saus Tomat", "Keju", "Topping"]},
    {"name": "ramen", "display_name": "Ramen", "desc": "Mie kuah Jepang yang disajikan dengan berbagai tambahan seperti daging, telur, dan sayuran.", "carbo": "70", "protein": "15", "calory": "500", "lemak": "20", "bahan": ["Mie Ramen", "Kaldu", "Daging", "Telur Rebus", "Sayuran (Daun Bawang, Nori, Tauge)", "Bumbu"]},
    {"name": "rawon", "display_name": "Rawon", "desc": "Sup daging sapi khas Indonesia yang berwarna hitam karena menggunakan kluwek sebagai bumbu utama.", "carbo": "25", "protein": "20", "calory": "300", "lemak": "15", "bahan": ["Daging Sapi", "Kluwek", "Bawang Merah"]},
    {"name": "rendang", "display_name": "Rendang", "desc": "Hidangan daging sapi yang dimasak dengan santan dan rempah-rempah hingga kering, berasal dari Minangkabau, Indonesia.", "carbo": "15", "protein": "25", "calory": "400", "lemak": "30", "bahan": ["Daging Sapi", "Santan", "Rempah-rempah"]},
    {"name": "sate", "display_name": "Sate", "desc": "Potongan daging yang ditusuk dan dipanggang, biasanya disajikan dengan saus kacang, berasal dari Indonesia.", "carbo": "20", "protein": "25", "calory": "350", "lemak": "20", "bahan": ["daging Ayam", "Bumbu Kacang"]},
    {"name": "soto", "display_name": "Soto", "desc": "Sup tradisional Indonesia yang biasanya terbuat dari kaldu daging dengan tambahan sayuran dan bihun atau lontong.", "carbo": "25", "protein": "20", "calory": "300", "lemak": "15", "bahan": ["Ayam", "Sayuran", "Bumbu Kuning"]}
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
        predicted_label_index = np.argmax(predictions)
        predicted_label = labels[predicted_label_index]
        confidence_score = predictions[0][predicted_label_index] * 100
        
        description = descriptions[predicted_label]
        description['confidence'] = f"{confidence_score:.2f}%"
        description['name'] = description.pop('display_name')
        
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
        # Copy data untuk merubah name menjadi display_name di response
        foods_display = [{**food, 'name': food['display_name']} for food in foods]
        response = {
            "status": 200,
            "message": "success",
            "data": foods_display
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": 500, "message": str(e), "data": None}), 500
