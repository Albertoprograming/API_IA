from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Inicializar la app y habilitar CORS
app = Flask(__name__)
CORS(app)

# Cargar el modelo
model = load_model("CNN_entrenada.h5")

# Etiquetas de las clases
class_labels = ['Clase1_Cables', 'Clase2_Case', 'Clase3_CPU', 'Clase4_GPU', 'Clase5_HDD', 
                'Clase6_Headset', 'Clase7_Keyboard', 'Clase8_Microphone', 'Clase9_Monitor', 'Clase10_MotherBoard', 
                'Clase11_Mouse', 'Clase12_RAM', 'Clase13_Speakers', 'Clase14_Webcam']

# Ruta para la clasificación
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen del cliente
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")
        
        # Preprocesar la imagen
        image = image.resize((224, 224))  # Redimensionar
        image_array = np.array(image) / 255.0  # Normalizar
        image_array = np.expand_dims(image_array, axis=0)  # Añadir batch
        
        # Hacer predicción
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        
        # Respuesta
        return jsonify({"class": predicted_label, "confidence": float(np.max(predictions))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar el servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
