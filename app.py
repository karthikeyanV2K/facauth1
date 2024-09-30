import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('facenet_keras.h5')
class_names = os.listdir('dataset/')

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Save the captured image
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    max_prob = np.max(predictions)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Confidence threshold
    confidence_threshold = 0.6
    predicted_label = class_names[predicted_class] if max_prob >= confidence_threshold else "Unknown"

    return jsonify({'predicted_class': predicted_label, 'confidence': float(max_prob)})

if __name__ == "__main__":
    app.run(debug=True)
