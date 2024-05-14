import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)

model = load_model('BrainTumorDetectionModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')





def get_className(classNo):
    if classNo == 0:
        return "Tumor has not been detected"
    elif classNo == 1:
        return "Brain has Tumor"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predict_x = model.predict(input_img)
    result = np.argmax(predict_x, axis=1)
    confidence = np.max(predict_x) * 100  # Get the confidence level
    return result[0], confidence


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value, confidence = getResult(file_path)
        result = get_className(value)
        
        precautions = ""
        treatments = ""
        if value == 1:
            precautions = [
    "Regular medical check-ups are crucial.",
    "Maintain a healthy diet and lifestyle.",
    "Avoid exposure to radiation and toxic chemicals.",
    "Stay informed about the symptoms and seek medical attention if you experience any unusual symptoms.",
    "Follow your doctor's advice and prescribed treatments.",
    "Exercise regularly to maintain overall health.",
    "Stay hydrated and avoid smoking.",
    "Manage stress through relaxation techniques like meditation."
]
            treatments = [
    "Consume a diet rich in fruits and vegetables.",
    "Use turmeric in your meals for its anti-inflammatory properties.",
    "Drink green tea daily for its antioxidants.",
    "Include garlic in your diet to boost your immune system.",
    "Practice yoga and breathing exercises to reduce stress.",
    "Apply essential oils like lavender for relaxation.",
    "Stay hydrated by drinking plenty of water.",
    "Consider supplements like Vitamin D after consulting with your doctor."
]

        
        return jsonify({
            "result": result,
            "confidence": confidence,
            "precautions": precautions,
            "home_treatment": treatments
        })
    return None




if __name__ == '__main__':
    app.run(debug=True)
