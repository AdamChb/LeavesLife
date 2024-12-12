from flask import Flask, render_template, request, redirect, url_for
from predict_image import predict_image
from train_model import train_model
import mlflow
from threading import Thread
import os
import base64

app = Flask(__name__)    

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            image_bytes = file.read()
            predicted_class, probability = predict_image(image_bytes)
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            return redirect(url_for('detection', predicted_class=predicted_class, probability=probability, image_data=encoded_image))
    return render_template("index.html")

@app.route('/detection')
def detection():
    predicted_class = request.args.get('predicted_class')
    probability = request.args.get('probability')
    image_data = request.args.get('image_data')
    return render_template("detection.html", predicted_class=predicted_class, probability=probability, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)