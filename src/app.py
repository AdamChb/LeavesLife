from flask import Flask, render_template, request
from predict_image import predict_image
from train_model import train_model
import mlflow
from threading import Thread

app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True

def train_models_in_background():
    for i in range(0, 10):
        print(f"\n\nTraining model with seed {i}...\n\n")
        train_model(seed=i)
        print(f"\n\nModel trained with seed {i}.\n\n")

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
            return render_template("index.html", predicted_class=predicted_class, probability=probability)
    return render_template("index.html")

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5001")  # For local tracking server
    # mlflow.set_tracking_uri("http://mlflow:5000")  # For Docker tracking server
    print("Training models before running the app...")
    train_models_in_background()
    print("Models trained before running the app.")
    app.run(debug=True, host='0.0.0.0', port=5000)
