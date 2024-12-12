from flask import Flask, render_template, request, send_from_directory, jsonify
from predict_image import predict_image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
classes = {
    0: {
        "plant": "Apple",
        "disease": "Apple Scab",
    },
    1: {
        "plant": "Apple",
        "disease": "Black Rot",
    },
    2: {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
    },
    3: {
        "plant": "Apple",
        "disease": "Healthy",
    },
    4: {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot",
    },
    5: {
        "plant": "Corn",
        "disease": "Common Rust",
    },
    6: {
        "plant": "Corn",
        "disease": "Healthy",
    },
    7: {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
    },
    8: {
        "plant": "Grape",
        "disease": "Black Rot",
    },
    9: {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
    },
    10: {
        "plant": "Grape",
        "disease": "Healthy",
    },
    11: {
        "plant": "Grape",
        "disease": "Leaf Blight",
    },
    12: {
        "plant": "Potato",
        "disease": "Early Blight",
    },
    13: {
        "plant": "Potato",
        "disease": "Healthy",
    },
    14: {
        "plant": "Potato",
        "disease": "Late Blight",
    },
    15: {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
    },
    16: {
        "plant": "Tomato",
        "disease": "Early Blight",
    },
    17: {
        "plant": "Tomato",
        "disease": "Healthy",
    },
    18: {
        "plant": "Tomato",
        "disease": "Late Blight",
    },
    19: {
        "plant": "Tomato",
        "disease": "Leaf Mold",
    },
    20: {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
    },
    21: {
        "plant": "Tomato",
        "disease": "Spider Mites",
    },
    22: {
        "plant": "Tomato",
        "disease": "Target Spot",
    },
    23: {
        "plant": "Tomato",
        "disease": "Mosaic Virus",
    },
    24: {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
    },
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)  # Sauvegarder le fichier avant de le lire
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            predicted_class, probability = predict_image(image_bytes)
            predicted_class = str(predicted_class)
            probability = round(float(probability) * 100, 1)
            return jsonify({
                "predicted_plant": classes[int(predicted_class)]["plant"],
                "predicted_disease": classes[int(predicted_class)]["disease"],
                "probability": probability,
                "image_path": file.filename
            })
    return render_template("index.html")

@app.route('/detection/', methods=['GET'])
def detection():
    predicted_plant = request.args.get('predicted_plant', "Unknown")
    predicted_disease = request.args.get('predicted_disease', "Unknown")
    probability = request.args.get('probability', "0.0")
    image_path = request.args.get('image_path', None)
    print(probability)
    return render_template("detection.html", 
                            predicted_plant=predicted_plant, 
                            predicted_disease=predicted_disease,
                            probability=probability, 
                            image_path=image_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)