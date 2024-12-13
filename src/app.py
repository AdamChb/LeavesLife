#-----------------------------------#
# Machine Learning Project
# LeavesLife: Plant Disease Detection
# Dates: 2024-11-27 - 2024-12-12
#
# Authors:
# - Mathias BENOIT
# - Adam CHABA
# - Eva MAROT
# - Sacha PORTAL
#
# File Description: 
# Flask application to serve the model
#-----------------------------------#

# Import the necessary libraries
import os
from flask import Flask, render_template, request, send_from_directory, jsonify

# Import the function to predict the class of an image
from predict_image import predict_image


# Create the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create the upload folder if it does not exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Classes of the model
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


# Define the route '/'
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    This function is the home page of the application. 
    It allows the user to upload an image and 
    get the prediction of the model.
    """
    
    # If the request method is POST, predict the class of the image
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        
        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == "":
            return "No selected file"
        
        if file:
            # Save the file to the server
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Read the image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                
            # Predict the class of the image
            predicted_class, probability = predict_image(image_bytes)
            
            # Format the prediction
            predicted_class = str(predicted_class)
            probability = round(float(probability) * 100, 1)
            
            # Return the prediction
            return jsonify({
                "predicted_plant": classes[int(predicted_class)]["plant"],
                "predicted_disease": classes[int(predicted_class)]["disease"],
                "probability": probability,
                "image_path": file.filename
            })
            
    # If the request method is GET, return the home page
    return render_template("index.html")


# Define the route '/detection/'
@app.route('/detection/', methods=['GET'])
def detection():
    """
    This function is the detection page of the application.
    It displays the prediction of the model.
    """
    
    # Get the parameters from the URL
    predicted_plant = request.args.get('predicted_plant', "Unknown")
    predicted_disease = request.args.get('predicted_disease', "Unknown")
    probability = request.args.get('probability', "0.0")
    image_path = request.args.get('image_path', None)

    # Return the detection page
    return render_template("detection.html", 
                            predicted_plant=predicted_plant, 
                            predicted_disease=predicted_disease,
                            probability=probability, 
                            image_path=image_path)
    

# Define the route '/uploads/<filename>'
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    This function serves the uploaded files.
    
    Arguments:
        - filename: str, name of the file to serve
    """
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Run the application
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
