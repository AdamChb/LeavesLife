from flask import Flask, render_template, request, redirect, url_for
from predict_image import predict_image

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
            return redirect(url_for('detection', predicted_class=predicted_class, probability=probability))
    return render_template("index.html")

@app.route('/detection')
def detection():
    predicted_class = request.args.get('predicted_class')
    probability = request.args.get('probability')
    return render_template("detection.html", predicted_class=predicted_class, probability=probability)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
