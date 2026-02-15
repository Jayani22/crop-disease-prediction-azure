from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("../model/crop_disease_model.h5")

# Class names (must match training folders order)
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

def predict_image(img):
    img = img.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return class_names[class_index]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file)

        result = predict_image(img)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
