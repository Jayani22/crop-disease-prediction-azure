import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("model/crop_disease_model.h5")

# Class names (same order as training folders)
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

# Load image
img = tf.keras.preprocessing.image.load_img(
    "test.jpg",
    target_size=(128,128)
)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence:", round(confidence,2), "%")
