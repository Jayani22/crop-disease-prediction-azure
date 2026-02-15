import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(128,128),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/valid",
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize images
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y:(normalization_layer(x),y))
val_ds = val_ds.map(lambda x,y:(normalization_layer(x),y))

# CNN Model
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(len(class_names),activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save model
model.save("model/crop_disease_model.h5")

# Save accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train','val'])
plt.title("Accuracy")
plt.savefig("model/accuracy_graph.png")
plt.show()
