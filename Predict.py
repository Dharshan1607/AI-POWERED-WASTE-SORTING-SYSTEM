predict.py
python
Copy
Edit
import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model = tf.keras.models.load_model('model/waste_model.h5')

# Class names
class_names = ['plastic', 'metal', 'paper', 'organic']  # Make sure this matches your folders

# Predict function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    return class_names[class_idx]

# Example
print(predict_image('example.jpg'))
