import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('waste_sorting_model.h5')
categories = ['plastic', 'paper', 'metal', 'organic']  # Update based on your dataset

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = categories[np.argmax(prediction)]

    cv2.putText(frame, f"Type: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Waste Sorting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
