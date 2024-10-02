import cv2
import numpy as np
from keras.models import load_model

# load the trained model
model = load_model('mask_detection_model')

# open the camera
cap = cv2.VideoCapture(0)

while True:
    # take a frame
    ret, frame = cap.read()

    if not ret:
        print("camera not working")
        break

    # find faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        # take area of face
        face = frame[y:y + h, x:x + w]

        # resize the face
        face_resized = cv2.resize(face, (128, 128))
        face_scaled = face_resized / 255.0
        face_reshaped = np.reshape(face_scaled, [1, 128, 128, 3])

        # make prediction
        prediction = model.predict(face_reshaped)
        pred_label = np.argmax(prediction)

        # plot square around face red or green
        if pred_label == 0:
            # if the person is wearing a mask, draw a green square
            color = (0, 255, 0)
            label = "Mask"
        else:
            # if the person is not wearing a mask, draw a red square
            color = (0, 0, 255)
            label = "No Mask"

        # draw the square
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # show the frame in the window
    cv2.imshow('Mask Detection', frame)

    # if 'q' is pressed, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the camera
cap.release()
cv2.destroyAllWindows()
