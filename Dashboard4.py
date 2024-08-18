import streamlit as st
import numpy as np
import cv2 as cv
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from Database.db import write_data  # Assuming this is correctly imported
from t2 import rotateMax, rotateMin  # Assuming these are correctly implemented

def camera():
    # INITIALIZE
    facenet = FaceNet()
    faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

    cap = cv.VideoCapture(0)

    # Set threshold for recognizing known faces and for considering unknown faces
    threshold_known = 0.5  # Adjust this threshold for known faces
    threshold_distance = 0.8  # Adjust this threshold for unknown faces

    # WHILE LOOP
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

        for x, y, w, h in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv.resize(img, (160, 160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)

            # Calculate distances between embeddings
            distances = []
            for emb in faces_embeddings['arr_0']:
                distances.append(np.linalg.norm(ypred - emb))
            min_distance = min(distances)
            final_name = None  # Initialize final_name
            if min_distance < threshold_known:
                idx = distances.index(min_distance)
            else:
                # Check if the minimum distance is greater than the threshold for unknown faces
                if min_distance > threshold_distance:
                    final_name = "Unknown"
                    write_data({"Visited": 0, "name": "Unknown person detected"})
                    st.write(f"Visitor : {final_name}")
                else:
                    idx = distances.index(min_distance)
                    final_name = Y[idx]
                    st.write(f"Visitor : {final_name}")
                    write_data({"Visited": 1, "name": final_name})

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Face Recognition:", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def home_page():
    st.title("Face Recognition System")
    st.write("Click the button below to open the camera:")
    if st.button("Allow"):
        rotateMax()
    if st.button("Disallow"):
        rotateMin()
    if st.button("Open Camera"):
        camera()

if __name__ == "__main__":
    home_page()
