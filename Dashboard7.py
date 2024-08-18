import streamlit as st
import numpy as np
import cv2 as cv
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from Database.db import write_data
from t2 import rotateMax, rotateMin
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# face recognition part IIapt install
# IMPORT
import cv2 as cv
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path
import time


def mail(message):
	email = 'infinityokayy@gmail.com'
	password = "xrit vgiv lyth ocvk"
	send_to_email = "prasadshetye123@gmail.com, adityamsahane07@gmail.com, 9850mayurshinde@gmail.com"
	subject = 'Face recognition'

	file_location = '/home/pi/Desktop/prathamesh/visitor.jpg'

	msg = MIMEMultipart()
	msg['From'] = email
	msg['To'] = send_to_email
	msg['Subject'] = subject

	msg.attach(MIMEText(message, 'plain'))

	# Setup the attachment
	
	filename = os.path.basename(file_location)
	attachment = open(file_location, "rb")
	part = MIMEBase('application', 'octet-stream')
	part.set_payload(attachment.read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

	# Attach the attachment to the MIMEMultipart object
	msg.attach(part)

	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(email, password)
	text = msg.as_string()
	server.sendmail(email, send_to_email, text)
	#server.quit()


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
    prevPersonDetected = ""
    currentPersonDetected = ""
    # WHILE LOOP
    idx = 1
    while cap.isOpened():
        _, frame = cap.read()
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
            if min_distance < threshold_known:
                idx = distances.index(min_distance)
            else:
                # Check if the minimum distance is greater than the threshold for unknown faces
                if min_distance > threshold_distance:
                    currentPersonDetected = "Unknown"
                    if prevPersonDetected != currentPersonDetected or prevPersonDetected == "Unknown" or currentPersonDetected == "Unknown" :
                        final_name = "Unknown"
                        write_data({"Visited": 0, "name": "Unknown person detected"})
                        st.write(f"Visitor : {final_name}")
                        prevPersonDetected = currentPersonDetected
                        msg="Unknown person detected "
                        filename='visitor.jpg'
                        cv.imwrite(filename, frame)
                        mail(msg)
                         
                            #if st.button("Allow", key=f"allow_{idx}"):
                            #rotateMax()
                            #cap.release()
                            #cv.destroyAllWindows()s
                            #break
                            #if st.button("Disallow", key = f"disallow_{idx}"):
                            #rotateMin()
                            #cap.release()
                            #cv.destroyAllWindows()
                            #break
                            
                        #idx += 1    
                        
                    
                else:
                    idx = distances.index(min_distance)
                    final_name = Y[idx]
                    currentPersonDetected = final_name
                    if prevPersonDetected != currentPersonDetected or prevPersonDetected == "Unknown" or currentPersonDetected == "Unknown" :
                        st.write(f"Visitor : {final_name}")
                        write_data({"Visited": 1, "name": final_name})
                        prevPersonDetected = currentPersonDetected
                        rotateMax()
                        rotateMin()
                        idx += 1
                        msg="known person detected " + final_name
                        filename='visitor.jpg'
                        cv.imwrite(filename, frame)
                        mail(msg)
					# schedule.every(10).seconds.do(mail(msg))
					#write_unkonwn(msg)
            
            
            idx += 1
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)
      
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow("Face Recognition:", frame)
        
    
    cap.release()
    cv.destroyAllWindows()

def home_page():
    st.title("Face Recognition System")
    st.write("Click the button below to open the camera:")
    
    if st.button("Open Camera"):
        camera()

if __name__ == "__main__":
    Shome_page()
