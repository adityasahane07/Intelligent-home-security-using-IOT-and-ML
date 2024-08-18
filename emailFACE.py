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
"""
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
"""
#from Database.db import write_data
	

def mail(message):
	email = 'infinityokayy@gmail.com'
	password = "xrit vgiv lyth ocvk"
	send_to_email = 'prasadshetye123@gmail.com'
	subject = 'Face recognition'

	file_location = '/home/pi/Desktop/BE/face_recognition5.py'

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet


"""
def init_fireabse():
	cred = credentials.Certificate('/home/pi/Desktop/BE/service_acc_key.json')
	# Initialize the app with a None auth variable, limiting the server's access
	firebase_admin.initialize_app(cred, {
		'databaseURL': 'https://intelligent-home-securit-49b8d-default-rtdb.firebaseio.com/',
		'databaseAuthVariableOverride': None
	})
	
def write_unkonwn(msg = "0"):
	

	# The app only has access to public data as defined in the Security Rules
	ref = db.reference('/Data')
	is_new_ref = ref.child('IS_NEW')
	is_unkwon_ref = ref.child('IS_KNOWN')
	is_new_ref.set(msg)
	is_unkwon_ref.set('0')
"""
def main():
	
	# INITIALIZE
	facenet = FaceNet()
	faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
	Y = faces_embeddings['arr_1']
	print(Y)
	encoder = LabelEncoder()
	encoder.fit(Y)
	haarcascade = cv.CascadeClassifier("/home/pi/Desktop/BE/haarcascade_frontalface_default (1).xml")
	model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

	cap = cv.VideoCapture(0)

	# Set threshold for recognizing known faces and for considering unknown faces
	threshold_known = 0.5  # Adjust this threshold for known faces
	threshold_distance = 0.8  # Adjust this threshold for unknown faces

	# WHILE LOOP
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
					final_name = "Unknown"
					msg="Unknown person detected "
					#write_unkonwn()
					#write_data({"Visited" : 1, "name": "Sonu"})
					# mail(msg)
				else:
					idx = distances.index(min_distance)
					final_name = Y[idx]
					msg="known person detected " + final_name
					# mail(msg)
					# schedule.every(10).seconds.do(mail(msg))
					#write_unkonwn(msg)

			cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
			cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
					   1, (0, 0, 255), 3, cv.LINE_AA)

		cv.imshow("Face Recognition:", frame)
		if cv.waitKey(1) & ord('q') == 27:
			break
			
	cap.release()
	cv.destroyAllWindows()
	
if __name__ == "__main__":
	#init_fireabse()
	main()
