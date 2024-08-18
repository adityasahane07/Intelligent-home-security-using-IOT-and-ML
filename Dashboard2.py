import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#from emailFACE import main
from keras_facenet import FaceNet
from Database.db import write_data, get_csv
import cv2 as cv
import pickle
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db





#def init_fireabse():
#       if not firebase_admin._apps:
#            cred = credentials.Certificate('/home/pi/Downloads/Pass Fail Predict/service_acc_key.json') 
#            a=firebase_admin.initialize_app(cred, {'databaseURL': 'https://smart-home-security-662b8-default-rtdb.firebaseio.com/S'})
      

"""
    cred = credentials.Certificate('/home/pi/Desktop/BE/service_acc_key.json')
    cred = credentials.Certificate('/home/pi/Downloads/Pass Fail Predict/service_acc_key.json')
    st.write(cred)
    print(cred)
	Initialize the app with a None auth variable, limiting the server's access
    firebase_admin.initialize_app(cred,{'databaseURL': 'https://intelligent-home-securit-49b8d-default-rtdb.firebaseio.com/','databaseAuthVariableOverride': None })
 
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://intelligent-home-securit-49b8d-default-rtdb.firebaseio.com/'})
"""

final_name = ""
def write_unkonwn(msg = "0"):
	# The app only has access to public data as defined in the Security Rules
	ref =db.reference('/Data')
	is_new_ref = ref.child('IS_NEW')
	is_unkwon_ref = ref.child('IS_KNOWN')
	is_new_ref.set(msg)
	is_unkwon_ref.set('0')

def camera():
	global final_name
	# INITIALIZE
	facenet = FaceNet()
	faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
	Y = faces_embeddings['arr_1']
	print(Y)
	encoder = LabelEncoder()
	encoder.fit(Y)
	haarcascade = cv.CascadeClassifier("/home/pi/Desktop/BE/haarcascade_frontalface_default.xml")
    #haarcascade = cv.CascadeClassifier("/home/pi/Downloads/Pass Fail Predict/haarcascade_frontalface_default.xml")
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
                 #  wrte_unkonwn(msg)
					write_data({"Visited" : 0, "name": msg})
					st.write(f"Visitor : {final_name}") 
					# mail(msg)
				else:
					idx = distances.index(min_distance)
					final_name = Y[idx]
					st.write(f"Visitor : {final_name}")
					write_data({"Visited" : 1, "name": final_name})
					
                    
                 # 	msg="known person detected "+final_name
           #        st.write(final_name)
					# mail(msg)
					# schedule.every(10).seconds.do(mail(msg))
      #             write_unkonwn(msg)
            

			cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
			cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
					   1, (0, 0, 255), 3, cv.LINE_AA)

		cv.imshow("Face Recognition:", frame)
		if cv.waitKey(1) & ord('q') == 27:
			break
			
	cap.release()
	cv.destroyAllWindows()
	

def home_page():
    #init_fireabse()
    st.button("Open camera",on_click=camera)

