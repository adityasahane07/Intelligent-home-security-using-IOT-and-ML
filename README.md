# Intelligent-home-security-using-IOT-and-ML and Web Application

Overview :

This project is an Intelligent Home Security System that leverages IoT devices, machine learning algorithms, and a web application to provide real-time monitoring, smart access control, and enhanced security for residential properties. The system aims to detect potential threats, alert the homeowner, and automatically manage access control through the integration of cameras, sensors, and smart locks. The solution provides 24/7 monitoring and ensures proactive security measures.

Features :

24/7 Live Surveillance: Continuous monitoring of the home environment using IoT devices (cameras, motion sensors, etc.).
Automated Access Control: Smart locks that open automatically when a known person is detected using facial recognition.
Real-Time Alerts: Send alerts to the homeowner through email and web notifications when an unknown person or suspicious activity is detected.
Predictive Threat Detection: Machine learning algorithms analyze behavior patterns and predict potential threats.
Remote Monitoring: Homeowners can monitor their homes remotely via the web application.
Emergency Response: Integration with emergency services for faster responses in case of security breaches.

Hardware Components : 

Cameras: Used for continuous surveillance and facial recognition.
Motion Sensors: Detect movement within the property and trigger alerts.
Door/Window Sensors: Monitor the status of doors and windows for unauthorized entry.
Smart Locks: Automatically controlled based on the identification of known individuals.
Microcontroller (Raspberry Pi/Arduino): Central processing unit that connects sensors and devices for real-time data collection and action.
Wi-Fi Module: Enables communication between IoT devices and the web application.

Software Components :

Web Application: Provides a user interface for monitoring and controlling the system remotely.
Machine Learning Algorithms: Analyze sensor data to detect threats, identify individuals, and automate security protocols.
IoT Platform: Enables connectivity and communication between hardware devices and the web application.

Installation Prerequisites :

Hardware Setup: Ensure all sensors, cameras, and smart locks are installed and connected to the microcontroller.
Python 3.x
Node.js
Machine Learning Libraries: TensorFlow, Keras, OpenCV
Flask/Django: For backend server handling.

Steps : 

1. Install Raspberry Pi Imager from https://www.raspberrypi.com/software/ on any Raspberry Pi using an SD card reader.
2. Install all the required Libraries using the command : pip install -r requirements.txt on the raspberry pi.
3. Open the path on terminal where the files are located.
4. create a virtual environment after opening the folder in the terminal.
   Type the below commands :
   cd venv
   cd bin
   source activate
   cd ..
   cd ..
   sudo pigpiod
5. Now run the main script using the command : streamlit run Login_SignUp.py

Configure Hardware:

Set up sensors, cameras, and smart locks, and connect them to the microcontroller.
Ensure communication between the microcontroller and the web application through the Wi-Fi module.

Usage : 
Live Feed Monitoring: View real-time camera feeds through the web application.
Automated Alerts: Receive alerts when suspicious activity is detected.
Access Control: Remotely control smart locks based on facial recognition.

Project Structure
bash
Copy code
├── hardware/
│   ├── camera/
│   ├── servo motor/
│   └── wires/
├── machine_learning/
│   ├── facial_recognition/
│   └── anomaly_detection/
├── web_application/
│   ├── templates/
│   ├── static/
│   └── app.py
└── README.md


