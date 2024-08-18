import streamlit as st
import firebase_admin
from firebase_admin import credentials
import json
import requests
from config.config import config

from Dashboard7 import home_page
#from firebase_init import init

 # Check if Firebase Admin SDK has been initialized
#if not firebase_admin._apps:
     # If not initialized, initialize Firebase Admin SDK
#    cred = credentials.Certificate("Credentials/serviceAccount.json")
#    firebase_admin.initialize_app(cred)
#init()


def init_fireabse():
       if not firebase_admin._apps:
            cred = credentials.Certificate('/home/pi/Downloads/Pass Fail Predict/service_acc_key.json') 
            a=firebase_admin.initialize_app(cred, {'databaseURL': 'https://smart-home-security-662b8-default-rtdb.firebaseio.com/S'})

def app():
# Usernm = []
    st.title('Welcome to :White[Home Security System] :sunglasses:')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''
        
        
    def send_password_reset_email():
        try:
            # reset_email = st.text_input('Email Address')
            request_ref = f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={config['API_KEY']}"
            headers = {"content-type": "application/json; charset=UTF-8"}
            data = json.dumps({"requestType": "PASSWORD_RESET", "email": email})
            request_object = requests.post(request_ref, headers=headers, data=data)
        
            if request_object.status_code == 200:
                st.success("Resent link send.")   
            else:
                st.error("Please enter details correctly")
        except Exception as e:
            st.error(e)
        


    def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
        try:
            rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": return_secure_token
            }
            if username:
                payload["displayName"] = username 
            payload = json.dumps(payload)
            r = requests.post(rest_api_url, params={"key": config["API_KEY"]}, data=payload)
            try:
                return r.json()['email']
            except:
                st.warning(r.json()['error']['message'])
        except Exception as e:
            st.warning(f'Signup failed: {e}')

    def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

        try:
            payload = {
                "returnSecureToken": return_secure_token
            }
            if email:
                payload["email"] = email
            if password:
                payload["password"] = password
            payload = json.dumps(payload)
            # print('payload sigin',payload)
            r = requests.post(rest_api_url, params={"key": config["API_KEY"]}, data=payload)
            try:
            
                data = r.json()
                user_info = {
                    'email': data['email'],
                    'username': data.get('displayName')  # Retrieve username if available
                }
               # st.write(user_info)
                return user_info

            except:
                st.warning(data['error']['message'])
        except Exception as e:
            st.warning(f'Login failed: {e}')

    def f(): 
        try:

            userinfo = sign_in_with_email_and_password(st.session_state.email_input,st.session_state.password_input)
            st.session_state.username = userinfo['username']
            st.session_state.useremail = userinfo['email']

            
            global Usernm
            Usernm=(userinfo['username'])
            
            st.session_state.signedout = True
            st.session_state.signout = True    
  
            
        except: 
            st.warning('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False   
        st.session_state.username = ''

     
    if "signedout"  not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False    
              
    
    if  not st.session_state["signedout"]: # only show if the state is False, hence the button has never been clicked
        choice = st.selectbox('Login/Signup',['Login','Sign up', "Reset Password"])
        email = st.text_input('Email Address')
        if choice != "Reset Password":
            password = st.text_input('Password',type='password')
            st.session_state.email_input = email
            st.session_state.password_input = password

        
        if choice == 'Sign up':
            username = st.text_input("Enter  your unique username")
            
            if st.button('Create my account'):
                # user = auth.create_user(email = email, password = password,uid=username)
                user = sign_up_with_email_and_password(email=email,password=password,username=username)
                # print(user)
                if user != None:
                    st.success('Account created successfully!')
                    st.markdown('Please Login using your email and password')
                    st.balloons()
        if choice == "Login":
            # st.button('Login', on_click=f)          
            st.button('Login', on_click=f)
            
        if choice == "Reset Password":
            if st.button("Reset password"):
                send_password_reset_email()
            
            
            
    if st.session_state.signout:
                st.text('Name '+st.session_state.username)
                st.text('Email id: '+st.session_state.useremail)
                st.button('Sign out', on_click=t) 
                home_page()
                                  
if __name__ == "__main__":      
    app()
