import firebase_admin
from firebase_admin import credentials

def init():
    # Check if Firebase Admin SDK has been initialized
    if not firebase_admin._apps:
        # If not initialized, initialize Firebase Admin SDK
        cred = credentials.Certificate("Credentials/serviceAccount.json")
        app = firebase_admin.initialize_app(cred)
        
    app = firebase_admin.get_app()
    # print(app)
    return app