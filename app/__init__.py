from flask import Flask
from app.routes import main

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../config.py')
    
    app.register_blueprint(main)
    
    return app
