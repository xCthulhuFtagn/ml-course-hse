from app import app, db
from flask import request
import bcrypt

from models import User

def get_hashed_password(password: str):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed_password):
    return get_hashed_password(password) == hashed_password

@app.route('/')
def base_handler_func():
    return '', 200

@app.route('/signup')
def signup_handler(request):
    data = request.json
    email = data['email']
    password_hash = get_hashed_password(data['password'])
    
    u = User(email=email,password_hash=password_hash)
    db.session.add(u)
    db.session.commit()
    return jsonify({'msg': 'User created'}), 200