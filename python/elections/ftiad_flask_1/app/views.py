from flask import render_template, request, redirect, url_for, jsonify
from app import app
from .utils import get_hashed_password, check_password
from .models import User
from app import db
import sys

@app.route('/')
def hello_world():
    return "Hello, world!"

@app.route('/signup', methods=['POST'])
def sign_up():
    data = request.json
    email = data['email']
    usertype = data['usertype']
    password_hash = get_hashed_password(data['password'])
    u = User(email=email, usertype=usertype, password_hash=password_hash)
    db.session.add(u)
    db.session.commit()
    return jsonify({'response': f'User {email} created'}), 200





