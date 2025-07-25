from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import os
import jwt
import datetime
from dotenv import load_dotenv
from functools import wraps

load_dotenv()
auth = Blueprint('auth', __name__)

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_EXPIRES = int(os.getenv("JWT_EXPIRES", 3600))  # default 1 hour

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

# Middleware to protect routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from header
        if 'Authorization' in request.headers:
            bearer = request.headers['Authorization']
            if bearer.startswith("Bearer "):
                token = bearer.split(" ")[1]

        if not token:
            return jsonify({"error": "Token is missing!"}), 401

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user = data  # Save user info
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired!"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token!"}), 401

        return f(*args, **kwargs)
    return decorated


@auth.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
        existing = cur.fetchone()
        if existing:
            return jsonify({"error": "User already exists"}), 409

        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s) RETURNING id",
                    (username, email, hashed_password))
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "message": "Signup successful!",
            "user": {"id": user_id, "username": username, "email": email}
        }), 201

    except psycopg2.Error as e:
        return jsonify({"error": str(e)}), 500


@auth.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[2], password):
            payload = {
                "id": user[0],
                "username": user[1],
                "email": email,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRES)
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

            return jsonify({
                "message": "Login successful!",
                "token": token,
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "email": email
                }
            }), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except psycopg2.Error as e:
        return jsonify({"error": str(e)}), 500


@auth.route('/users', methods=['GET'])
@token_required
def get_users():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email FROM users")
        users = cur.fetchall()
        cur.close()
        conn.close()

        user_list = [{"id": u[0], "username": u[1], "email": u[2]} for u in users]
        return jsonify(user_list), 200

    except psycopg2.Error as e:
        return jsonify({"error": str(e)}), 500
