from flask import Flask
from dotenv import load_dotenv
from auth.routes import auth, get_db_connection 
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Register authentication blueprint
app.register_blueprint(auth, url_prefix='/auth')

@app.route('/')
def home():
    return "✅ Chatbot Backend Running"

if __name__ == '__main__':
    app.run(debug=True)

    # Test DB connection before starting server
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(cur.fetchone())
    cur.close()
    conn.close()

