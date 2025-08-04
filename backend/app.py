from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from auth.routes import auth, get_db_connection, token_required
import os
import time
from retrieval import search_documents
from llm_handler import generate_response  # Import the new LLM handler
import sys

print("Python version:", sys.version)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Enable CORS for all routes
CORS(app)

# Register authentication blueprint
app.register_blueprint(auth, url_prefix='/auth')

@app.route('/')
def home():
    return "✅ Chatbot Backend Running"

@app.route('/api/v1/hackrx/run', methods=['POST'])
# @token_required  # Uncomment for production
def hackrx_run():
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    top_k = data.get('top_k', 4)  # Default to 4 results for LLM context
    use_llm = data.get('use_llm', True)  # Whether to use Gemini
    include_raw = data.get('include_raw', False)  # New parameter to control raw results
    
    start_time = time.time()
    
    try:
        # First get raw search results
        search_result = search_documents(question, top_k)
        
        # If no results or LLM not requested, return raw results
        if not search_result.get('results') or not use_llm:
            return jsonify(search_result), 200
            
        # If LLM is requested, use Gemini to combine the results
        llm_response = generate_response(question, search_result['results'])
        
        # Create clean response (without raw results)
        response = {
            "query": question,
            "answer": llm_response["answer"],
            "sources": llm_response["sources"],
            "llm_provider": llm_response["provider"],
            "time_taken": round(time.time() - start_time, 3)
        }
        
        # Only add raw results if explicitly requested
        if include_raw:
            response["raw_results"] = search_result['results']
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "query": question,
            "time_taken": round(time.time() - start_time, 3)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    # Test DB connection before starting server
    # conn = get_db_connection()
    # cur = conn.cursor()
    # cur.execute("SELECT version();")
    # print(cur.fetchone())
    # cur.close()
    # conn.close()

