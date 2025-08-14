from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from auth.routes import auth, get_db_connection, token_required
import os
import time
from retrieval import search_documents
from llm_handler import generate_response  # Import the new LLM handler
import sys
import threading

# print("Python version:", sys.version)

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

@app.route('/api/health/qdrant')
def check_qdrant():
    """Health check for Qdrant connection"""
    try:
        from retrieval import QdrantClient, QDRANT_URL, QDRANT_API_KEY
        
        # Create client and test connection
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10  # 10 second timeout
        )
        collections = client.get_collections()
        
        return jsonify({
            "status": "connected",
            "collections": [c.name for c in collections.collections]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/v1/hackrx/run', methods=['POST'])
# @token_required  # Uncomment for production
def hackrx_run():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received request to /api/v1/hackrx/run")
    data = request.get_json()
    
    print(f"Request data: {data}")
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    top_k = data.get('top_k', 4)
    use_llm = data.get('use_llm', True)
    include_raw = data.get('include_raw', False)
    
    start_time = time.time()
    
    try:
        # First get raw search results
        print(f"Starting document search for: {question}")
        search_result = search_documents(question, top_k)
        print(f"Search complete in {time.time() - start_time:.2f}s. Found {len(search_result.get('results', []))} results")
        
        # If no results or LLM not requested, return raw results
        if not search_result.get('results') or not use_llm:
            return jsonify(search_result), 200
            
        # If LLM is requested, use Gemini to combine the results
        print("Starting LLM response generation")
        llm_start_time = time.time()
        llm_response = generate_response(question, search_result['results'])
        print(f"LLM response generated in {time.time() - llm_start_time:.2f}s")
        
        # Rest of the function remains the same...
        
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


# Start a background thread to preload models
def preload_models():
    print("Preloading ML models...")
    try:
        from retrieval import get_embedding_model, get_cross_encoder, get_qdrant_client
        from llm_handler import ensure_genai_initialized
        
        # Load models in background
        get_embedding_model()
        get_cross_encoder()
        get_qdrant_client()
        ensure_genai_initialized()
        print("✓ All models preloaded successfully")
    except Exception as e:
        print(f"! Error preloading models: {e}")

# Start preloading after app is created but before it runs
threading.Thread(target=preload_models, daemon=True).start()

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

