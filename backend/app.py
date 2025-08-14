from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from auth.routes import auth, get_db_connection, token_required
import os
import time
import sys
import threading

# Print Python version for debugging
print(f"Python version: {sys.version}")

# Import the dependencies safely with error handling
try:
    from retrieval import search_documents
    print("✓ Successfully imported retrieval module")
except ImportError as e:
    print(f"! Error importing retrieval module: {e}")
    def search_documents(*args, **kwargs):
        return {"error": f"Retrieval module not available: {str(e)}", "results": []}
        
try:
    from llm_handler import generate_response
    print("✓ Successfully imported LLM handler")
except ImportError as e:
    print(f"! Error importing LLM handler: {e}")
    def generate_response(*args, **kwargs):
        return {"answer": "LLM module not available", "sources": [], "provider": "none"}

# print("Python version:", sys.version)

load_dotenv()

app = Flask(_name_)
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
        
        # Clean and format the answer text
        clean_answer = llm_response["answer"]
        # Replace newlines with spaces
        clean_answer = clean_answer.replace('\n', ' ')
        # Replace multiple spaces with a single space
        import re
        clean_answer = re.sub(r'\s+', ' ', clean_answer)
        # Remove markdown asterisks for bold formatting
        clean_answer = clean_answer.replace('', '')
        # Remove markdown bullets
        clean_answer = clean_answer.replace('* ', '')
        
        # Create clean response (without raw results or sources)
        response = {
            "query": question,
            "answer": clean_answer.strip(),
            "time_taken": round(time.time() - start_time, 3)
        }
        
        # Add provider info optionally
        include_provider = data.get('include_provider', False)
        if include_provider:
            response["llm_provider"] = llm_response["provider"]
        
        # Only add raw results if explicitly requested
        if include_raw:
            response["raw_results"] = search_result['results']
            
        # Only include sources if explicitly requested
        include_sources = data.get('include_sources', False)
        if include_sources:
            response["sources"] = llm_response["sources"]
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "query": question,
            "time_taken": round(time.time() - start_time, 3)
        }), 500


@app.route('/api/v1/hackrx/answer', methods=['POST'])
# @token_required  # Uncomment for production
def simple_answer():
    """Simplified endpoint that returns just the answer with minimal formatting"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    top_k = data.get('top_k', 4)
    
    start_time = time.time()
    
    try:
        # Get search results
        search_result = search_documents(question, top_k)
        
        # If no results, return simple message
        if not search_result.get('results'):
            return jsonify({
                "answer": "I couldn't find any relevant information to answer your question."
            }), 200
            
        # Generate response using LLM
        llm_response = generate_response(question, search_result['results'])
        
        # Clean and format answer text
        clean_answer = llm_response["answer"]
        # Replace newlines with spaces
        clean_answer = clean_answer.replace('\n', ' ')
        # Replace multiple spaces with a single space
        import re
        clean_answer = re.sub(r'\s+', ' ', clean_answer)
        # Remove markdown asterisks for bold formatting
        clean_answer = clean_answer.replace('', '')
        # Remove markdown bullets
        clean_answer = clean_answer.replace('* ', '')
        
        # Return just the answer in a clean format
        return jsonify({
            "answer": clean_answer.strip()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Start a background thread to preload models
def preload_models():
    print("Preloading ML models...")
    try:
        print("Importing modules...")
        from retrieval import get_embedding_model, get_cross_encoder, get_qdrant_client
        # Import llm_handler conditionally as it might not be needed immediately
        try:
            from llm_handler import ensure_genai_initialized
        except ImportError:
            print("Note: llm_handler not available - skipping Gemini initialization")
            ensure_genai_initialized = None
        
        # Load models in background with proper error handling
        print("Loading embedding model...")
        try:
            get_embedding_model()
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"! Error loading embedding model: {e}")
            print("  Will retry on first request")
            
        print("Loading cross-encoder model...")
        try:
            get_cross_encoder()
            print("✓ Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"! Error loading cross-encoder model: {e}")
            print("  Will retry on first request")
            
        print("Loading Qdrant client...")
        try:
            get_qdrant_client()
            print("✓ Qdrant client initialized successfully")
        except Exception as e:
            print(f"! Error initializing Qdrant client: {e}")
            
        # Only try to load Gemini if available
        if ensure_genai_initialized:
            print("Initializing LLM...")
            try:
                ensure_genai_initialized()
                print("✓ LLM initialized successfully")
            except Exception as e:
                print(f"! Error initializing LLM: {e}")
                print("  Will retry when needed")
                
        print("✓ Model preloading completed")
    except Exception as e:
        print(f"! Error in preload process: {e}")

# Start preloading after app is created but before it runs
threading.Thread(target=preload_models, daemon=True).start()

if _name_ == '_main_':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    # Test DB connection before starting server
    # conn = get_db_connection()
    # cur = conn.cursor()
    # cur.execute("SELECT version();")
    # print(cur.fetchone())
    # cur.close()
    # conn.close()