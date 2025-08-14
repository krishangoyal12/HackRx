import os
import sys
import time
from functools import wraps
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize clients and models only when needed
_embedding_model = None
_cross_encoder = None
_qdrant_client = None

# --- Simple circuit breaker implementation (in-file) ---
_circuit_states = {}

def circuit_breaker(name, failure_threshold=3, recovery_timeout=60):
    """Simple circuit breaker decorator.

    name: key for tracking this circuit
    failure_threshold: failures before opening
    recovery_timeout: seconds to keep circuit open
    """
    state = _circuit_states.setdefault(name, {"failures": 0, "opened_at": None})

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If circuit is open and recovery timeout not passed, raise
            if state["opened_at"] is not None:
                if time.time() - state["opened_at"] < recovery_timeout:
                    raise Exception(f"Circuit '{name}' is open")
                else:
                    # reset circuit
                    state["failures"] = 0
                    state["opened_at"] = None

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                state["failures"] += 1
                if state["failures"] >= failure_threshold:
                    state["opened_at"] = time.time()
                raise

            # success -> reset failures
            state["failures"] = 0
            state["opened_at"] = None
            return result

        return wrapper
    return decorator

# Lazy loading functions
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        # Ensure CPU inference - safer for production
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        print(f"Loading embedding model on {device}...")

        try:
            # Use a small retry for model load to avoid transient download issues
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
            def _load():
                return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

            _embedding_model = _load()
            # If model has .to, ensure device placement
            if hasattr(_embedding_model, "to"):
                try:
                    _embedding_model.to(device)
                except Exception:
                    # some SentenceTransformer wrappers don't need .to
                    pass
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            raise

    return _embedding_model

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        device = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
        print(f"Loading cross-encoder model on {device}...")

        try:
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
            def _load_ce():
                return CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)

            _cross_encoder = _load_ce()
        except Exception as e:
            print(f"Error loading cross-encoder model: {str(e)}")
            raise

    return _cross_encoder

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        # Use a short retry when creating client (network issues, DNS)
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
        def _create_client():
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=int(os.getenv("QDRANT_TIMEOUT", "30")))

        _qdrant_client = _create_client()
    return _qdrant_client

def search_documents(question, retrieve_top_k=10, rerank_top_k=4):
    """
    Search for relevant document chunks using a two-stage retrieve-and-rerank pipeline.
    
    Args:
        question (str): The user's question.
        retrieve_top_k (int): Number of initial candidates to retrieve from Qdrant.
        rerank_top_k (int): Final number of results to return after re-ranking.
        
    Returns:
        dict: Search results with metadata and improved relevance scores.
    """
    start_time = time.time()
    
    try:
        print(f"Starting search for query: '{question}'")
        
        # Initialize models only when needed
        print("Loading models...")
        try:
            embedding_model = get_embedding_model()
            cross_encoder = get_cross_encoder()
            qdrant_client = get_qdrant_client()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return {
                "query": question, 
                "results": [], 
                "total_results": 0,
                "error": f"Model loading error: {str(e)}",
                "time_taken": round(time.time() - start_time, 3)
            }
        
        # === STAGE 1: RETRIEVAL ===
        # Generate embedding for the question using the base embedding model.
        print("Generating embedding...")
        try:
            # Retry embedding generation on transient failures
            @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), retry=retry_if_exception_type(Exception))
            def _encode(q):
                emb = embedding_model.encode(q, convert_to_tensor=False)
                # Normalize to list
                if not isinstance(emb, list):
                    try:
                        emb = emb.tolist()
                    except Exception:
                        emb = list(emb)
                return emb

            question_embedding = _encode(question)
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return {
                "query": question, 
                "results": [], 
                "total_results": 0,
                "error": f"Embedding generation error: {str(e)}",
                "time_taken": round(time.time() - start_time, 3)
            }
        
        # Retrieve a larger set of initial candidates from Qdrant.
        print("Querying Qdrant...")
        try:
            # Wrap Qdrant search with retry and circuit breaker
            @circuit_breaker("qdrant_search", failure_threshold=3, recovery_timeout=int(os.getenv("QDRANT_CB_RECOVERY", "60")))
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=0.5, max=8), retry=retry_if_exception_type(Exception))
            def _qdrant_search(vec, limit):
                return qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vec,
                    limit=limit,
                    with_payload=True
                )

            initial_results = _qdrant_search(question_embedding, retrieve_top_k)
            print(f"Found {len(initial_results)} initial results from Qdrant")
        except Exception as e:
            print(f"Error querying Qdrant: {str(e)}")
            return {
                "query": question, 
                "results": [], 
                "total_results": 0,
                "error": f"Qdrant query error: {str(e)}",
                "time_taken": round(time.time() - start_time, 3)
            }
        
        if not initial_results:
            return {
                "query": question, "results": [], "total_results": 0,
                "time_taken": round(time.time() - start_time, 3)
            }

        # === STAGE 2: RE-RANKING ===
        print("Starting re-ranking...")
        try:
            # Prepare pairs of [question, document_text] for the cross-encoder.
            cross_inp = [[question, result.payload.get("text", "")] for result in initial_results]

            # Wrap cross-encoder predict with retry and circuit breaker
            @circuit_breaker("cross_encoder", failure_threshold=3, recovery_timeout=int(os.getenv("CE_CB_RECOVERY", "60")))
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), retry=retry_if_exception_type(Exception))
            def _predict(pairs):
                return cross_encoder.predict(pairs)

            cross_scores = _predict(cross_inp)
            print("Re-ranking completed successfully")
        except Exception as e:
            print(f"Error during re-ranking: {str(e)}")
            # If re-ranking fails, we can still return initial results without re-ranking
            print("Falling back to initial results without re-ranking")
            
            # Format the initial results
            final_results = []
            for idx, result in enumerate(initial_results[:min(rerank_top_k, len(initial_results))]):
                result_item = {
                    "id": idx + 1,
                    "text": result.payload.get("text", "No content available"),
                    "source": result.payload.get("source", "Unknown source"),
                    "page": result.payload.get("page", "Unknown page"),
                    "original_similarity": round(result.score, 4),
                }
                final_results.append(result_item)
            
            return {
                "query": question,
                "results": final_results,
                "total_results": len(final_results),
                "error": f"Re-ranking error (using fallback results): {str(e)}",
                "time_taken": round(time.time() - start_time, 3)
            }
        
        # Create a list of tuples with (result, rerank_score) instead of modifying ScoredPoint objects
        reranked_results = []
        for idx, result in enumerate(initial_results):
            reranked_results.append({
                "original_result": result,
                "rerank_score": float(cross_scores[idx])
            })
            
        # Sort the results based on the new cross-encoder scores in descending order.
        reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)
        
        # --- Format and return the final, top-k results ---
        final_results = []
        for idx, item in enumerate(reranked_results[:rerank_top_k]):
            result = item["original_result"]
            result_item = {
                "id": idx + 1,
                "text": result.payload.get("text", "No content available"),
                "source": result.payload.get("source", "Unknown source"),
                "page": result.payload.get("page", "Unknown page"),
                "original_similarity": round(result.score, 4),  # From Qdrant's vector search
                "reranked_score": round(item["rerank_score"], 4)  # More accurate score from CrossEncoder
            }
            final_results.append(result_item)
            
        return {
            "query": question,
            "results": final_results,
            "total_results": len(final_results),
            "time_taken": round(time.time() - start_time, 3)
        }
        
    except Exception as e:
        print(f"Error during document search with re-ranking: {str(e)}")
        return {
            "query": question, "results": [], "total_results": 0, "error": str(e),
            "time_taken": round(time.time() - start_time, 3)
        }

# --- Example Usage ---
if __name__ == 'main':
    test_question = "What is the waiting period for knee surgery with a 3 month old policy?"
    search_result = search_documents(test_question)
    
    import json
    print("\n--- Search Results ---")
    print(json.dumps(search_result, indent=2))