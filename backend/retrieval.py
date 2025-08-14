import os
import sys
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient

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

# Lazy loading functions
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Ensure CPU inference - more stable for production and avoids CUDA errors
        device = "cpu"
        print(f"Loading embedding model on {device}...")
        
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            # Force model to correct device if needed
            if hasattr(_embedding_model, "to"):
                _embedding_model.to(device)
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            raise
            
    return _embedding_model

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        import torch
        
        # Ensure CPU inference - more stable for production
        device = "cpu"
        print(f"Loading cross-encoder model on {device}...")
        
        try:
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
        except Exception as e:
            print(f"Error loading cross-encoder model: {str(e)}")
            raise
            
    return _cross_encoder

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
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
            question_embedding = embedding_model.encode(question, convert_to_tensor=False)
            if not isinstance(question_embedding, list):
                question_embedding = question_embedding.tolist()
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
            initial_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_embedding,
                limit=retrieve_top_k,
                with_payload=True
            )
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
            
            # Predict more accurate relevance scores for these pairs.
            cross_scores = cross_encoder.predict(cross_inp)
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
if __name__ == '_main_':
    test_question = "What is the waiting period for knee surgery with a 3 month old policy?"
    search_result = search_documents(test_question)
    
    import json
    print("\n--- Search Results ---")
    print(json.dumps(search_result, indent=2))