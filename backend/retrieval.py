import os
import sys
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# A specialized model for re-ranking search results for higher accuracy.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Initialize Clients and Models ---
# This should be done once when your application starts to avoid reloading.
print("Initializing models... This may take a moment.")
# Model for creating vector embeddings (the 'retriever')
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# Model for re-scoring the retrieved results (the 're-ranker')
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
# Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
print("Models and client initialized successfully.")


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
        # === STAGE 1: RETRIEVAL ===
        # Generate embedding for the question using the base embedding model.
        question_embedding = embedding_model.encode(question).tolist()
        
        # Retrieve a larger set of initial candidates from Qdrant.
        initial_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=retrieve_top_k,
            with_payload=True
        )
        
        if not initial_results:
            return {
                "query": question, "results": [], "total_results": 0,
                "time_taken": round(time.time() - start_time, 3)
            }

        # === STAGE 2: RE-RANKING ===
        # Prepare pairs of [question, document_text] for the cross-encoder.
        cross_inp = [[question, result.payload.get("text", "")] for result in initial_results]
        
        # Predict more accurate relevance scores for these pairs.
        cross_scores = cross_encoder.predict(cross_inp)
        
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
if __name__ == '__main__':
    test_question = "What is the waiting period for knee surgery with a 3 month old policy?"
    search_result = search_documents(test_question)
    
    import json
    print("\n--- Search Results ---")
    print(json.dumps(search_result, indent=2))