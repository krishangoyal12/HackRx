import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

_genai_initialized = False

load_dotenv()

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# REMOVED immediate initialization
# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)

def ensure_genai_initialized():
    """Lazy initialization of Gemini API to save memory at startup"""
    global _genai_initialized
    if not _genai_initialized and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _genai_initialized = True
        return True
    return bool(GEMINI_API_KEY)

def generate_response(question, search_results):
    """
    Generate a comprehensive answer using Gemini based on multiple search results
    
    Args:
        question (str): The user's original question
        search_results (list): List of retrieved document chunks with metadata
        
    Returns:
        dict: LLM response with attribution
    """
    # Prepare context from search results
    contexts = []
    sources = []
    
    # Exit early if no results
    if not search_results:
        return {
            "answer": "I don't have enough information to answer your question based on the available documents.",
            "sources": [],
            "provider": "none"
        }
    
    # Process each search result
    for result in search_results:
        # Format each result with its source
        context = f"Document: {result['source']}, Page: {result['page']}\n{result['text']}\n"
        contexts.append(context)
        
        # Add source if not already present
        source_info = {"source": result['source'], "page": result['page']}
        if source_info not in sources:
            sources.append(source_info)
    
    # Join all contexts
    all_contexts = "\n---\n".join(contexts)
    
    # Create the prompt for Gemini
    prompt = f"""
You are an insurance policy expert assistant. Your task is to provide a clear, helpful answer to the user's question based ONLY on the information in the provided policy document excerpts.

Please follow these guidelines:
1. Use ONLY information from the provided excerpts
2. If the information is not in the excerpts, say "I don't have enough information to answer that question."
3. Do not make up information
4. Write in a clear, friendly, and conversational tone
6. Be concise but thorough
7. Format the answer in a way that is easy to read

USER QUESTION: {question}

POLICY DOCUMENT EXCERPTS:
{all_contexts}

YOUR ANSWER:
"""
    
    # UPDATED: Check if Gemini API key is set using lazy initialization
    if not ensure_genai_initialized():
        return {
            "answer": "Google Gemini API key is not configured. Please add your GEMINI_API_KEY to the .env file.",
            "sources": sources,
            "provider": "none"
        }
    
    try:
        # Initialize the Gemini model - UPDATED TO GEMINI 2.0 FLASH
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response with Gemini 2.0 configurations
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Lower temperature for more factual responses
                max_output_tokens=1024,  # Generous length for comprehensive answers
                top_p=0.95,  # Slightly reduce randomness
                top_k=40  # Standard top_k parameter
            )
        )
        
        if response.text:
            return {
                "answer": response.text.strip(),
                "sources": sources,
                "provider": "gemini-2.0-flash"
            }
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {
            "answer": f"Sorry, I couldn't generate a response. Error: {str(e)}",
            "sources": sources,
            "provider": "error"
        }
    
    # Return a graceful error message if generation fails
    return {
        "answer": "Sorry, I couldn't generate a response. The Gemini service may be unavailable.",
        "sources": sources,
        "provider": "none"
    }
