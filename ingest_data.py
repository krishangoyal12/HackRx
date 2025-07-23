# ==============================================================================
# Phase 1: Data Ingestion and Vector Store Creation
#
# Description:
# This script performs the complete data ingestion pipeline for the PolicyPal chatbot.
# It reads PDF documents, processes the text, generates vector embeddings, and
# stores them in a Qdrant vector database. This script needs to be run only once
# to build the knowledge base.
#
# Process:
# 1.  Load PDFs: Scans a directory for PDF files.
# 2.  Extract Text: Uses PyMuPDF to extract raw text and metadata from each page.
# 3.  Chunk Text: Uses LangChain's splitter to break the text into smaller,
#     semantically coherent chunks.
# 4.  Embed & Store: Generates embeddings for each chunk using a Sentence
#     Transformer model and upserts the vectors and their payloads into a
#     Qdrant collection.
#
# ==============================================================================

import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm # For progress bars
import uuid

# --- Configuration Constants ---
DOCUMENTS_DIR = "policy_documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # This is specific to the all-MiniLM-L6-v2 model
folder_path = r"C:\Users\krish\Desktop\chat bot\policy_documents"

def load_documents_from_folder(folder_path):
    """
    Loads all PDF documents from a specified folder and extracts text.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list: A list of LangChain Document objects, each containing the page content
              and metadata (source file and page number).
    """
    documents = []
    print(f"Loading documents from '{folder_path}'...")
    
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return []

    for filename in tqdm(os.listdir(folder_path), desc="Processing PDFs"):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text:  # Only add pages with actual text content
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1
                            }
                        })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    print(f"Successfully loaded and extracted text from {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    """
    Splits the loaded documents into smaller chunks for effective embedding.

    Args:
        documents (list): A list of document dictionaries.

    Returns:
        list: A list of smaller text chunks with their associated metadata.
    """
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking Pages"):
        # The splitter expects a list of texts, so we process one page at a time
        chunks = text_splitter.create_documents([doc["text"]], metadatas=[doc["metadata"]])
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} text chunks.")
    return all_chunks

def main():
    """
    Main function to execute the entire data ingestion pipeline.
    """
    # Step 1: Load and extract text from PDFs
    raw_documents = load_documents_from_folder(folder_path)
    if not raw_documents:
        print("No documents were loaded. Please check the 'policy_documents' folder.")
        return

    # Step 2: Chunk the documents into smaller pieces
    # We pass the raw document dictionaries to the chunking function
    chunked_documents = chunk_documents(raw_documents)

    # Step 3: Initialize models and database client
    print("Initializing embedding model and Qdrant client...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Initialization complete.")

    # Step 4: Create a collection in Qdrant if it doesn't exist
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Collection may already exist. Error: {e}")


    # Step 5: Generate embeddings and upsert into Qdrant in batches
    print("Generating embeddings and storing in Qdrant...")
    batch_size = 128  # Process documents in batches for efficiency
    
    for i in tqdm(range(0, len(chunked_documents), batch_size), desc="Upserting to Qdrant"):
        batch = chunked_documents[i:i + batch_size]
        
        # Get the text content for the batch
        texts_to_embed = [chunk.page_content for chunk in batch]
        
        # Generate embeddings for the batch
        embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()
        
        # Prepare the payload (metadata) and unique IDs for each point
        payloads = [chunk.metadata for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        # Upsert the batch to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=payloads
            ),
            wait=True # Wait for the operation to complete
        )

    print("\n==================================================")
    print("Data ingestion complete!")
    print(f"All document chunks have been embedded and stored in the Qdrant collection: '{COLLECTION_NAME}'")
    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Total points in collection: {collection_info.points_count}")
    print("==================================================")


if __name__ == "__main__":
    main()