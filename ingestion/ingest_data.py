import os
import sys
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
DOCUMENTS_DIR = "policy_documents"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_SIZE = 768

# --- Setup base directory ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
folder_path = os.path.join(BASE_DIR, DOCUMENTS_DIR)

def load_documents_from_folder(path):
    """Loads PDF documents from a specified folder, creating it if it doesn't exist."""
    documents = []
    print(f"Loading documents from '{path}'...")

    if not os.path.isdir(path):
        print(f"Directory not found at '{path}'. Creating it...")
        try:
            os.makedirs(path)
            print(f"Created directory. Please add PDF documents to '{path}' and run again.")
            return []
        except Exception as e:
            print(f"Error: Failed to create directory: {e}")
            return []

    pdf_files = [f for f in os.listdir(path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{path}'. Please add documents to ingest.")
        return []

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(path, filename)
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "doc_id": filename.replace(".pdf", ""),
                            "ingested_at": datetime.now().isoformat()
                        }
                    })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print(f"Successfully loaded and extracted text from {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller, manageable chunks."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )

    all_chunks = []
    for doc in tqdm(documents, desc="Chunking Pages"):
        metadata = doc["metadata"]
        chunks = text_splitter.create_documents([doc["text"]], metadatas=[metadata])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            all_chunks.append(chunk)

    print(f"Created {len(all_chunks)} text chunks.")
    return all_chunks

def main():
    """Main function to run the data ingestion pipeline."""
    # Step 1: Load PDFs and extract text
    raw_documents = load_documents_from_folder(folder_path)
    if not raw_documents:
        print("No documents loaded. Exiting.")
        return

    # Step 2: Chunk documents
    chunked_documents = chunk_documents(raw_documents)
    if not chunked_documents:
        print("No chunks created. Exiting.")
        return

    # Step 3: Initialize embedding model and Qdrant client
    print("Initializing embedding model and Qdrant client...")
    if not QDRANT_URL:
        print("Error: QDRANT_URL environment variable not set.")
        sys.exit(1)
        
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60  # Increased timeout for cloud operations
        )
        # Test connection
        qdrant_client.get_collections()
        print("Initialization complete. Connected to Qdrant.")
    except Exception as e:
        print(f"Error: Could not connect to Qdrant. {e}")
        sys.exit(1)

    # Step 4: Check if collection exists, otherwise create it
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        if COLLECTION_NAME not in collection_names:
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            print(f"Collection '{COLLECTION_NAME}' created successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Skipping recreation.")
    except Exception as e:
        print(f"Error managing collection: {e}")
        return

    # Step 5: Batch embed and upsert into Qdrant
    print("Generating embeddings and storing in Qdrant...")
    batch_size = 64  # Smaller batch size for more reliable cloud uploads
    failed_batches = 0

    for i in tqdm(range(0, len(chunked_documents), batch_size), desc="Upserting to Qdrant"):
        batch = chunked_documents[i:i + batch_size]
        texts_to_embed = [chunk.page_content for chunk in batch]

        try:
            embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Embedding generation failed for batch starting at index {i}: {e}")
            failed_batches += 1
            continue

        payloads = [{"text": chunk.page_content, **chunk.metadata} for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(ids=ids, vectors=embeddings, payloads=payloads),
                wait=True
            )
        except Exception as e:
            print(f"Failed to upsert batch starting at index {i}: {e}")
            failed_batches += 1

    # Step 6: Final summary
    print("\n==================================================")
    print("Data ingestion process complete!")
    if failed_batches > 0:
        print(f"Warning: {failed_batches} batches failed to upload.")
        
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Total points in collection '{COLLECTION_NAME}': {collection_info.points_count}")
    except Exception as e:
        print(f"\nCould not retrieve final collection info: {e}")
        print("This might be due to a client/server version mismatch.")
        print("Try upgrading your client: pip install --upgrade qdrant-client")
    print("==================================================")

if __name__ == "__main__":
    main()