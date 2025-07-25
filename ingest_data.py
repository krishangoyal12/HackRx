import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid
from datetime import datetime
from transformers import AutoTokenizer

# --- Configuration Constants ---
DOCUMENTS_DIR = "policy_documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # More powerful model
VECTOR_SIZE = 384  # For all-mpnet-base-v2

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
folder_path = os.path.join(BASE_DIR, DOCUMENTS_DIR)

# Tokenizer to calculate token length for better chunking
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def load_documents_from_folder(folder_path):
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
                    if text:
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1,
                                "doc_id": filename.replace(".pdf", ""),  # Enriched metadata
                                "ingested_at": datetime.utcnow().isoformat()  # Ingestion time
                            }
                        })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Successfully loaded and extracted text from {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    print("Chunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better handling of policies
        chunk_overlap=200,  # Overlap for better continuity
        length_function=lambda text: len(tokenizer.encode(text, truncation=False)),
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
    # Step 1: Load PDFs and extract text
    raw_documents = load_documents_from_folder(folder_path)
    if not raw_documents:
        print("No documents loaded. Exiting.")
        return

    # Step 2: Chunk documents
    chunked_documents = chunk_documents(raw_documents)

    # Step 3: Initialize embedding model and Qdrant client
    print("Initializing embedding model and Qdrant client...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Initialization complete.")

    # Step 4: Check if collection exists before creating
    existing_collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in existing_collections]

    if COLLECTION_NAME in collection_names:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping recreation.")
    else:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")

    # Step 5: Batch embed and upsert into Qdrant
    print("Generating embeddings and storing in Qdrant...")
    batch_size = 128

    for i in tqdm(range(0, len(chunked_documents), batch_size), desc="Upserting to Qdrant"):
        batch = chunked_documents[i:i + batch_size]
        texts_to_embed = [chunk.page_content for chunk in batch]

        try:
            embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Embedding generation failed for batch {i}-{i + batch_size}: {e}")
            continue

        # Enrich metadata and add text to the payload
        payloads = [
            {
                **chunk.metadata,
                "text": chunk.page_content
            }
            for chunk in batch
        ]

        ids = [str(uuid.uuid4()) for _ in batch]

        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                ),
                wait=True
            )
        except Exception as e:
            print(f"Failed to upsert batch {i}-{i + batch_size}: {e}")

    print("\n==================================================")
    print("Data ingestion complete!")
    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Total points in collection: {collection_info.points_count}")
    print("==================================================")

if __name__ == "__main__":
    main()