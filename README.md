# HackRx Chatbot

**Date:** July 23, 2025

## Today's Development Goals

The focus for today is to build the core components of the RAG-based chatbot. The data ingestion and vectorization pipeline is complete. Now, we need to implement the logic for retrieval and generation, and expose it through a user-facing application.

### Task 1: Implement the RAG Handler (`rag_handler.py`)  

This module will be the bridge between the user's query and the knowledge base.

- [ ] **Connect to Qdrant:** Establish a connection to the existing Qdrant vector database.  
- [ ] **Query Embedding:** Create a function that takes a user's text query and generates an embedding using the `all-MiniLM-L6-v2` model.  
- [ ] **Similarity Search:** Implement the logic to use the query embedding to search the `policy_documents` collection in Qdrant and retrieve the most relevant text chunks.  
- [ ] **Context Formatting:** Create a function to process and format the retrieved documents into a clean context string to be fed into the language model.  

#### How to Run

1.  **Start Qdrant:** Make sure your Qdrant Docker container is running.  
2.  **Run the data ingestion script (if needed):**  
    ```bash
    python ingest_data.py
    ```

---

**Date:** July 26, 2025

## Today's Development Goals

Today’s focus is on implementing secure user authentication to enable personalized access for the chatbot.  

### Task 2: JWT Authentication & Protected Routes  
This task focuses on creating a secure authentication system to manage user access and protect backend endpoints.  

- [ ] **Signup/Login API:** Built POST /signup and POST /login routes in auth/routes.py.  
- [ ] **JWT Token Handling:** On successful login/signup, a JWT is generated and sent to the client.  
- [ ] **Token Verification:** Middleware checks the token in headers to allow/deny protected route access.  
- [ ] **PostgreSQL Integration:** User credentials are validated and stored in the Neon-hosted PostgreSQL database.  

#### How to Run:  
Start the Flask server.  
```python app.py```
