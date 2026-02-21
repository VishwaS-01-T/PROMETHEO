import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional # Added for typing

# --- 1. Load Models (but not the DB) ---
print("--- 🧠 Initializing Embedding model... ---")
# These are loaded once at the start
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print("--- ✅ Embedding model loaded. ---")

# ---
# --- THIS IS THE FIX ---
# ---
# We start with an empty DB and Retriever
db: Optional[FAISS] = None
retriever = None

def load_knowledge_base() -> bool:
    """
    Tries to load the FAISS index from disk.
    Returns True on success, False on failure.
    """
    global db, retriever
    
    if not os.path.exists("faiss_index"):
        print("--- ⚠️  Knowledge base not found. Waiting for creation... ---")
        return False
    
    try:
        print("--- 🧠 Found 'faiss_index' folder. Loading... ---")
        db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
        print("--- ✅ Knowledge Base Loaded. ---")
        return True
    except Exception as e:
        print(f"--- ❌ ERROR: Failed to load 'faiss_index': {e} ---")
        # This could happen if the folder is empty or corrupted
        return False

# --- 2. Create the FastAPI App ---
app = FastAPI()

# --- 3. Define API Request Models ---
class SearchQuery(BaseModel):
    query: str

# --- 4. The Endpoints ---

@app.get("/")
async def root():
    return {"message": "Knowledge Base (KB) Server is running."}

@app.post("/search-kb")
async def search_knowledge_base(search_query: SearchQuery):
    """
    Search the knowledge base for relevant information.
    """
    
    # Check if DB is loaded. If not, try to load it.
    if db is None:
        if not load_knowledge_base():
            return {"result": "The knowledge base is currently being created. Please tell the user to wait a moment and try asking again."}

    # If we get here, the db is loaded and retriever is available
    try:
        print(f"--- 🔍 KB Query Received: {search_query.query} ---")
        docs = retriever.invoke(search_query.query)
        context = "\n\n".join([d.page_content for d in docs])
        print(f"--- ✅ Context Found: {context[:100]}... ---")
        return {"result": context}
    except Exception as e:
        print(f"--- ❌ KB SEARCH ERROR: {e} ---")
        return {"result": "I could not find an answer to that."}

if __name__ == "__main__":
    print("--- 🚀 Starting KB Server on http://localhost:8001 ---")
    uvicorn.run(app, host="0.0.0.0", port=8001)