import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Knowledge Base Background Task ---
def _create_kb_task(url: str):
    print(f"--- Background Task: Starting KB creation for {url} ---")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        print(f"--- Background Task: Loaded {len(docs)} document(s) ---")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"--- Background Task: Split into {len(splits)} chunks ---")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        db = FAISS.from_documents(splits, embeddings)
        db.save_local("faiss_index")
        print("--- ✅ Background Task: Knowledge Base creation SUCCESSFUL. ---")
        
    except Exception as e:
        print(f"--- ❌ Background Task ERROR: {e} ---")

# --- 3. Create the FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Define API Request Models ---
class CreateKBRequest(BaseModel):
    url: str

# --- 5. The Endpoints ---

@app.get("/")
async def root():
    return {"message": "Main Control Server is running."}

# ENDPOINT 1: Create Knowledge Base
@app.post("/create-knowledge-base")
async def create_knowledge_base(request: CreateKBRequest, background_tasks: BackgroundTasks):
    print(f"---  KNOWLEDGE BASE: Received request to create KB from {request.url} ---")
    background_tasks.add_task(_create_kb_task, request.url)
    return {"message": "Knowledge base creation started in the background. This may take a few minutes."}

if __name__ == "__main__":
    print("--- 🚀 Starting Main Control Server on http://localhost:8002 ---")
    uvicorn.run(app, host="0.0.0.0", port=8002)