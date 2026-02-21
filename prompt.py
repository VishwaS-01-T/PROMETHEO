import os
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing. Please set it in your .env file.")

# --- 2. FastAPI App & LLM Setup ---
app = FastAPI(title="Dynamic Prompt Generator API")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

# --- 3. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 4. The "Meta-Prompt" (A prompt that generates a prompt) ---
# This is the core logic.
async def create_system_prompt(product_name: str, product_url: str) -> str:
    print(f"Generating system prompt for {product_name}...")
    
    # --- A. Scrape the website ---
    loader = WebBaseLoader(product_url)
    try:
        # Run the synchronous .load() in a separate thread
        docs = await asyncio.to_thread(loader.load)
    except Exception as e:
        print(f"Error scraping {product_url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {e}")

    if not docs:
        print("Failed to load content.")
        raise HTTPException(status_code=404, detail="Could not load any content from the URL.")
        
    content = "\n\n".join(d.page_content for d in docs)[:15000]
    
    # --- B. Generate the new system prompt using the content ---
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert prompt engineer. Your task is to generate a complete system prompt for a friendly AI sales assistant named Alex."
        ),
        (
            "human",
            """
            The AI will be pitching a product called: "{product_name}"
            
            Use this scraped website content for all product knowledge:
            ---
            {content}
            ---

            Generate a complete system prompt that instructs the AI to do the following:
            1.  **First Message:** The AI's first message MUST be: "Hi, my name is Alex, and I'm a conversational AI. I'm calling today about {product_name}. Do you have 30 seconds for me to explain?"
            2.  **The Pitch:** If the user agrees, the AI must deliver a concise, compelling 5-6 line pitch based *only* on the provided website content.
            3.  **Next Step:** After delivering the pitch, the AI must ask a follow-up question to see if the user is interested in scheduling a meeting.
            
            Respond with *only* the system prompt itself, starting with "You are a friendly AI sales assistant...". Do not include any other text.
            """
        ),
    ])
    chain = prompt_template | llm | StrOutputParser()

    try:
        system_prompt = await chain.ainvoke({
            "product_name": product_name,
            "content": content
        })
        print(f"Generated system prompt: {system_prompt}")
        return system_prompt
    except Exception as e:
        print(f"Error calling Groq: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate prompt with LLM: {e}")

# --- 5. Define Request and Response Models ---
class PromptRequest(BaseModel):
    product_name: str
    product_url: str

class PromptResponse(BaseModel):
    system_prompt: str

# --- 6. The API Endpoint ---
@app.post("/generate-prompt", response_model=PromptResponse)
async def handle_generate_prompt(req: PromptRequest):
    print(f"Received API request for {req.product_name}")
    prompt_text = await create_system_prompt(
        product_name=req.product_name,
        product_url=req.product_url
    )
    return PromptResponse(system_prompt=prompt_text)

@app.get("/")
async def root():
    return {"message": "Dynamic Prompt Server is running. POST to /generate-prompt"}

# --- 7. Run the Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)