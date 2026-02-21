import os
import sys
import asyncio
import uvicorn 
import time 
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import pprint
from dotenv import load_dotenv

# --- Imports for Research Agent ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- NEW Imports for Design/BRD Agent ---
import requests
from fpdf import FPDF # <-- NEW IMPORT

load_dotenv()

_grok_key = os.getenv("GROQ_API_KEY")
if _grok_key:
    print("--- 🔐 GROQ_API_KEY loaded from environment/.env ---")
else:
    print("--- ⚠️  GROQ_API_KEY not found. Set GROQ_API_KEY in your environment or .env file. ---")

_tavily_key = os.getenv("TAVILY_API_KEY")
if _tavily_key:
    print("--- 🔐 TAVILY_API_KEY loaded from environment/.env ---")
else:
    print("--- ⚠️  TAVILY_API_KEY not found. Set TAVILY_API_KEY in your environment or .env file. ---")

_unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
if _unsplash_key:
    print("--- 🔐 UNSPLASH_ACCESS_KEY loaded from environment/.env ---")
else:
    print("--- ⚠️  UNSPLASH_ACCESS_KEY not found. Set UNSPLASH_ACCESS_KEY in your environment or .env file. ---")


llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
print(f"--- 🤖 Groq LLM Initialized (llama-3.1-8b-instant) ---") 

class EmailStep(BaseModel):
    """A single email in the nurture sequence"""
    subject: str = Field(description="The subject line of the email")
    body_markdown: str = Field(description="The markdown content of the email")
    send_delay_days: int = Field(description="Days to wait before sending (0=immediate)")

class SocialPost(BaseModel):
    """A single social media post"""
    platform: str = Field(description="e.g., 'LinkedIn', 'X (Twitter)'")
    content: str = Field(description="The text content of the post")
    image_prompt: str = Field(description="A simple search keyword for a stock photo (e.g., 'tech', 'code', 'team')")

class BrandKit(BaseModel):
    """The visual identity for the campaign"""
    logo_prompt: str = Field(description="A DALL-E prompt for a minimalist logo")
    color_palette: List[str] = Field(description="List of 5 hex color codes")
    font_pair: str = Field(description="e.g., 'Inter and Roboto'")

class CampaignState(BaseModel):
    """
    The main state object passed between all agents.
    """
    
    # --- 1. Filled by Planner_Agent ---
    initial_prompt: str = Field(description="The user's first natural language prompt")
    goal: Optional[str] = None
    topic: Optional[str] = None
    target_audience: Optional[str] = None
    source_docs_url: Optional[str] = None
    campaign_date: Optional[datetime] = None

    # --- 2. Filled by Research_Agent ---
    audience_persona: Optional[Dict[str, str]] = None
    core_messaging: Optional[Dict[str, str]] = None

    # --- 3. Filled by Content_Agent ---
    webinar_details: Optional[Dict[str, str]] = None # {title, abstract}
    webinar_image_prompt: Optional[str] = None # The prompt for the landing page banner
    blog_post: Optional[str] = None
    email_sequence: List[EmailStep] = []
    social_posts: List[SocialPost] = []

    # --- 4. Filled by Design_Agent ---
    brand_kit: Optional[BrandKit] = None
    generated_assets: Dict[str, str] = {} # e.g., {"logo_url": "...", "webinar_banner_url": "..."}

    # --- 5. Filled by Web_Agent ---
    landing_page_code: Optional[str] = None
    landing_page_url: Optional[str] = None
    
    # --- 6. Filled by BRD_Agent ---
    brd_url: Optional[str] = None # <-- NEW FIELD
    
    # --- 7. Filled by Ops_Agent ---
    automation_status: Dict[str, str] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        arbitrary_types_allowed = True

# --- 3. AGENT LOGIC (LCEL Chains) ---

# --- 3.1: PLANNER AGENT SCHEMA & CHAIN ---
class PlannerOutput(BaseModel):
    goal: str = Field(description="The primary objective, e.g., 'Launch a webinar', 'Promote a whitepaper'.")
    topic: str = Field(description="The main subject or product feature, e.g., 'Agentic-Fix'.")
    target_audience: str = Field(description="The specific user persona, e.g., 'VPs of Engineering'.")
    source_docs_url: Optional[str] = Field(description="The URL (e.g., Notion) containing the source content, if provided.")
    campaign_date: Optional[datetime] = Field(description="The target date for the campaign, in YYYY-MM-DD format. Infer from context. If not mentioned, leave as null.")

planner_parser = PydanticOutputParser(pydantic_object=PlannerOutput)
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert parsing assistant. You are part of an automated workflow and your "
            "output will be programmatically parsed. Your *only* job is to parse a user's unstructured "
            "campaign brief into a structured JSON object. "
            "Today's date is " + str(datetime.now().date()) +
            "\n\n{format_instructions}\n\n"
            "IMPORTANT: Your response MUST be ONLY the JSON object, with no other text, "
            "markdown, or commentary before or after the JSON. The JSON must be the only "
            "thing in your response."
        ),
        (
            "human", 
            "Parse the following campaign brief:\n\n{brief}"
        ),
    ]
).partial(format_instructions=planner_parser.get_format_instructions())
planner_chain = planner_prompt | llm | planner_parser
print("--- 📋 Planner Agent LCEL Chain Compiled ---")


# --- 3.2: RESEARCH AGENT SCHEMA & CHAIN (Simplified) ---
class ResearchOutput(BaseModel):
    audience_persona: Dict[str, str] = Field(description="A 3-key dictionary describing the target audience, with keys 'pain_point', 'motivation', and 'preferred_channel'.")
    core_messaging: Dict[str, str] = Field(description="A 3-key dictionary for the marketing strategy, with keys 'value_proposition', 'tone_of_voice', and 'call_to_action'.")

research_parser = PydanticOutputParser(pydantic_object=ResearchOutput)

def scrape_webpage(url: str) -> str:
    """Scrapes a single webpage and returns its content."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
    except Exception as e:
        print(f"--- ❌ ERROR scraping {url}: {e} ---")
        return f"Error scraping {url}: {e}"

tavily_tool = TavilySearch(max_results=3) 
research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class marketing strategist. Your job is to synthesize "
            "product information and audience research into a clear marketing strategy. "
            "Respond ONLY with the required JSON object, with no other text."
            "\n\n{format_instructions}"
        ),
        (
            "human",
            "--- PRODUCT CONTEXT ---\n"
            "{scraped_content}\n\n" 
            "--- AUDIENCE RESEARCH ---\n"
            "Topic: {topic}, Audience: {target_audience}\n"
            "Research Results:\n{search_results}\n\n"
            "--- SYNTHESIS ---"
            "Based on all this info, generate the **audience_persona** and **core_messaging**."
        ),
    ]
).partial(format_instructions=research_parser.get_format_instructions())
research_search_only_chain = (
    RunnablePassthrough.assign(
        scraped_content=lambda x: "No document provided.", # Default content
        search_results=lambda x: tavily_tool.invoke(f"common pain points for {x['target_audience']} related to {x['topic']}")
    )
    | research_prompt
    | llm
    | research_parser
)
print("--- 🧠 Research Agent LCEL Chain Compiled (Search-Only) ---")


# --- 3.3: CONTENT AGENT SCHEMA & CHAIN (MODIFIED) ---
class WebinarDetails(BaseModel):
    """Details for the campaign's webinar"""
    title: str = Field(description="The catchy, professional title of the webinar")
    abstract: str = Field(description="A 2-3 sentence abstract for the webinar landing page.")
class ContentAgentOutput(BaseModel):
    """The creative content for the campaign"""
    webinar_details: WebinarDetails
    social_posts: List[SocialPost] = Field(description="A list of 2 social media posts for the campaign (1 Instagram, 1 X/Twitter).")
    webinar_image_prompt: str = Field(description="A stock photo search query for the main webinar banner.")
content_parser = PydanticOutputParser(pydantic_object=ContentAgentOutput)
content_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class marketing copywriter... "
            "For the 'image_prompt', create a simple, effective search keyword for a stock photo... "
            "Respond ONLY with the required JSON object, with no other text."
            "\n\n{format_instructions}"
        ),
        (
            "human",
            "--- CAMPAIGN CONTEXT ---"
            "\nGoal: {goal}"
            "\nTopic: {topic}"
            "\nTarget Audience: {target_audience}"
            "\nAudience Persona: {persona}"
            "\nCore Messaging: {messaging}"
            "\n\n--- TASK ---"
            "\nGenerate the following content based on the context provided:"
            "\n1. Webinar Details: A catchy title and a 2-3 sentence abstract."
            "\n2. Social Posts: A list of *exactly 2* social media posts:"
            "\n   - One post for 'Instagram'. (Content should be visual-first, engaging, with emojis and hashtags)."
            "\n   - One post for 'X (Twitter)'. (Content should be short, punchy, and include a call-to-action)."
            "\n   - For each post, provide the platform, the full text content, and a simple stock photo search query for the 'image_prompt'."
            "\n3. Webinar Image Prompt: A simple stock photo search query for the main webinar banner (e.g., 'professional tech conference', 'abstract data')."
        ),
    ]
).partial(format_instructions=content_parser.get_format_instructions())
content_chain = content_prompt | llm | content_parser
print("--- ✍️  Content Agent LCEL Chain Compiled ---")


# --- 3.4: DESIGN AGENT (Using Unsplash) ---
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
UNSPLASH_HEADERS = {"Authorization": f"Client-ID {_unsplash_key}"}
def get_unsplash_image(search_query: str) -> str:
    """
    Calls the Unsplash API to get a stock photo and returns a URL.
    """
    print(f"--- 🎨 Querying Unsplash for: '{search_query}' ---")
    params = {
        "query": search_query,
        "per_page": 1,
        "orientation": "landscape"
    }
    
    try:
        response = requests.get(UNSPLASH_API_URL, headers=UNSPLASH_HEADERS, params=params, timeout=10)
        response.raise_for_status() 
        data = response.json()
        if data["results"]:
            image_url = data["results"][0]["urls"]["regular"]
            print(f"--- 🎨 Found image URL: {image_url[:50]}... ---")
            return image_url
        else:
            print(f"--- ⚠️ Unsplash found no results for '{search_query}', using placeholder. ---")
            return f"https://placehold.co/800x400/CCCCCC/FFFFFF?text=No+Image+For+{search_query.replace(' ', '+')}"
    except Exception as e:
        print(f"--- ❌ ERROR: Unsplash API failed: {e} ---")
        return "https://placehold.co/800x400/FF0000/FFFFFF?text=Error"


# --- 3.5: WEB AGENT (MODIFIED) ---
web_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class UI/UX designer and frontend developer... "
            "You have complete creative freedom... "
            "You MUST use inline CSS for all styling (inside a `<style>` tag in the `<head>`). "
            "Your response MUST be ONLY the complete HTML code..."
        ),
        (
            "human",
            "Please generate the HTML for a promotional landing page... "
            "\n\n--- CAMPAIGN STRATEGY ---"
            "\n- Topic: {topic}"
            "\n- Audience Persona: {audience_persona}"
            "\n- Core Messaging: {core_messaging}"
            "\n\n--- BRAND ASSETS ---"
            "\n- All Available Images: {generated_assets}"
            "\n\n--- REQUIREMENTS ---"
            "\n1. Create a beautiful, multi-section page including: a Hero (using the 'webinar_banner_url'), a 'Problem' section (based on pain_point), a 'Solution' section (introducing the topic), and a 'Who Is This For' section."
            "\n2. Use the *other* images (e.g., 'post_1_image_url') in the other sections or in a small gallery to make the page more visually appealing."
            "\n3. This is a promotional-only website. It must *not* have any buttons, 'Sign Up' forms, input fields, or 'mailto:' links."
        ),
    ]
)
web_agent_chain = web_agent_prompt | llm | StrOutputParser()
print("--- 🕸️  Web Agent LCEL Chain Compiled ---")


# --- 3.6: BRD AGENT (NEW) ---

# This prompt will generate the BRD as Markdown with proper business content
brd_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Senior Product Manager creating a professional Business Requirements Document (BRD). "
            "Generate a comprehensive, well-structured BRD in Markdown format. "
            "Include specific business metrics, success criteria, timelines, and resource requirements. "
            "Your response MUST be ONLY the Markdown text, starting with '# Business Requirements Document', with NO other commentary."
        ),
        (
            "human",
            "Create a detailed BRD for this product launch campaign:"
            "\n\n--- PRODUCT & CAMPAIGN INFO ---"
            "\n- Product/Topic: {topic}"
            "\n- Business Goal: {goal}"
            "\n- Target Audience: {target_audience}"
            "\n- Audience Pain Point: {pain_point}"
            "\n- Value Proposition: {value_proposition}"
            "\n- Tone of Voice: {tone}"
            "\n- Call to Action: {cta}"
            "\n\n--- SECTIONS TO INCLUDE (IN THIS ORDER) ---"
            "\n\n1. **Executive Summary**"
            "\n   - Brief overview of the product/campaign"
            "\n   - Key business objectives and expected ROI"
            "\n   - Success metrics (2-3 KPIs)"
            "\n\n2. **Project Scope**"
            "\n   - What's included in this initiative"
            "\n   - Out-of-scope items"
            "\n   - Timeline/milestones"
            "\n\n3. **Market & Competitive Analysis**"
            "\n   - Current market size and growth opportunity"
            "\n   - Key competitors and differentiation"
            "\n   - Market trends relevant to this product"
            "\n\n4. **Target Audience & User Personas**"
            "\n   - Primary persona with demographics, goals, and pain points"
            "\n   - Secondary personas if applicable"
            "\n   - Audience segments and sizing"
            "\n\n5. **Business Requirements**"
            "\n   - Functional requirements (what the product must do)"
            "\n   - Non-functional requirements (performance, security, scalability)"
            "\n   - Regulatory or compliance requirements if any"
            "\n\n6. **User Stories & Use Cases**"
            "\n   - 4-6 user stories in 'As a [user], I want to [action], so that [benefit]' format"
            "\n   - Acceptance criteria for each story"
            "\n\n7. **Success Criteria & KPIs**"
            "\n   - Specific, measurable success metrics"
            "\n   - Target numbers for engagement, conversion, or adoption"
            "\n   - Timeline for measurement (30, 60, 90 days)"
            "\n\n8. **Resource Requirements**"
            "\n   - Team roles and responsibilities"
            "\n   - Budget estimate (if applicable)"
            "\n   - Dependencies and risks"
            "\n\n9. **Implementation Timeline**"
            "\n   - Phase-based rollout plan"
            "\n   - Key milestones and deliverables"
            "\n   - Go-live date and post-launch support plan"
            "\n\n10. **Risk Assessment & Mitigation**"
            "\n    - Key risks (technical, market, resource)"
            "\n    - Mitigation strategies"
            "\n    - Contingency plans"
        ),
    ]
)
brd_agent_chain = brd_agent_prompt | llm | StrOutputParser()
print("--- 📄 BRD Agent LCEL Chain Compiled ---")


# --- 4. AGENT "WORKSTATIONS" (The Nodes) ---

# --- NEW PDF HELPER FUNCTION WITH PROPER MARKDOWN PARSING ---
def save_markdown_as_pdf(markdown_text: str, filename: str) -> str:
    """
    Converts a Markdown string to a professionally formatted PDF file using fpdf2.
    Properly parses and formats Markdown with automatic text wrapping to prevent overflow.
    """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Set proper margins
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.set_top_margin(15)
        
        # Available width for text
        cell_width = 180  # A4 width minus margins
        
        # Parse and format Markdown
        lines = markdown_text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip completely empty lines
            if not line_stripped:
                pdf.ln(1)
                continue
            
            try:
                # H1 Headers (# ...)
                if line_stripped.startswith('# '):
                    pdf.set_font("Helvetica", 'B', size=16)
                    pdf.set_text_color(0, 0, 0)
                    text = line_stripped[2:].strip()
                    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                
                # H2 Headers (## ...)
                elif line_stripped.startswith('## '):
                    pdf.set_font("Helvetica", 'B', size=13)
                    pdf.set_text_color(40, 80, 160)  # Dark blue
                    text = line_stripped[3:].strip()
                    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(1)
                
                # H3 Headers (### ...)
                elif line_stripped.startswith('### '):
                    pdf.set_font("Helvetica", 'B', size=11)
                    pdf.set_text_color(80, 80, 80)  # Gray
                    text = line_stripped[4:].strip()
                    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(0.5)
                
                # Bullet points (- or * followed by space and text)
                elif (line_stripped.startswith('- ') or line_stripped.startswith('* ')) and len(line_stripped) > 2:
                    pdf.set_font("Helvetica", size=10)
                    pdf.set_text_color(0, 0, 0)
                    # Extract bullet text
                    bullet_text = line_stripped[2:].strip() if line_stripped.startswith('- ') else line_stripped[2:].strip()
                    
                    # Use dash instead of bullet character for compatibility
                    text_with_bullet = '- ' + bullet_text
                    # Write text with wrapping - multi_cell handles it automatically
                    pdf.multi_cell(cell_width, 5.5, text_with_bullet, border=0)
                
                # Numbered lists (1. 2. 3. etc)
                elif len(line_stripped) > 2 and line_stripped[0].isdigit() and line_stripped[1:3].strip().startswith('.'):
                    pdf.set_font("Helvetica", size=10)
                    pdf.set_text_color(0, 0, 0)
                    
                    # Extract number and text
                    dot_index = line_stripped.find('.')
                    if dot_index > 0:
                        num_part = line_stripped[:dot_index+1]
                        text_part = line_stripped[dot_index+1:].strip()
                        
                        current_x = pdf.get_x()
                        current_y = pdf.get_y()
                        
                        # Write number
                        pdf.set_x(current_x + 3)
                        pdf.cell(5, 6, num_part, ln=False)
                        pdf.set_x(current_x + 10)
                        
                        # Write text with wrapping
                        pdf.multi_cell(cell_width - 10, 5.5, text_part, border=0)
                    else:
                        # Fallback if parsing fails
                        pdf.multi_cell(cell_width, 5.5, line_stripped, border=0)
                
                # Bold text (**...**) or (__...__) 
                elif ('**' in line_stripped or '__' in line_stripped) and len(line_stripped) > 4:
                    pdf.set_font("Helvetica", 'B', size=10)
                    pdf.set_text_color(0, 0, 0)
                    clean_line = line_stripped.replace('**', '').replace('__', '').replace('_', '')
                    pdf.multi_cell(cell_width, 5.5, clean_line, border=0)
                    pdf.set_font("Helvetica", size=10)
                
                # Regular text (body paragraphs) - most common case
                else:
                    pdf.set_font("Helvetica", size=10)
                    pdf.set_text_color(0, 0, 0)
                    # multi_cell automatically wraps text based on cell_width
                    pdf.multi_cell(cell_width, 5.5, line_stripped, border=0)
                    
            except Exception as line_err:
                print(f"--- ⚠️ Warning: Could not render line: {line_stripped[:40]}... Error: {line_err} ---")
                try:
                    pdf.set_font("Helvetica", size=9)
                    pdf.multi_cell(cell_width, 4, line_stripped, border=0)
                except:
                    pass
        
        pdf.output(filename)
        print(f"--- 📄 PDF saved as: {filename} (properly formatted Markdown) ---")
        return filename
    except Exception as e:
        print(f"--- ❌ ERROR saving PDF: {e} ---")
        import traceback
        traceback.print_exc()
        return "error_saving_pdf.pdf"

def planner_agent_node(state: CampaignState) -> dict:
    print("--- 1. 📋 Calling Planner Agent (REAL) ---")
    brief = state.initial_prompt
    try:
        planner_output: PlannerOutput = planner_chain.invoke({"brief": brief})
        return planner_output.model_dump()
    except Exception as e:
        print(f"--- ❌ ERROR in Planner Agent: {e} ---")
        return {}

def research_agent_node(state: CampaignState) -> dict:
    print("--- 2. 🧠 Calling Research Agent (REAL) ---")
    
    inputs = {
        "topic": state.topic,
        "target_audience": state.target_audience,
    }
    
    try:
        if state.source_docs_url:
            print(f"--- ⚠️ source_docs_url provided, but IGNORING IT to avoid token limits. ---")
        print("--- 🔎 Running search-only research chain... ---")
        research_output: ResearchOutput = research_search_only_chain.invoke(inputs)
        return research_output.model_dump()
    except Exception as e:
        print(f"--- ❌ ERROR in Research Agent: {e} ---")
        pprint.pprint(e) 
        return {} 

def content_agent_node(state: CampaignState) -> dict:
    print("--- 3. ✍️ Calling Content Agent (REAL) ---")
    try:
        inputs = {
            "goal": state.goal,
            "topic": state.topic,
            "target_audience": state.target_audience,
            "persona": state.audience_persona,
            "messaging": state.core_messaging,
        }
        content_output: ContentAgentOutput = content_chain.invoke(inputs)
        return content_output.model_dump()
    except Exception as e:
        print(f"--- ❌ ERROR in Content Agent: {e} ---")
        pprint.pprint(e)
        return {}

def design_agent_node(state: CampaignState) -> dict:
    print("--- 4. 🎨 Calling Design Agent (REAL) ---")
    
    mock_brand_kit = BrandKit(
        logo_prompt=f"A minimalist, tech-inspired logo for {state.topic}",
        color_palette=["#0A0A0A", "#FFFFFF", "#4F46E5", "#FBBF24", "#10B981"], # Dark, White, Blue, Yellow, Green
        font_pair="Inter" # Using a single modern font
    )
    
    generated_assets = {}
    
    print("--- 🎨 Generating Webinar Banner... ---")
    generated_assets["webinar_banner_url"] = get_unsplash_image(state.webinar_image_prompt)
    
    for i, post in enumerate(state.social_posts):
        print(f"--- 🎨 Generating image for social post {i+1} ({post.platform})... ---")
        image_url = get_unsplash_image(post.image_prompt)
        generated_assets[f"post_{i+1}_image_url"] = image_url

    print("--- ✅ Design Agent finished ---")
    
    return {
        "brand_kit": mock_brand_kit,
        "generated_assets": generated_assets
    }

def web_agent_node(state: CampaignState) -> dict:
    print("--- 5. 🕸️ Calling Web Agent (REAL) ---")
    
    try:
        inputs = {
            "topic": state.topic,
            "audience_persona": state.audience_persona,
            "core_messaging": state.core_messaging,
            "generated_assets": state.generated_assets
        }
        
        print("--- 🕸️ Generating HTML code based on research (full autonomy)... ---")
        html_code = web_agent_chain.invoke(inputs)
        
        return {
            "landing_page_code": html_code,
            "landing_page_url": "campaign_preview.html"
        }

    except Exception as e:
        print(f"--- ❌ ERROR in Web Agent: {e} ---")
        pprint.pprint(e)
        return {}


# --- NEW AGENT NODE ---
def brd_agent_node(state: CampaignState) -> dict:
    print("--- 6. 📄 Calling BRD Agent (REAL) ---")
    try:
        # Extract individual fields from audience_persona and core_messaging
        persona = state.audience_persona or {}
        messaging = state.core_messaging or {}
        
        inputs = {
            "topic": state.topic or "Product",
            "goal": state.goal or "Launch and market the product",
            "target_audience": state.target_audience or "Target users",
            "pain_point": persona.get("pain_point", "Unmet market need"),
            "value_proposition": messaging.get("value_proposition", "Innovative solution"),
            "tone": messaging.get("tone_of_voice", "Professional and engaging"),
            "cta": messaging.get("call_to_action", "Learn more"),
        }
        
        print("--- 📄 Generating BRD Markdown... ---")
        brd_markdown = brd_agent_chain.invoke(inputs)
        
        # Save the PDF
        # Create a 'campaign_outputs' directory if it doesn't exist
        output_dir = "campaign_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a safe filename from the topic
        safe_filename = state.topic.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f"{safe_filename}_brd.pdf"
        file_path = os.path.join(output_dir, filename)
        
        pdf_path = save_markdown_as_pdf(brd_markdown, file_path)
        
        # Return the download URL
        return {"brd_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

    except Exception as e:
        print(f"--- ❌ ERROR in BRD Agent: {e} ---")
        pprint.pprint(e)
        return {}


def ops_agent_node(state: CampaignState) -> dict:
    print("--- 7. ⚙️ Calling Ops Agent (MOCK) ---") # Step 7
    mock_ops_data = {"automation_status": {"mailchimp_automation_id": "abc-123", "scheduled_post_count": "5"}}
    return mock_ops_data


# --- 5. LANGGRAPH "FACTORY FLOOR" (The Graph) ---

graph_builder = StateGraph(CampaignState)

# Add all nodes
graph_builder.add_node("planner_agent", planner_agent_node)
graph_builder.add_node("research_agent", research_agent_node)
graph_builder.add_node("content_agent", content_agent_node)
graph_builder.add_node("design_agent", design_agent_node)
graph_builder.add_node("web_agent", web_agent_node)
graph_builder.add_node("brd_agent", brd_agent_node) # <-- NEW
graph_builder.add_node("ops_agent", ops_agent_node)

# Add all edges (sequential flow)
graph_builder.set_entry_point("planner_agent")
graph_builder.add_edge("planner_agent", "research_agent")
graph_builder.add_edge("research_agent", "content_agent")
graph_builder.add_edge("content_agent", "design_agent")
graph_builder.add_edge("design_agent", "web_agent")
graph_builder.add_edge("web_agent", "brd_agent") # <-- NEW
graph_builder.add_edge("brd_agent", "ops_agent") # <-- NEW
graph_builder.add_edge("ops_agent", END)


# Compile the graph
print("--- 🏭 Compiling AI Campaign Foundry Graph (Sequential) ---")
sys.setrecursionlimit(200) 
foundry_app = graph_builder.compile()
print("--- ✅ Foundry Graph Compiled ---")


# --- 6. FASTAPI SERVER (The Streaming Endpoint) ---

app = FastAPI()

class StreamRequest(BaseModel):
    initial_prompt: str

@app.websocket("/ws_stream_campaign")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- 🔌 WebSocket Connection Accepted ---")
    try:
        json_data = await websocket.receive_json()
        request_data = StreamRequest(**json_data)
        
        initial_input = {"initial_prompt": request_data.initial_prompt}
        
        current_state_dict = initial_input.copy()
        
        print(f"--- 🚀 Received input, starting stream... ---")
        
        async for s in foundry_app.astream(initial_input):
            node_that_ran = list(s.keys())[0]
            state_snapshot_diff = s[node_that_ran] # This is a dict
            
            if state_snapshot_diff:
                for key, value in state_snapshot_diff.items():
                    if isinstance(value, list) and key in current_state_dict:
                        current_state_dict[key].extend(value)
                    elif isinstance(value, dict) and key in current_state_dict:
                        current_state_dict[key].update(value)
                    else:
                        current_state_dict[key] = value
            
            state_json = CampaignState.model_validate(current_state_dict).model_dump_json(indent=2)
            
            await websocket.send_json({
                "event": "step",
                "node": node_that_ran,
                "data": state_json
            })
            
        await websocket.send_json({"event": "done"})
        print("--- ✨ Stream Complete ---")
        
        await websocket.close()

    except WebSocketDisconnect:
        print("--- 🔌 WebSocket Disconnected ---")
    
    except Exception as e:
        print(f"--- ❌ WebSocket Error: {e} ---")
        try:
            await websocket.send_json({"event": "error", "data": str(e)})
        except Exception:
            pass 
        
        try:
            await websocket.close()
        except Exception:
            pass 
    
    finally:
        pass


@app.get("/download_pdf/{filename}")
async def download_pdf(filename: str):
    """
    Serves PDF files from the campaign_outputs directory.
    Prevents directory traversal attacks by sanitizing filename.
    """
    # Sanitize filename to prevent directory traversal
    filename = filename.replace("../", "").replace("..\\", "").replace("..", "")
    
    file_path = os.path.join("campaign_outputs", filename)
    
    # Convert to absolute path for verification
    abs_file_path = os.path.abspath(file_path)
    abs_output_dir = os.path.abspath("campaign_outputs")
    
    print(f"--- 📥 Download request for: {filename} ---")
    print(f"--- 📍 File path: {abs_file_path} ---")
    print(f"--- 📍 Output dir: {abs_output_dir} ---")
    
    # Verify the file exists
    if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
        print(f"--- ❌ PDF file not found: {abs_file_path} ---")
        return {"error": "PDF file not found"}
    
    # Verify it's inside campaign_outputs (not outside)
    if not abs_file_path.startswith(abs_output_dir):
        print(f"--- ❌ Attempted directory traversal: {abs_file_path} ---")
        return {"error": "Invalid file path"}
    
    print(f"--- ✅ Serving PDF: {abs_file_path} ---")
    
    return FileResponse(
        abs_file_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/")
async def root():
    return {"message": "AI Campaign Foundry Server is running. Connect via WebSocket."}

if __name__ == "__main__":
    print("--- 🚀 Starting FastAPI server on http://localhost:8000 ---")
    uvicorn.run(app, host="localhost", port=8000)