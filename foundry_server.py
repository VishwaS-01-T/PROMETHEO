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
from fastapi.middleware.cors import CORSMiddleware
import pprint
from dotenv import load_dotenv

# --- Imports for Research Agent ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- NEW Imports for Design/BRD Agent ---
import requests
from fpdf import FPDF, XPos, YPos  # <-- Import with position enums

load_dotenv()

# --- Auto-rotate API keys to avoid context limit exhaustion ---
try:
    from rotate_keys import rotate_api_keys
    rotate_api_keys()
except ImportError:
    print("--- ⚠️  rotate_keys module not found, skipping key rotation ---")
except Exception as e:
    print(f"--- ⚠️  Key rotation error: {e} ---")

_grok_key0 = os.getenv("GROQ_API_KEY0")
if _grok_key0:
    print("--- 🔐 GROQ_API_KEY0 loaded — jurisdiction, research ---")
else:
    print("--- ⚠️  GROQ_API_KEY0 not found. Set GROQ_API_KEY0 in your environment or .env file. ---")

_grok_key1 = os.getenv("GROQ_API_KEY1")
if _grok_key1:
    print("--- 🔐 GROQ_API_KEY1 loaded — planner, strategy, brd ---")
else:
    print("--- ⚠️  GROQ_API_KEY1 not found. Set GROQ_API_KEY1 in your environment or .env file. ---")

_grok_key2 = os.getenv("GROQ_API_KEY2")
if _grok_key2:
    print("--- 🔐 GROQ_API_KEY2 loaded — content, design, web ---")
else:
    print("--- ⚠️  GROQ_API_KEY2 not found. Set GROQ_API_KEY2 in your environment or .env file. ---")

_grok_key3 = os.getenv("GROQ_API_KEY3")
if _grok_key3:
    print("--- 🔐 GROQ_API_KEY3 loaded — strategy, breakdown, brd ---")
else:
    print("--- ⚠️  GROQ_API_KEY3 not found. Set GROQ_API_KEY3 in your environment or .env file. ---")

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


llm0 = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, api_key=_grok_key0)
print(f"--- 🤖 Groq LLM (Key 0) Initialized — jurisdiction, research ---")

llm1 = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, api_key=_grok_key1)
print(f"--- 🤖 Groq LLM (Key 1) Initialized — planner ---")

llm2 = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, api_key=_grok_key2)
print(f"--- 🤖 Groq LLM (Key 2) Initialized — content, design, web ---")

llm3 = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, api_key=_grok_key3)
print(f"--- 🤖 Groq LLM (Key 3) Initialized — strategy, breakdown, brd ---") 

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
    company_name: Optional[str] = None
    source_docs_url: Optional[str] = None
    campaign_date: Optional[datetime] = None
    location: Optional[str] = None

    # --- 2. Filled by Research_Agent ---
    audience_persona: Optional[Dict[str, str]] = None
    core_messaging: Optional[Dict[str, str]] = None
    jurisdiction_info: Optional[Dict[str, str]] = None  # {department_name, department_url, jurisdiction_type}
    registration_procedure: Optional[List[str]] = None  # Step-by-step procedure from the department website
    required_documents: Optional[List[Dict[str, str]]] = None  # Location-specific legal/regulatory documents

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
    brd_url: Optional[str] = None
    
    # --- 7. Filled by Strategy_Agent (MODIFIED) ---
    strategy_markdown: Optional[str] = None # <-- CHANGED
    
    # --- 9. Filled by Validation Agent ---
    raw_govt_content: Optional[str] = None          # raw text scraped from govt website (always preserved)
    validation_rounds: int = 0                       # how many re-validation loops we've done
    step_confidence: Dict[str, float] = {}           # {"0": 0.9, "1": 0.4, ...} keyed by step index str
    document_confidence: Dict[str, float] = {}       # {"Certificate of Incorporation": 0.87, ...}
    validation_mismatches: List[str] = []            # mismatch/gap descriptions fed back into research
    govt_fallback_only: bool = False                 # True → frontend shows ONLY raw govt scrape
    overall_confidence: float = 1.0                  # aggregate 0.0–1.0 confidence score
    
    # --- 8. Filled by Ops_Agent ---
    automation_status: Dict[str, Any] = {}  # Changed from Dict[str, str] to Dict[str, Any] to support complex data
    
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
    company_name: Optional[str] = Field(description="The company/brand name to display on generated assets. Infer from brief if present; otherwise null.")
    source_docs_url: Optional[str] = Field(description="The URL (e.g., Notion) containing the source content, if provided.")
    campaign_date: Optional[datetime] = Field(description="The target date for the campaign, in YYYY-MM-DD format. Infer from context. If not mentioned, leave as null.")
    location: Optional[str] = Field(default=None, description="The country or region for the campaign. Infer from brief if present; otherwise null.")

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
planner_chain = planner_prompt | llm1 | planner_parser
print("--- 📋 Planner Agent LCEL Chain Compiled ---")


# --- 3.2: RESEARCH AGENT — MULTI-STEP (Jurisdiction → Scrape → Documents) ---

# --- Step 1 Models: Jurisdiction Discovery ---
class JurisdictionInfo(BaseModel):
    """The specific government department/authority that oversees startup registration."""
    department_name: str = Field(description="The exact official name of the government department or regulatory body.")
    department_url: str = Field(description="The official website URL of this department (must be a real, valid URL).")
    jurisdiction_type: str = Field(description="The type of jurisdiction, e.g., 'Company Registration', 'Business Licensing', 'Startup Registration'.")

jurisdiction_parser = PydanticOutputParser(pydantic_object=JurisdictionInfo)
jurisdiction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in international business law and startup regulations. "
            "Given a country and a startup topic/industry, identify the EXACT government department "
            "or regulatory body that handles company registration and startup licensing in that country. "
            "Return the department's official name and its real website URL. "
            "Respond ONLY with the required JSON object, with no other text."
            "\n\n{format_instructions}"
        ),
        (
            "human",
            "Country: {location}\n"
            "Startup Industry/Topic: {topic}\n"
            "Company Name: {company_name}\n\n"
            "Web search results about the regulatory body:\n{jurisdiction_search}\n\n"
            "Identify the specific government department that handles startup/company registration "
            "for this type of business in {location}. Provide its exact official name and website URL."
        ),
    ]
).partial(format_instructions=jurisdiction_parser.get_format_instructions())
print("--- 🏛️  Jurisdiction Discovery Chain Compiled ---")


# --- Step 2: Department Website Procedure Extraction ---
class ProcedureOutput(BaseModel):
    """Registration procedure extracted from the department website."""
    registration_steps: List[str] = Field(description="An ordered list of step-by-step instructions for registering a startup, as found on the department's website.")

procedure_parser = PydanticOutputParser(pydantic_object=ProcedureOutput)
procedure_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at reading government websites and extracting registration procedures. "
            "Given information scraped from a government department's website, extract the exact "
            "step-by-step procedure to register/apply for a startup or company in that jurisdiction. "
            "Each step should be a clear, actionable instruction. "
            "Respond ONLY with the required JSON object, with no other text."
            "\n\n{format_instructions}"
        ),
        (
            "human",
            "Department: {department_name} ({department_url})\n"
            "Country: {location}\n"
            "Startup Topic: {topic}\n\n"
            "--- WEBSITE CONTENT ---\n{website_content}\n\n"
            "--- ADDITIONAL SEARCH RESULTS ---\n{procedure_search}\n\n"
            "Extract the step-by-step registration/application procedure for this type of startup. "
            "Include all necessary steps: name reservation, document preparation, filing, fees, timelines, etc."
        ),
    ]
).partial(format_instructions=procedure_parser.get_format_instructions())
print("--- 📋 Procedure Extraction Chain Compiled ---")


# --- Step 3 Models: Full Research Output with Documents ---
class RequiredDocument(BaseModel):
    """A regulatory or legal document required for the campaign in the target country."""
    document_name: str = Field(description="The official name of the required document or permit.")
    issuing_authority: str = Field(description="The government body or authority that issues this document.")
    purpose: str = Field(description="Why this document is needed for the campaign.")
    deadline_note: str = Field(description="When this document should be obtained relative to the campaign date (e.g., '30 days before launch').")

class ResearchOutput(BaseModel):
    audience_persona: Dict[str, str] = Field(description="A 3-key dictionary describing the target audience, with keys 'pain_point', 'motivation', and 'preferred_channel'.")
    core_messaging: Dict[str, str] = Field(description="A 3-key dictionary for the marketing strategy, with keys 'value_proposition', 'tone_of_voice', and 'call_to_action'.")
    required_documents: List[RequiredDocument] = Field(default_factory=list, description="A list of regulatory, legal, or compliance documents required to launch this startup in the specified country by the campaign date. Derive these from the registration procedure and department website. If no location is provided, return an empty list.")

research_parser = PydanticOutputParser(pydantic_object=ResearchOutput)
tavily_tool = TavilySearch(max_results=5)
research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class marketing strategist with expertise in international compliance. "
            "Your job is to synthesize product information and audience research into a clear marketing strategy, "
            "AND to generate the exact list of required documents based on the registration procedure "
            "and regulatory research for the target country. "
            "Today's date is " + str(datetime.now().date()) + ". "
            "Factor in any recent regulatory changes or upcoming deadlines relative to the campaign date. "
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
            "--- JURISDICTION & PROCEDURE ---\n"
            "Government Department: {department_name} ({department_url})\n"
            "Registration Procedure:\n{registration_procedure}\n\n"
            "--- CAMPAIGN LOCATION & DATE ---\n"
            "Location/Country: {location}\n"
            "Campaign Launch Date: {campaign_date}\n"
            "Recent Regulatory News:\n{regulatory_news}\n\n"
            "--- SYNTHESIS ---\n"
            "Based on all this info:\n"
            "1. Generate the **audience_persona** and **core_messaging**.\n"
            "2. Generate **required_documents**: List ALL legal, regulatory, and compliance documents "
            "required to launch this startup in {location} by {campaign_date}. "
            "Base these on the registration procedure steps above. "
            "Include company registration forms, tax registrations, data protection filings, "
            "industry-specific licenses, advertising approvals, etc. "
            "For each document, specify the exact issuing authority from the procedure. "
            "If no location is provided, return an empty list for required_documents."
        ),
    ]
).partial(format_instructions=research_parser.get_format_instructions())
print("--- 🧠 Research Agent LCEL Chain Compiled (Multi-Step) ---")


# --- 3.2.5: VALIDATION AGENT MODEL & CHAIN ---

MAX_VALIDATION_ROUNDS = 2         # hard ceiling on re-validation loops
CONFIDENCE_THRESHOLD  = 0.68      # below this we re-run research if rounds allow

class ValidationOutput(BaseModel):
    """Structured output from the validation agent."""
    is_validated: bool = Field(
        description="True if the generated docs/steps closely match what the govt website specifies."
    )
    overall_confidence: float = Field(
        description="Aggregate confidence 0.0-1.0 across all steps and documents."
    )
    step_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Confidence score per registration step, keyed by step index as a string "
            "('0', '1', …). Score is 0.0-1.0."
        )
    )
    document_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Confidence score per required document, keyed by document_name. Score is 0.0-1.0."
        )
    )
    mismatches: List[str] = Field(
        default_factory=list,
        description=(
            "List of specific discrepancies found: wrong document names, wrong issuing authorities, "
            "missing steps, extra/invented documents not mentioned by the govt website, etc."
        )
    )
    missing_docs: List[str] = Field(
        default_factory=list,
        description="Document names that appear on the govt site but are absent from required_documents."
    )
    missing_steps: List[str] = Field(
        default_factory=list,
        description="Registration steps found in web research that are absent from registration_procedure."
    )

validation_parser = PydanticOutputParser(pydantic_object=ValidationOutput)

validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a meticulous compliance auditor specialising in startup registration procedures. "
        "You are given:\n"
        "  A) The raw text scraped directly from the government department's website — treat this as GROUND TRUTH.\n"
        "  B) A structured list of registration steps extracted by an AI from that scrape.\n"
        "  C) A list of required documents generated by an AI research agent.\n"
        "  D) Independent web-search results that cross-reference the above.\n\n"
        "Your job:\n"
        "  1. Compare B and C against A and D.\n"
        "  2. Assign a confidence score (0.0-1.0) to EACH step and EACH document.\n"
        "     – 1.0 = explicitly confirmed on govt website or multiple authoritative sources.\n"
        "     – 0.7 = mentioned in web search but not explicitly on govt site.\n"
        "     – 0.4 = plausible but not verified.\n"
        "     – 0.1 = contradicted or clearly hallucinated.\n"
        "  3. List ALL mismatches: wrong names, wrong authorities, extra invented docs, wrong order.\n"
        "  4. List documents present on the govt site but missing from the AI list.\n"
        "  5. Set is_validated=True ONLY if overall_confidence >= 0.75 and no critical mismatches.\n\n"
        "CRITICAL: The raw govt website content in section A is the single source of truth. "
        "Never trust AI-generated content over what the govt site explicitly states.\n\n"
        "{format_instructions}"
    ),
    (
        "human",
        "=== A) RAW GOVERNMENT WEBSITE CONTENT (SOURCE OF TRUTH) ===\n{raw_govt_content}\n\n"
        "=== B) AI-EXTRACTED REGISTRATION STEPS ===\n{registration_steps}\n\n"
        "=== C) AI-GENERATED REQUIRED DOCUMENTS ===\n{required_documents}\n\n"
        "=== D) INDEPENDENT WEB-SEARCH CROSS-REFERENCE ===\n{web_search_results}\n\n"
        "Country/Jurisdiction: {location}\n"
        "Startup Type: {topic}\n\n"
        "Audit all items. Return ONLY the JSON validation report."
    )
]).partial(format_instructions=validation_parser.get_format_instructions())

validation_chain = validation_prompt | llm0 | validation_parser
print("--- 🔍 Validation Agent Chain Compiled ---")


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
            "\n2. Social Posts: A list of *exactly 2* social media posts..."
            "\n3. Webinar Image Prompt: A simple stock photo search query for the main webinar banner..."
        ),
    ]
).partial(format_instructions=content_parser.get_format_instructions())
content_chain = content_prompt | llm2 | content_parser
print("--- ✍️  Content Agent LCEL Chain Compiled ---")


# --- 3.4: DESIGN AGENT (Using Unsplash) ---
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
UNSPLASH_HEADERS = {"Authorization": f"Client-ID {_unsplash_key}"}
def get_unsplash_image(search_query: str) -> str:
    print(f"--- 🎨 Querying Unsplash for: '{search_query}' ---")
    params = {"query": search_query, "per_page": 1, "orientation": "landscape"}
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
        return "https_://placehold.co/800x400/FF0000/FFFFFF?text=Error"


# --- 3.5: WEB AGENT (MODIFIED) ---
# We hard-code the HTML boilerplate (doctype/head/sticky navbar/footer) to guarantee consistency,
# and let the LLM generate only the 3 in-page sections.
web_sections_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Output ONLY raw HTML: three <section> tags with ids home, about, contact. "
            "No <html>/<head>/<body>/<style> wrappers. No buttons/forms/inputs. No commentary."
        ),
        (
            "human",
            "Topic: {topic}\nCompany: {company_name}\nPersona: {audience_persona}\nMessaging: {core_messaging}\nImages: {generated_assets}\n\n"
            "Generate 3 sections:\n"
            "<section id=\"home\">Hero with webinar_banner_url image + value proposition</section>\n"
            "<section id=\"about\">Problem/Solution based on pain_point</section>\n"
            "<section id=\"contact\">Who is this for (descriptive only)</section>"
        ),
    ]
)

web_sections_chain = web_sections_prompt | llm2 | StrOutputParser()
print("--- 🕸️  Web Agent LCEL Chain Compiled (Sections + Hardcoded Boilerplate) ---")


def _extract_body_like_html(raw: str) -> str:
    """Best-effort extraction if an LLM accidentally returns full HTML."""
    if not raw:
        return ""

    # Strip markdown code fences the LLM might wrap output in
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    raw = text.strip()

    lower = raw.lower()
    if (
        "<section" in lower
        and "id=\"home\"" in lower
        and "id=\"about\"" in lower
        and "id=\"contact\"" in lower
    ):
        return raw.strip()

    # Try to pull content inside <body>...</body>
    body_start = lower.find("<body")
    if body_start != -1:
        body_tag_end = lower.find(">", body_start)
        body_end = lower.rfind("</body>")
        if body_tag_end != -1 and body_end != -1 and body_end > body_tag_end:
            return raw[body_tag_end + 1:body_end].strip()

    return raw.strip()


def build_landing_page_html(*, company_name: str, sections_html: str) -> str:
    year = datetime.now().year
    safe_company = (company_name or "Company").strip() or "Company"
    footer_text = f"© {safe_company}.{year}.generated with PROMETHEON."

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{safe_company}</title>
    <style>
        :root {{
            --bg: #0b1020;
            --panel: rgba(255,255,255,0.06);
            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.70);
            --border: rgba(255,255,255,0.12);
            --accent: #8b5cf6;
            --accent2: #ec4899;
            --max: 1040px;
        }}
        * {{ box-sizing: border-box; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            background: radial-gradient(1200px 700px at 20% 0%, rgba(139,92,246,0.28), transparent 60%),
                                    radial-gradient(900px 600px at 90% 10%, rgba(236,72,153,0.20), transparent 55%),
                                    var(--bg);
            color: var(--text);
            line-height: 1.55;
        }}
        a {{ color: inherit; text-decoration: none; }}
        .container {{ max-width: var(--max); margin: 0 auto; padding: 0 20px; }}

        /* Sticky navbar */
        .nav {{
            position: sticky;
            top: 0;
            z-index: 50;
            backdrop-filter: blur(10px);
            background: rgba(11,16,32,0.72);
            border-bottom: 1px solid var(--border);
        }}
        .nav-inner {{ display: flex; align-items: center; justify-content: space-between; height: 64px; }}
        .brand {{ font-weight: 800; letter-spacing: 0.2px; }}
        .links {{ display: flex; gap: 16px; }}
        .links a {{
            padding: 8px 10px;
            border-radius: 10px;
            color: var(--muted);
        }}
        .links a:hover {{ background: rgba(255,255,255,0.06); color: var(--text); }}

        /* Sections */
        main {{ padding: 24px 0 44px; }}
        section {{
            scroll-margin-top: 84px;
            margin: 18px 0;
            padding: 28px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--panel);
            overflow: hidden;
        }}
        h1,h2,h3 {{ margin: 0 0 12px; line-height: 1.15; }}
        p {{ margin: 0 0 12px; color: var(--muted); }}
        img {{ max-width: 100%; border-radius: 14px; border: 1px solid var(--border); display: block; }}

        /* Footer (MUST be only the required line) */
        .footer {{
            border-top: 1px solid var(--border);
            color: rgba(255,255,255,0.78);
            text-align: center;
            padding: 18px 12px;
        }}
    </style>
</head>
<body>
    <header class=\"nav\">
        <div class=\"container nav-inner\">
            <div class=\"brand\">{safe_company}</div>
            <nav class=\"links\" aria-label=\"Primary\">
                <a href=\"#home\">Home</a>
                <a href=\"#contact\">Contact</a>
                <a href=\"#about\">About</a>
            </nav>
        </div>
    </header>

    <main class=\"container\">
        {sections_html}
    </main>

    <footer class=\"footer\">{footer_text}</footer>
</body>
</html>"""


# --- 3.6: BRD AGENT (NEW) ---
# BRD Agent: Uses Key 3 to generate full Business Requirements Document
brd_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Senior Product Manager specializing in business requirements. Your job is to generate a complete, "
            "well-structured Business Requirements Document (BRD) based on the strategy markdown. "
            "The document must be detailed, professional, and formatted as clean Markdown. "
            "Use clear headings (# and ##), bullet points, and numbered lists. "
            "Your response MUST be ONLY the Markdown text, starting with '# Business Requirements Document'."
        ),
        (
            "human",
            "Based on the following strategic approach, please generate a comprehensive Business Requirements Document in Markdown format:"
            "\n\n--- STRATEGIC APPROACH ---\n{strategy_markdown}"
            "\n\n--- BRD STRUCTURE (to follow) ---\n"
            "# Business Requirements Document\n\n"
            "## Executive Summary\n"
            "(Brief overview of the project/product)\n\n"
            "## 1. Project Overview\n"
            "### 1.1 Objectives\n"
            "### 1.2 Scope\n\n"
            "## 2. Business Requirements\n"
            "### 2.1 Functional Requirements\n"
            "### 2.2 Non-Functional Requirements\n\n"
            "## 3. Success Metrics & KPIs\n"
            "## 4. Timeline & Milestones\n"
            "## 5. Risk Assessment\n\n"
            "Generate the full BRD with substantial content for each section."
        ),
    ]
)
brd_agent_chain = brd_agent_prompt | llm3 | StrOutputParser()
print("--- 📄 BRD Agent LCEL Chain Compiled (Uses Key 3) ---")


# --- 3.7: STRATEGY AGENT (NEW) ---
strategy_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Chief Strategist. Your job is to generate a high-level strategic plan. "
            "The output must be a single Markdown string. Use clear headings and bullet points. "
            "Your response MUST be ONLY the Markdown text, starting with '# Strategic Approach'."
        ),
        (
            "human",
            "Please generate a strategic approach for the following goal. Break it down into 3-5 key phases or points. "
            "For each point, provide a brief description of *how* to approach it."
            "\n\n- **Project Topic:** {topic}"
            "\n- **Primary Goal:** {goal}"
        ),
    ]
)
strategy_agent_chain = strategy_agent_prompt | llm3 | StrOutputParser()
print("--- 📈 Strategy Agent LCEL Chain Compiled (Uses Key 3) ---")


# --- 4. AGENT "WORKSTATIONS" (The Nodes) ---

# --- NEW PDF HELPER FUNCTION ---
def save_markdown_as_pdf(markdown_text: str, filename: str) -> str:
    """
    Converts a Markdown string to a professional PDF file using fpdf2.
    Includes improved formatting with proper margins, spacing, and fonts.
    """
    try:
        pdf = FPDF()
        pdf.set_margins(15, 15, 15)  # Left, top, right margins
        pdf.add_page()
        
        # Function to sanitize text for Latin-1 encoding
        def sanitize_text(text):
            """Replace problematic Unicode characters with ASCII equivalents"""
            replacements = {
                '\u2011': '-',  # non-breaking hyphen → regular hyphen
                '\u2010': '-',  # hyphen → regular hyphen
                '\u2013': '--', # en dash → double hyphen
                '\u2014': '---',# em dash → triple hyphen
                '\u2018': "'",  # left single quote → apostrophe
                '\u2019': "'",  # right single quote → apostrophe
                '\u201c': '"',  # left double quote → quote
                '\u201d': '"',  # right double quote → quote
                '\u2026': '...', # ellipsis → three dots
                '\u00b0': 'o',  # degree symbol
                '\u00a9': '(c)',# copyright
                '\u00ae': '(R)',# registered
                '\u2122': '(TM)',# trademark
                '\u2022': '*',  # bullet point → asterisk
                '\u2023': '*',  # triangular bullet → asterisk
            }
            for unicode_char, ascii_equiv in replacements.items():
                text = text.replace(unicode_char, ascii_equiv)
            # Remove any remaining non-ASCII characters
            text = text.encode('ascii', 'ignore').decode('ascii')
            return text
        
        # Ensure output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert markdown to professional PDF (handle headings, lists, etc.)
        lines = markdown_text.split('\n')
        
        for line in lines:
            # Sanitize the line first
            line = sanitize_text(line)
            
            # Check if we need a new page (leave 30mm for footer)
            if pdf.get_y() > 250:
                pdf.add_page()
            
            if line.startswith('# '):
                # Main title - add spacing before, bold, larger font
                pdf.ln(3)
                pdf.set_font("Helvetica", 'B', 18)
                pdf.cell(0, 10, line.replace('# ', '').strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('## '):
                # Section heading - add spacing, bold
                pdf.ln(2)
                pdf.set_font("Helvetica", 'B', 14)
                pdf.cell(0, 10, line.replace('## ', '').strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('### '):
                # Subsection - slightly bold
                pdf.ln(1)
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 8, line.replace('### ', '').strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point - with proper indentation (use * instead of •)
                bullet_text = line.replace('- ', '').replace('* ', '').strip()
                pdf.set_font("Helvetica", size=10)
                pdf.set_x(20)  # Indent bullet points
                pdf.multi_cell(0, 6, '* ' + bullet_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            elif line.startswith('1. ') or (len(line) > 2 and line[0].isdigit() and line[1:3] == '. '):
                # Numbered list item
                pdf.set_font("Helvetica", size=10)
                pdf.set_x(20)  # Indent numbered items
                pdf.multi_cell(0, 6, line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            elif line.strip():
                # Regular paragraph text with better spacing
                pdf.set_font("Helvetica", size=11)
                pdf.multi_cell(0, 7, line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                # Empty line for spacing between sections
                pdf.ln(1.5)
        
        # Output to file
        pdf.output(filename)
        print(f"--- 📄 PDF saved as: {filename} ---")
        
        # Verify file was created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"--- ✅ PDF file verified: {filename} (size: {file_size} bytes) ---")
            return filename
        else:
            print(f"--- ❌ PDF file was not created: {filename} ---")
            return None
    except Exception as e:
        print(f"--- ❌ ERROR saving PDF: {e} ---")
        import traceback
        traceback.print_exc()
        return None

def planner_agent_node(state: CampaignState) -> dict:
    print("--- 1. 📋 Calling Planner Agent ---")
    # If fields are already populated (user confirmed an edited plan), skip LLM
    if state.goal and state.topic and state.target_audience:
        print("--- 📋 Planner Agent: Using pre-populated fields (user override). Skipping LLM. ---")
        # Return the pre-populated fields so they appear in the state diff
        return {
            "goal": state.goal,
            "topic": state.topic,
            "target_audience": state.target_audience,
            "company_name": state.company_name,
            "source_docs_url": state.source_docs_url,
            "campaign_date": state.campaign_date,
            "location": state.location,
        }
    brief = state.initial_prompt
    try:
        planner_output: PlannerOutput = planner_chain.invoke({"brief": brief})
        return planner_output.model_dump()
    except Exception as e:
        print(f"--- ❌ ERROR in Planner Agent: {e} ---")
        return {}

# --- KNOWN COUNTRY PORTALS (Tier 1 lookup) ---
KNOWN_COUNTRY_PORTALS = {
    "united states": "https://www.sba.gov/business-guide/launch-your-business",
    "usa": "https://www.sba.gov/business-guide/launch-your-business",
    "united kingdom": "https://www.gov.uk/set-up-business",
    "uk": "https://www.gov.uk/set-up-business",
    "canada": "https://ised-isde.canada.ca/site/corporations-canada/en",
    "australia": "https://business.gov.au/registrations",
    "india": "https://www.startupindia.gov.in",
    "germany": "https://www.existenzgruender.de/EN/Home/inhalt.html",
    "france": "https://www.guichet-entreprises.fr/en/",
    "singapore": "https://www.acra.gov.sg",
    "uae": "https://www.economy.gov.ae/english/pages/default.aspx",
    "south africa": "https://www.cipc.co.za",
    "nigeria": "https://www.cac.gov.ng",
    "kenya": "https://brs.go.ke",
    "brazil": "https://www.gov.br/empresas-e-negocios/pt-br",
    "japan": "https://www.moj.go.jp/ENGLISH/",
    "egypt": "https://www.gafi.gov.eg",
    "morocco": "https://www.invest.gov.ma",
}
print(f"--- 🌍 Loaded {len(KNOWN_COUNTRY_PORTALS)} known country portals ---")


def resolve_jurisdiction_from_portal(portal_url: str, country: str, topic: str):
    print(f"--- Portal scrape: {portal_url} ---")
    try:
        docs = WebBaseLoader(portal_url).load()
        content = docs[0].page_content[:4000] if docs else ""
    except:
        return None

    if not content.strip():
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Task: From this government portal content, find the ONE agency that handles business/startup/company registration. "
         "Return JSON only (name + real URL). Do not hallucinate.\n{format_instructions}"),
        ("human",
         "Country: {country}\nTopic: {topic}\n\nCONTENT:\n{content}")
    ]).partial(format_instructions=jurisdiction_parser.get_format_instructions())

    try:
        chain = prompt | llm0 | jurisdiction_parser
        r = chain.invoke({"country": country, "topic": topic, "content": content})
        if r.department_url not in ("", "N/A", "Unknown"):
            return r
    except Exception as e:
        print("Portal LLM fail:", e)
    return None


def search_jurisdiction_fallback_with_extract(country: str, topic: str, company_name: str):
    print(f"--- Fallback search: {country} ---")
    try:
        search = tavily_tool.invoke(
            f"official agency for business/startup/company registration in {country} {topic}"
        )
    except Exception as e:
        print("Tavily fail:", e)
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Task: Use search results to identify the correct real government department for company/startup registration. "
         "Return JSON only.\n{format_instructions}"),
        ("human",
         "Country: {country}\nTopic: {topic}\nCompany: {company_name}\nSearch:\n{search_results}")
    ]).partial(format_instructions=jurisdiction_parser.get_format_instructions())

    try:
        chain = prompt | llm0 | jurisdiction_parser
        r = chain.invoke({
            "country": country,
            "topic": topic,
            "company_name": company_name,
            "search_results": str(search)
        })
        if r.department_url not in ("", "Unknown", "N/A"):
            return r
    except Exception as e:
        print("Fallback LLM fail:", e)

    return None


def finalize_jurisdiction(country: str, topic: str, company_name: str):
    country_key = (country or "").lower()

    # Tier 1: Known portal
    portal = next((u for k, u in KNOWN_COUNTRY_PORTALS.items() if k in country_key), None)
    if portal:
        j = resolve_jurisdiction_from_portal(portal, country, topic)
        if j:
            return j
        print("Portal failed → fallback")

    # Tier 2: Search fallback
    j = search_jurisdiction_fallback_with_extract(country, topic, company_name)
    if j:
        return j

    print("❌ No jurisdiction found")
    return None


def jurisdiction_agent_node(state: CampaignState) -> dict:
    """Step 3 in user flow: discover jurisdiction & ministries, scrape procedures."""
    print("--- 2. 🏛️ Calling Jurisdiction Agent ---")
    location = state.location or ""
    topic = state.topic or ""
    company_name = state.company_name or ""
    campaign_date = state.campaign_date.isoformat() if state.campaign_date else ""

    result = {}

    if not location:
        print("--- ⚠️ No location provided — skipping jurisdiction discovery ---")
        result["jurisdiction_info"] = {
            "department_name": "N/A",
            "department_url": "",
            "jurisdiction_type": "N/A"
        }
        result["registration_procedure"] = []
        return result

    try:
        # ========================================
        # STEP 1: Jurisdiction Discovery
        # ========================================
        print(f"--- STEP 1: Jurisdiction for {location} ({topic}) ---")
        jurisdiction = finalize_jurisdiction(location, topic, company_name)

        if jurisdiction:
            result["jurisdiction_info"] = jurisdiction.model_dump()
            print(f"--- ✅ STEP 1 resolved → {jurisdiction.department_name} ---")
        else:
            result["jurisdiction_info"] = {
                "department_name": "Unknown",
                "department_url": "",
                "jurisdiction_type": "Unknown"
            }
            result["registration_procedure"] = []
            print(f"--- ❌ STEP 1 unresolved for {location} ---")
            return result

        # ========================================
        # STEP 2: Scrape Department Website + Extract Procedure
        # ========================================
        if jurisdiction.department_url:
            print(f"--- 📋 STEP 2: Reading department website: {jurisdiction.department_url} ---")
            website_content, raw_govt_content = _scrape_govt_website(jurisdiction.department_url)
            result["raw_govt_content"] = raw_govt_content   # save full scrape to state for validation

            try:
                procedure_search = tavily_tool.invoke(
                    f"how to register startup company at {jurisdiction.department_name} {location} "
                    f"step by step procedure requirements {campaign_date}"
                )
                procedure_inputs = {
                    "department_name": jurisdiction.department_name,
                    "department_url": jurisdiction.department_url,
                    "location": location,
                    "topic": topic,
                    "website_content": website_content,
                    "procedure_search": procedure_search,
                }
                procedure_chain = procedure_prompt | llm0 | procedure_parser
                procedure_output = procedure_chain.invoke(procedure_inputs)
                result["registration_procedure"] = procedure_output.registration_steps
                print(f"--- 📋 Extracted {len(procedure_output.registration_steps)} registration steps ---")
            except Exception as e:
                print(f"--- ⚠️ STEP 2 failed (procedure extraction): {e} ---")
                result["registration_procedure"] = []

        print(f"--- ✅ Jurisdiction Agent Complete ---")
        return result

    except Exception as e:
        print(f"--- ❌ ERROR in Jurisdiction Agent: {e} ---")
        pprint.pprint(e)
        return result if result else {}


def _scrape_govt_website(url: str) -> tuple:
    """
    Scrape the government website and return (truncated_content, full_raw_content).
    Always returns at least empty strings — never raises.
    """
    raw = ""
    try:
        # Try WebBaseLoader first
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs and len(docs[0].page_content) > 100:  # Only trust if we got substantial content
            raw = docs[0].page_content
            print(f"--- 📋 Scraped {len(raw)} chars from govt website ---")
            truncated = raw[:4000] if raw else "Could not load website."
            return truncated, raw
    except Exception as e:
        print(f"--- ⚠️ WebBaseLoader failed for {url}: {str(e)[:100]} ---")
    
    # Fallback: Try requests + beautifulsoup
    if not raw:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=5, headers=headers)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from main content areas
                main_content = soup.find(['main', 'article', 'div']) or soup.body
                if main_content:
                    raw = main_content.get_text(separator='\n', strip=True)
                    if raw:
                        print(f"--- 📋 Scraped {len(raw)} chars (fallback method) ---")
        except Exception as e:
            print(f"--- ⚠️ Fallback scrape also failed: {str(e)[:100]} ---")
    
    truncated = raw[:4000] if raw else "Could not load website. Using default documents."
    return truncated, raw


def research_agent_node(state: CampaignState) -> dict:
    """Step 4a: Audience research + required documents (uses jurisdiction from previous step)."""
    print("--- 3. 🧠 Calling Research Agent ---")
    location = state.location or ""
    topic = state.topic or ""
    target_audience = state.target_audience or ""
    campaign_date = state.campaign_date.isoformat() if state.campaign_date else ""

    # Pull jurisdiction data from state (set by jurisdiction_agent)
    jurisdiction_info = state.jurisdiction_info or {}
    department_name = jurisdiction_info.get("department_name", "N/A")
    department_url = jurisdiction_info.get("department_url", "N/A")
    registration_steps = state.registration_procedure or []

    # Default fallback with realistic documents for most jurisdictions
    location_lower = location.lower() if location else ""
    
    # Generic but realistic default documents
    default_documents = [
        {"document_name": "Business Registration Certificate", "issuing_authority": "Company Registration Office", "purpose": "Legal business entity registration", "deadline_note": "Before business launch"},
        {"document_name": "Company Incorporation Certificate", "issuing_authority": "State Business Registry", "purpose": "Proof of legal incorporation", "deadline_note": "Required for all startups"},
        {"document_name": "Tax Registration (GST/VAT)", "issuing_authority": "Tax Authority", "purpose": "Indirect tax compliance", "deadline_note": "30 days after launch (or as per local law)"},
        {"document_name": "Director/Owner Identification & Address Proof", "issuing_authority": "Government Authority", "purpose": "KYC compliance", "deadline_note": "Before registration"},
        {"document_name": "Memorandum & Articles of Association", "issuing_authority": "Company", "purpose": "Define company governance rules", "deadline_note": "Required during incorporation"},
    ]
    
    # Location-specific additions
    if location_lower in ["india", "in"]:
        default_documents.extend([
            {"document_name": "DIN (Director Identification Number)", "issuing_authority": "Ministry of Corporate Affairs", "purpose": "Director identification for Indian companies", "deadline_note": "Before incorporating company"},
            {"document_name": "ROC Registration (Registrar of Companies)", "issuing_authority": "ROC", "purpose": "Official company registration", "deadline_note": "Initial registration mandatory"},
        ])

    default_result = {
        "audience_persona": {
            "pain_point": f"Complex regulatory requirements for {topic} startups in {location}; Need clear guidance on compliance",
            "motivation": "Streamline business setup and ensure legal compliance",
            "preferred_channel": "Email guides, webinars, live support"
        },
        "core_messaging": {
            "value_proposition": f"Simplify {topic} startup registration with expert-guided compliance in {location}",
            "tone_of_voice": "Professional, supportive, transparent",
            "call_to_action": "Start your journey with confidence"
        },
        "required_documents": default_documents
    }

    result = {}

    # Build correction note from previous validation round (if any)
    correction_note = ""
    if state.validation_mismatches:
        correction_note = (
            "\n\n⚠️ CORRECTION REQUIRED (from validation audit round "
            f"{state.validation_rounds}):\n"
            + "\n".join(f"  - {m}" for m in state.validation_mismatches)
            + "\n\nPlease fix ALL of the above issues in this revised output. "
              "Do not repeat the errors listed."
        )

    try:
        print("--- 🔎 Running full research chain (audience + documents) ---")

        search_results = tavily_tool.invoke(
            f"common pain points for {target_audience} related to {topic}"
        )

        regulatory_news = ""
        if location:
            regulatory_news = tavily_tool.invoke(
                f"latest regulatory changes startup business registration {location} {campaign_date} news"
            )

        # Use the raw govt website content scraped by jurisdiction_agent as
        # primary product context.
        raw_govt = state.raw_govt_content or ""
        if raw_govt and len(raw_govt) > 100:  # Substantial content available
            scraped_content = (
                "=== OFFICIAL GOVERNMENT WEBSITE CONTENT (use as primary source for documents) ===\n"
                + raw_govt[:6000]
            )
            print(f"--- 📄 Feeding {len(raw_govt[:6000])} chars of govt content to research LLM ---")
        else:
            print(f"--- ⚠️ Minimal govt content ({len(raw_govt) if raw_govt else 0} chars), using web search + defaults ---")
            # Use search results as primary source instead
            scraped_content = (
                "=== WEB RESEARCH ON REGISTRATION REQUIREMENTS (fallback) ===\n"
                + str(search_results)[:4000]
                + "\n\nUse the above research to identify required documents and provide realistic defaults."
            )

        if correction_note:
            scraped_content += correction_note

        research_inputs = {
            "scraped_content": scraped_content,
            "topic": topic,
            "target_audience": target_audience,
            "search_results": search_results,
            "department_name": department_name,
            "department_url": department_url,
            "registration_procedure": "\n".join(f"{i+1}. {s}" for i, s in enumerate(registration_steps)) if registration_steps else "No procedure available.",
            "location": location,
            "campaign_date": campaign_date,
            "regulatory_news": str(regulatory_news) if regulatory_news else "No recent news.",
        }

        try:
            research_chain = research_prompt | llm0 | research_parser
            research_output: ResearchOutput = research_chain.invoke(research_inputs)
            research_dict = research_output.model_dump()

            result["audience_persona"] = research_dict.get("audience_persona", default_result["audience_persona"])
            result["core_messaging"] = research_dict.get("core_messaging", default_result["core_messaging"])

            if 'required_documents' in research_dict and research_dict['required_documents'] and len(research_dict['required_documents']) > 0:
                result['required_documents'] = [
                    doc if isinstance(doc, dict) else doc.model_dump()
                    for doc in research_dict['required_documents']
                ]
            else:
                result['required_documents'] = default_result['required_documents']

            print(f"--- ✅ Research Agent Complete: {len(result.get('required_documents', []))} documents found ---")
            return result
        except Exception as llm_err:
            print(f"--- ⚠️ Research LLM failed ({str(llm_err)[:100]}), using defaults ---")
            return default_result

    except Exception as e:
        print(f"--- ❌ ERROR in Research Agent: {str(e)[:100]} ---")
        print(f"--- ⚠️ Using default fallback values ---")
        # Return sensible defaults instead of empty dict
        return default_result


def validation_agent_node(state: CampaignState) -> dict:
    """
    Cross-validates registration steps and required documents against:
      1. The raw govt website content (always treated as ground truth).
      2. Fresh web searches via Tavily (2-3 targeted queries).

    Returns confidence scores, mismatches, and a flag for govt-only fallback.
    Never raises — always returns a safe dict so the graph can continue.
    """
    print(f"--- 🔍 Validation Agent — round {(state.validation_rounds or 0) + 1}/{MAX_VALIDATION_ROUNDS} ---")

    result: dict = {
        "validation_rounds": (state.validation_rounds or 0) + 1,
    }

    location    = state.location or ""
    topic       = state.topic    or ""
    raw_content = state.raw_govt_content or ""
    steps       = state.registration_procedure or []
    docs        = state.required_documents or []

    # Failsafe A: nothing to validate
    if not steps and not docs:
        print("--- ⚠️ Validation: no steps or docs to check — skipping ---")
        result["govt_fallback_only"] = bool(raw_content)
        result["overall_confidence"] = 0.0
        result["validation_mismatches"] = ["No registration steps or documents were generated."]
        return result

    # Failsafe B: no raw govt content — lower confidence ceiling
    if not raw_content:
        print("--- ⚠️ Validation: no raw govt content — web-search only mode ---")
        raw_content = "Government website could not be scraped. Use web search results as reference."

    # Step 1: 2-3 targeted Tavily searches
    web_search_results = ""
    search_queries = [
        f"{topic} company registration official requirements {location}",
        f"required documents {topic} startup registration {location}",
    ]
    if docs:
        first_doc = docs[0].get("document_name", "") if isinstance(docs[0], dict) else str(docs[0])
        if first_doc:
            search_queries.append(f'"{first_doc}" {location} official registration')

    for query in search_queries:
        try:
            result_chunk = tavily_tool.invoke(query)
            web_search_results += f"\n--- Query: {query} ---\n{result_chunk}\n"
        except Exception as e:
            print(f"--- ⚠️ Tavily search failed for '{query}': {e} ---")

    # Step 2: Format inputs for validation LLM
    steps_formatted = "\n".join(
        f"{i}. {s}" for i, s in enumerate(steps)
    ) or "No steps provided."

    docs_formatted = "\n".join(
        f"  • {d.get('document_name','?')} — Issuing authority: {d.get('issuing_authority','?')}"
        if isinstance(d, dict) else f"  • {d}"
        for d in docs
    ) or "No documents provided."

    # Step 3: Call validation LLM
    try:
        validation_output: ValidationOutput = validation_chain.invoke({
            "raw_govt_content":   raw_content,
            "registration_steps": steps_formatted,
            "required_documents": docs_formatted,
            "web_search_results": web_search_results or "No web results retrieved.",
            "location":           location,
            "topic":              topic,
        })

        print(
            f"--- ✅ Validation complete: confidence={validation_output.overall_confidence:.2f}, "
            f"validated={validation_output.is_validated}, "
            f"mismatches={len(validation_output.mismatches)} ---"
        )

        result["step_confidence"]     = validation_output.step_confidence
        result["document_confidence"] = validation_output.document_confidence
        result["overall_confidence"]  = validation_output.overall_confidence
        result["validation_mismatches"] = (
            validation_output.mismatches
            + [f"Missing doc: {d}" for d in validation_output.missing_docs]
            + [f"Missing step: {s}" for s in validation_output.missing_steps]
        )

        # Append missing documents discovered by validation back into
        # required_documents so they appear in the UI even if the research
        # LLM omitted them.  These are documents the govt website lists
        # that the AI never generated.
        existing_doc_names = {
            (d.get("document_name", "") if isinstance(d, dict) else str(d)).lower()
            for d in docs
        }
        new_docs = list(docs)  # shallow copy
        for doc_name in validation_output.missing_docs:
            if doc_name.lower() not in existing_doc_names:
                new_docs.append({
                    "document_name": doc_name,
                    "issuing_authority": "See official government website",
                    "purpose": "Listed on the official government registration page but not generated by AI research.",
                    "deadline_note": "Check official website for deadline",
                })
                print(f"--- 📎 Added missing doc from validation: {doc_name} ---")
        if len(new_docs) > len(docs):
            result["required_documents"] = new_docs
            # Also give the newly-added docs a low confidence score
            for doc_name in validation_output.missing_docs:
                if doc_name not in result["document_confidence"]:
                    result["document_confidence"][doc_name] = 0.5

        # Failsafe C: catastrophic confidence + no govt content
        if validation_output.overall_confidence < 0.3 and not state.raw_govt_content:
            print("--- ❌ Confidence critically low and no govt scrape — activating fallback ---")
            result["govt_fallback_only"] = True
        else:
            result["govt_fallback_only"] = False

    except Exception as e:
        print(f"--- ❌ Validation LLM failed: {e} — generating fallback validation notes ---")
        result["step_confidence"]     = {str(i): 0.6 for i in range(len(steps))}
        result["document_confidence"] = {
            (d.get("document_name", f"doc_{i}") if isinstance(d, dict) else f"doc_{i}"): 0.6
            for i, d in enumerate(docs)
        }
        result["overall_confidence"]  = 0.6
        
        # Generate meaningful fallback validation notes instead of empty list
        fallback_mismatches = []
        if len(docs) < 3:
            fallback_mismatches.append("⚠️ Limited documents found - verify official registration requirements")
        if len(steps) < 3:
            fallback_mismatches.append("⚠️ Limited procedure steps found - check official government portal")
        if not raw_content or len(raw_content) < 100:
            fallback_mismatches.append("⚠️ Unable to verify against official government website - check source directly")
        if not fallback_mismatches:
            fallback_mismatches.append("✓ Documentation review pending - cross-reference with official sources")
        
        result["validation_mismatches"] = fallback_mismatches
        result["govt_fallback_only"]  = False

    return result


def route_after_validation(state: CampaignState) -> str:
    """
    Conditional router after validation_agent.

    Loop back to research_agent if:
      - Confidence is below threshold, AND
      - We haven't hit the max re-validation rounds, AND
      - There are actual mismatches to correct.

    Otherwise: proceed to strategy_agent (even if imperfect — govt scrape is the safety net).
    """
    rounds      = state.validation_rounds or 0
    confidence  = state.overall_confidence if state.overall_confidence is not None else 1.0
    mismatches  = state.validation_mismatches or []

    needs_rerun = (
        confidence < CONFIDENCE_THRESHOLD
        and rounds < MAX_VALIDATION_ROUNDS
        and len(mismatches) > 0
    )

    if needs_rerun:
        print(
            f"--- 🔄 Routing back to research_agent "
            f"(conf={confidence:.2f}, round={rounds}/{MAX_VALIDATION_ROUNDS}) ---"
        )
        return "research_agent"

    if state.govt_fallback_only:
        print("--- 🏛️  Govt fallback mode active — proceeding with raw scrape only ---")

    print(f"--- ➡️  Routing to strategy_agent (conf={confidence:.2f}, round={rounds}) ---")
    return "strategy_agent"


def content_agent_node(state: CampaignState) -> dict:
    print("--- 4. ✍️ Calling Content Agent (REAL) ---")
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
    print("--- 5. 🎨 Calling Design Agent (REAL) ---")
    
    mock_brand_kit = BrandKit(
        logo_prompt=f"A minimalist, tech-inspired logo for {state.topic}",
        color_palette=["#0A0A0A", "#FFFFFF", "#4F46E5", "#FBBF24", "#10B981"], # Dark, White, Blue, Yellow, Green
        font_pair="Inter" # Using a single modern font
    )
    
    generated_assets = {}
    
    print("--- 🎨 Generating Webinar Banner... ---")
    generated_assets["webinar_banner_url"] = get_unsplash_image(state.webinar_image_prompt or state.topic or "abstract")
    
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
    print("--- 6. 🕸️ Calling Web Agent (REAL) ---")
    
    try:
        company_name = state.company_name or state.topic or "Company"
        inputs = {
            "topic": state.topic,
            "audience_persona": state.audience_persona,
            "core_messaging": state.core_messaging,
            "company_name": company_name,
            "generated_assets": state.generated_assets,
        }

        print("--- 🕸️ Generating landing page sections (LLM) + wrapping boilerplate... ---")
        sections_raw = web_sections_chain.invoke(inputs)
        sections_html = _extract_body_like_html(sections_raw)
        html_code = build_landing_page_html(company_name=company_name, sections_html=sections_html)

        return {"landing_page_code": html_code, "landing_page_url": "campaign_preview.html"}

    except Exception as e:
        print(f"--- ❌ ERROR in Web Agent: {e} ---")
        pprint.pprint(e)
        return {}

# --- NEW AGENT NODE (BRD) ---
def brd_agent_node(state: CampaignState) -> dict:
    print("--- 7. 📄 Calling BRD Agent (Generate BRD via Key 3) ---")
    try:
        strategy_markdown = state.strategy_markdown or "# Strategic Approach\n\nNo strategy available."
        
        # Call BRD agent to generate full BRD based on strategy
        inputs = {"strategy_markdown": strategy_markdown}
        print("--- 📄 Generating Business Requirements Document... ---")
        brd_markdown = brd_agent_chain.invoke(inputs)
        
        # Create a directory for outputs if it doesn't exist
        output_dir = "campaign_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        topic_slug = (state.topic or "campaign").lower().replace(' ', '_')
        filename = f"{output_dir}/{topic_slug}_brd.pdf"
        pdf_path = save_markdown_as_pdf(brd_markdown, filename)
        
        return {"brd_url": pdf_path}

    except Exception as e:
        print(f"--- ❌ ERROR in BRD Agent: {e} ---")
        pprint.pprint(e)
        return {}

# --- MODIFIED STRATEGY AGENT (Uses Key 3) ---
def strategy_agent_node(state: CampaignState) -> dict:
    print("--- 6. 📈 Calling Strategy Agent (Key 3) ---")
    try:
        inputs = {
            "topic": state.topic,
            "goal": state.goal,
        }
        print("--- 📈 Generating Strategy Markdown via Key 3... ---")
        strategy_markdown = strategy_agent_chain.invoke(inputs)
        
        return {"strategy_markdown": strategy_markdown}

    except Exception as e:
        print(f"--- ❌ ERROR in Strategy Agent: {e} ---")
        pprint.pprint(e)
        return {}


def ops_agent_node(state: CampaignState) -> dict:
    print("--- 8. ⚙️ Ops Agent (Slack + Telegram) Started ---")

    # ------------------------------  
    # 1. Load Slack + Telegram credentials
    # ------------------------------
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    results = {
        "slack": [],
        "telegram": []
    }

    # ------------------------------
    # 2. Loop through each generated post
    # ------------------------------
    for i, post in enumerate(state.social_posts):
        text = post.content.strip()
        image_key = f"post_{i+1}_image_url"
        image_url = state.generated_assets.get(image_key)

        # =====================================================
        #  A) SEND TO SLACK
        # =====================================================
        if SLACK_WEBHOOK:
            try:
                print(f"📤 Sending post {i+1} to Slack...")

                if image_url:
                    slack_payload = {
                        "blocks": [
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": text}
                            },
                            {
                                "type": "image",
                                "image_url": image_url,
                                "alt_text": f"image_post_{i+1}"
                            }
                        ]
                    }
                else:
                    slack_payload = {"text": text}

                resp = requests.post(
                    SLACK_WEBHOOK,
                    json=slack_payload,
                    timeout=10
                )
                results["slack"].append({
                    "post_number": i + 1,
                    "status": resp.status_code
                })
            except Exception as e:
                results["slack"].append({
                    "post_number": i + 1,
                    "error": str(e)
                })
                print(f"--- ❌ Slack Error on post {i+1}: {e} ---")

        # =====================================================
        #  B) SEND TO TELEGRAM
        # =====================================================
        if BOT_TOKEN and CHAT_ID:
            try:
                print(f"📤 Sending post {i+1} to Telegram...")

                if image_url:
                    tg_resp = requests.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                        data={"chat_id": CHAT_ID, "caption": text, "photo": image_url},
                        timeout=10
                    ).json()
                else:
                    tg_resp = requests.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                        data={"chat_id": CHAT_ID, "text": text},
                        timeout=10
                    ).json()

                results["telegram"].append({
                    "post_number": i + 1,
                    "response": tg_resp
                })
            except Exception as e:
                results["telegram"].append({
                    "post_number": i + 1,
                    "error": str(e)
                })
                print(f"--- ❌ Telegram Error on post {i+1}: {e} ---")

    print("--- 8. ⚙️ Ops Agent Finished (Slack + Telegram) ---")

    return {
        "automation_status": {
            "slack_results": results["slack"],
            "telegram_results": results["telegram"],
            "status": "completed"
        }
    }



# --- 5. LANGGRAPH "FACTORY FLOOR" (The Graph) ---

graph_builder = StateGraph(CampaignState)

# Add all nodes
graph_builder.add_node("planner_agent",      planner_agent_node)
graph_builder.add_node("jurisdiction_agent",  jurisdiction_agent_node)
graph_builder.add_node("research_agent",      research_agent_node)
graph_builder.add_node("validation_agent",    validation_agent_node)   # ← NEW
graph_builder.add_node("content_agent",       content_agent_node)
graph_builder.add_node("design_agent",        design_agent_node)
graph_builder.add_node("web_agent",           web_agent_node)
graph_builder.add_node("brd_agent",           brd_agent_node)
graph_builder.add_node("strategy_agent",      strategy_agent_node)
graph_builder.add_node("ops_agent",           ops_agent_node)

# Linear flow up to validation
graph_builder.set_entry_point("planner_agent")
graph_builder.add_edge("planner_agent",      "jurisdiction_agent")
graph_builder.add_edge("jurisdiction_agent", "research_agent")
graph_builder.add_edge("research_agent",     "validation_agent")    # ← NEW

# Conditional loop: validation → research_agent (retry) OR strategy_agent (proceed)
graph_builder.add_conditional_edges(
    "validation_agent",
    route_after_validation,
    {
        "research_agent":  "research_agent",   # re-run with corrections
        "strategy_agent":  "strategy_agent",   # proceed
    }
)

# Remainder of pipeline unchanged
graph_builder.add_edge("strategy_agent", "content_agent")
graph_builder.add_edge("content_agent",  "design_agent")
graph_builder.add_edge("design_agent",   "web_agent")
graph_builder.add_edge("web_agent",      "brd_agent")
graph_builder.add_edge("brd_agent",      "ops_agent")
graph_builder.add_edge("ops_agent",      END)


# Compile the graph
print("--- 🏭 Compiling AI Campaign Foundry Graph (with Validation Loop) ---")
sys.setrecursionlimit(200) 
foundry_app = graph_builder.compile()
print("--- ✅ Foundry Graph Compiled ---")


# --- 6. FASTAPI SERVER (The Streaming Endpoint) ---

from fastapi.responses import FileResponse

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferPlanRequest(BaseModel):
    initial_prompt: str

class StreamRequest(BaseModel):
    initial_prompt: str
    # Optional planner overrides — if provided, planner agent will use these instead of LLM
    goal: Optional[str] = None
    topic: Optional[str] = None
    target_audience: Optional[str] = None
    company_name: Optional[str] = None
    source_docs_url: Optional[str] = None
    campaign_date: Optional[str] = None
    location: Optional[str] = None



class RegenerateWebRequest(BaseModel):
    topic: Optional[str] = None
    goal: Optional[str] = None
    audience_persona: Optional[Dict[str, str]] = None
    core_messaging: Optional[Dict[str, str]] = None
    generated_assets: Optional[Dict[str, str]] = None
    company_name: Optional[str] = None


# Separate LLM with temperature for regeneration variety
regen_llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.9, api_key=_grok_key2)
print(f"--- 🤖 Regen LLM (Key 2) Initialized ---")
regen_sections_chain = web_sections_prompt | regen_llm | StrOutputParser()


@app.post("/regenerate_landing_page")
async def regenerate_landing_page(request: RegenerateWebRequest):
    """Regenerate the landing page HTML by calling the Web Agent only."""
    try:
        company_name = request.company_name or request.topic or "Company"
        inputs = {
            "topic": request.topic or "",
            "audience_persona": request.audience_persona or {},
            "core_messaging": request.core_messaging or {},
            "company_name": company_name,
            "generated_assets": request.generated_assets or {},
        }

        sections_raw = regen_sections_chain.invoke(inputs)
        sections_html = _extract_body_like_html(sections_raw)
        html_code = build_landing_page_html(company_name=company_name, sections_html=sections_html)
        return {"success": True, "html": html_code}

    except Exception as e:
        print(f"--- ❌ ERROR regenerating landing page: {e} ---")
        return {"success": False, "error": str(e)}

@app.post("/infer_plan")
async def infer_plan(request: InferPlanRequest):
    """Run only the planner agent to infer a business plan from the prompt."""
    try:
        planner_output: PlannerOutput = planner_chain.invoke({"brief": request.initial_prompt})
        result = planner_output.model_dump()
        # Convert datetime to string for JSON serialization
        if result.get("campaign_date"):
            result["campaign_date"] = result["campaign_date"].isoformat() if hasattr(result["campaign_date"], 'isoformat') else str(result["campaign_date"])
        return {"success": True, "plan": result}
    except Exception as e:
        print(f"--- ❌ ERROR in /infer_plan: {e} ---")
        return {"success": False, "error": str(e)}


@app.websocket("/ws_stream_campaign")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- 🔌 WebSocket Connection Accepted ---")
    try:
        json_data = await websocket.receive_json()
        request_data = StreamRequest(**json_data)
        
        initial_input = {"initial_prompt": request_data.initial_prompt}
        
        # If planner overrides are provided, pre-populate the state
        if request_data.goal:
            initial_input["goal"] = request_data.goal
        if request_data.topic:
            initial_input["topic"] = request_data.topic
        if request_data.target_audience:
            initial_input["target_audience"] = request_data.target_audience
        if request_data.company_name:
            initial_input["company_name"] = request_data.company_name
        if request_data.source_docs_url:
            initial_input["source_docs_url"] = request_data.source_docs_url
        if request_data.campaign_date:
            try:
                initial_input["campaign_date"] = datetime.fromisoformat(request_data.campaign_date)
            except Exception:
                pass
        if request_data.location:
            initial_input["location"] = request_data.location
        
        current_state_dict = initial_input.copy()
        
        print(f"--- 🚀 Received input, starting stream... ---")
        
        async for s in foundry_app.astream(initial_input):
            node_that_ran = list(s.keys())[0]
            state_snapshot_diff = s[node_that_ran]  # Dict of only the fields this node changed

            if state_snapshot_diff:
                # Always replace — never mutate. extend/update would corrupt lists and dicts
                for key, value in state_snapshot_diff.items():
                    current_state_dict[key] = value

            # model_dump() returns a plain dict — NOT a string.
            state_dict = CampaignState.model_validate(current_state_dict).model_dump(mode="json")

            await websocket.send_json({
                "event": "step",
                "node": node_that_ran,
                "data": state_dict   # plain dict → frontend receives a proper object
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


@app.get("/")
async def root():
    return {"message": "AI Campaign Foundry Server is running. Connect via WebSocket."}

@app.get("/download_brd/{filename}")
async def download_brd(filename: str):
    """Serve BRD PDF files for download"""
    file_path = os.path.join("campaign_outputs", filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

class DeployRequest(BaseModel):
    html_content: str
    project_name: str

@app.post("/deploy_to_vercel")
async def deploy_to_vercel(request: DeployRequest):
    """Deploy HTML content to Vercel"""
    
    VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")
    
    if not VERCEL_TOKEN:
        return {"error": "VERCEL_TOKEN not found in environment variables"}
    
    try:
        # Create deployment payload
        deployment_payload = {
            "name": request.project_name,
            "files": [
                {
                    "file": "index.html",
                    "data": request.html_content
                }
            ],
            "projectSettings": {
                "framework": None
            }
        }
        
        # Deploy to Vercel
        headers = {
            "Authorization": f"Bearer {VERCEL_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.vercel.com/v13/deployments",
            headers=headers,
            json=deployment_payload,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            deployment_url = data.get("url", "")
            
            # Vercel returns URL without protocol
            if deployment_url and not deployment_url.startswith("http"):
                deployment_url = f"https://{deployment_url}"
            
            return {
                "success": True,
                "url": deployment_url,
                "id": data.get("id"),
                "name": data.get("name")
            }
        else:
            error_data = response.json()
            return {
                "error": error_data.get("error", {}).get("message", "Deployment failed"),
                "status_code": response.status_code
            }
            
    except Exception as e:
        print(f"--- ❌ ERROR deploying to Vercel: {e} ---")
        return {"error": str(e)}

if __name__ == "__main__":
    print("--- 🚀 Starting FastAPI server on http://localhost:8000 ---")
    uvicorn.run(app, host="localhost", port=8000)