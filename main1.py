"""
Bettermind Labs Project Generator Backend - Milestone 3 (Profile-Driven Identity Inversion)
Implements an identity-first 7-stage reasoning loop with local dense embeddings.
Preserves FastAPI routing signatures, Tally ingestion layers, and strict API contracts.
"""

import os
import uuid
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("bettermind-backend")

app = FastAPI(
    title="Bettermind Labs Project Generator",
    version="3.0.0",
    description="Milestone 3 - Identity-First Profile Enrichment & Advanced Semantic Match Matrix"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GLOBAL STATE & DEEP EMBEDDING CORES
# ---------------------------------------------------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
EMBEDDING_MODEL: Optional[SentenceTransformer] = None
PROJECTS_DF: Optional[pd.DataFrame] = None
PROJECT_EMBEDDINGS: Optional[np.ndarray] = None

CSV_FILE_PATH = "student_projects.csv"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TALLY_API_KEY = os.getenv("TALLY_API_KEY")
TALLY_FORM_ID = os.getenv("TALLY_FORM_ID", "0Q6NNB")

VALID_THEMES = [
    "AI + Transportation", "AI + Space Exploration", "AI + Healthcare", 
    "AI + Humanities", "AI + Cybersecurity", "AI + Education", 
    "AI + Finance", "AI + Law", "AI + Environment", "AI + Sports", 
    "AI + Social Good", "AI + Business", "AI + Robotics", "AI + Quants", 
    "AI + Psychology"
]

# ---------------------------------------------------------------------------
# PYDANTIC ROUTING SCHEMAS (STRICT COMPATIBILITY)
# ---------------------------------------------------------------------------
class StudentProfile(BaseModel):
    email: EmailStr
    interest_areas: str
    grade: int = Field(..., ge=9, le=12)
    city: str
    toughest_project: str
    proposed_idea: Optional[str] = None
    preferred_college: Optional[str] = None
    session_id: Optional[str] = None

class ParticipantInput(BaseModel):
    email: EmailStr
    interest_areas: Optional[str] = None
    grade: Optional[int] = Field(None, ge=9, le=12)
    proposed_idea: Optional[str] = None

class BettermindGenerateRequest(BaseModel):
    mode: str = Field("bettermind", description="Must be exactly 'bettermind'")
    primary_student: ParticipantInput
    participants: List[ParticipantInput] = Field(default_factory=list)

class SimilarProject(BaseModel):
    title: str
    student_name: str
    description: str
    github_url: str
    youtube_url: str
    project_category: str
    similarity_score: float
    matching_reason: str

class ProjectStrengthScore(BaseModel):
    research_depth: int = Field(..., ge=1, le=10)
    real_world_impact: int = Field(..., ge=1, le=10)
    technical_depth: int = Field(..., ge=1, le=10)
    college_application_strength: int = Field(..., ge=1, le=10)

class ProjectIdea(BaseModel):
    id: str
    title: str
    problem_statement: str
    research_question: str
    why_it_matters: str
    difficulty: str
    college_application_value: str
    similar_projects: List[SimilarProject] = []
    project_strength_score: Optional[ProjectStrengthScore] = None

class GenerateIdeasResponse(BaseModel):
    session_id: str
    ideas: List[ProjectIdea]

class NicheDownRequest(BaseModel):
    session_id: str
    idea_id: str

# ---------------------------------------------------------------------------
# TALLY IN-MEMORY TELEMETRY LAYERS
# ---------------------------------------------------------------------------
class TallyProfileService:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_updated: Optional[datetime] = None
        self.lock = asyncio.Lock()

    def get_mock_submissions(self) -> List[Dict[str, Any]]:
        return [
            {
                "submissionId": "mock_sub_1",
                "fields": [
                    {"label": "Full Name", "value": "Aarav Roy"},
                    {"label": "Email Address", "value": "aarav@example.com"},
                    {"label": "Grade Level", "value": "11"},
                    {"label": "Timezone", "value": "EST"},
                    {"label": "City", "value": "New York"},
                    {"label": "State", "value": "NY"},
                    {"label": "School", "value": "Stuyvesant High School"},
                    {"label": "Graduation Year", "value": "2027"},
                    {"label": "Future Academic/Career Goal", "value": "Wants to build a quantitative high-frequency trading platform and lead multi-agent fintech systems."},
                    {"label": "Intended College Major", "value": "Computer Science and Quantitative Economics"},
                    {"label": "Project Themes of Interest", "value": ["AI + Finance", "AI + Quants", "AI + Business"]},
                    {"label": "Programming Experience Level", "value": "Advanced"},
                    {"label": "AI/ML Experience Level", "value": "Advanced"},
                    {"label": "Programming Languages Known", "value": "Python, C++, Rust"},
                    {"label": "Most Challenging Subject Currently", "value": "Multivariable Calculus"},
                    {"label": "Extracurriculars / Clubs Info", "value": "Founder of Student Quant Investment Club, Captain of Math Team"},
                    {"label": "Toughest Project Completed", "value": "Published a 3D CNN network architecture for modeling order-book liquidity distribution changes."},
                    {"label": "Existing Project Idea Context", "value": "Liquidity risk analysis and NLP tracking of underpriced tech sector IPO allocations."},
                    {"label": "Preferred Call Time Window", "value": "4:00 PM - 7:00 PM"},
                    {"label": "Preferred Interactive Session Time", "value": "Saturdays"},
                    {"label": "LinkedIn Profile URL", "value": "https://linkedin.com/in/aarav-mock"},
                    {"label": "Counselor Name", "value": "Sarah Jenkins"},
                    {"label": "Counselor Contact Email", "value": "sjenkins@school.edu"}
                ]
            }
        ]

    async def fetch_all_submissions(self):
        async with self.lock:
            if not TALLY_API_KEY:
                logger.warning("TALLY_API_KEY environment configuration missing. Initializing Sandbox Fallbacks.")
                submissions = self.get_mock_submissions()
            else:
                try:
                    logger.info(f"Syncing with Tally Context Reference Form ID: {TALLY_FORM_ID}")
                    all_fetched_submissions = []
                    page = 1
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        while True:
                            url = f"https://api.tally.so/forms/{TALLY_FORM_ID}/submissions"
                            params = {"filter": "completed", "page": page, "limit": 50}
                            headers = {"Authorization": f"Bearer {TALLY_API_KEY}"}
                            response = await client.get(url, params=params, headers=headers)
                            if response.status_code != 200:
                                if page == 1:
                                    all_fetched_submissions = self.get_mock_submissions()
                                break
                            data = response.json()
                            submissions_chunk = data.get("submissions", [])
                            if not submissions_chunk:
                                break
                            q_items = data.get("questions", [])
                            questions_map = {q["id"]: q["title"] for q in q_items if isinstance(q, dict) and "id" in q and "title" in q}
                            for sub in submissions_chunk:
                                fields = []
                                for res in sub.get("responses", []):
                                    q_title = questions_map.get(res.get("questionId"))
                                    if q_title and res.get("answer") is not None:
                                        fields.append({"label": q_title, "value": res.get("answer")})
                                all_fetched_submissions.append({"submissionId": sub.get("id", "N/A"), "fields": fields})
                            if not data.get("hasMore") or page > 30:
                                break
                            page += 1
                    submissions = all_fetched_submissions
                except Exception as e:
                    logger.error(f"Fallback executed due to Tally API error: {str(e)}")
                    submissions = self.get_mock_submissions()

            new_cache = {}
            for item in submissions:
                normalized = self._normalize_fields(item.get("fields", []))
                if normalized.get("email"):
                    new_cache[normalized["email"]] = normalized
            self.cache = new_cache
            self.last_updated = datetime.utcnow()
            logger.info(f"Tally synchronization complete. Pool size: {len(self.cache)} profiles.")

    def _normalize_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_map = {f.get("label", "").lower().strip(): f.get("value") for f in fields if f.get("label")}
        extracted = {
            "full_name": raw_map.get("full name", raw_map.get("name", "")),
            "email": raw_map.get("email address", raw_map.get("email", "")),
            "grade": raw_map.get("grade level", raw_map.get("grade", "11")),
            "timezone": raw_map.get("timezone", "EST"),
            "city": raw_map.get("city", ""),
            "state": raw_map.get("state", ""),
            "school": raw_map.get("school", raw_map.get("high school", "")),
            "graduation_year": raw_map.get("graduation year", ""),
            "future_goal": raw_map.get("future academic/career goal", raw_map.get("future goal", "")),
            "major": raw_map.get("intended college major", raw_map.get("major", "")),
            "project_themes": raw_map.get("project themes of interest", raw_map.get("themes", [])),
            "programming_experience": raw_map.get("programming experience level", "Beginner"),
            "ai_ml_experience": raw_map.get("ai/ml experience level", "Beginner"),
            "programming_languages": raw_map.get("programming languages known", ""),
            "challenging_subject": raw_map.get("most challenging subject currently", ""),
            "club_information": raw_map.get("extracurriculars / clubs info", ""),
            "toughest_project": raw_map.get("toughest project completed", ""),
            "existing_project_idea": raw_map.get("existing project idea context", ""),
            "preferred_call_time": raw_map.get("preferred call time window", ""),
            "preferred_session_time": raw_map.get("preferred interactive session time", ""),
            "linkedin": raw_map.get("linkedin profile url", ""),
            "counselor_name": raw_map.get("counselor name", ""),
            "counselor_email": raw_map.get("counselor contact email", "")
        }
        if isinstance(extracted["email"], str):
            extracted["email"] = extracted["email"].lower().strip()
        if isinstance(extracted["project_themes"], str):
            extracted["project_themes"] = [t.strip() for t in extracted["project_themes"].split(",") if t.strip()]
        elif not isinstance(extracted["project_themes"], list):
            extracted["project_themes"] = []
        extracted["project_themes"] = [t for t in extracted["project_themes"] if t in VALID_THEMES]
        return extracted

    def lookup_student(self, email: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(email.lower().strip())

TALLY_SERVICE = TallyProfileService()

async def cache_refresh_scheduler():
    while True:
        await asyncio.sleep(1800)
        try:
            await TALLY_SERVICE.fetch_all_submissions()
        except Exception as e:
            logger.error(f"Error inside scheduler routine: {str(e)}")

# ---------------------------------------------------------------------------
# CORE APPLICATION STARTUP EVENTS & HISTORICAL METADATA ENRICHMENT
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_pipeline_sequence():
    global EMBEDDING_MODEL, PROJECTS_DF, PROJECT_EMBEDDINGS
    logger.info("Initializing Dense Vector Architecture Layer (all-MiniLM-L6-v2)...")
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    if not os.path.exists(CSV_FILE_PATH):
        mock_data = {
            "Title": ["ChiralAI", "EcoSense Nano", "AlphaSpread GNN", "DeFi Risk Engine"],
            "Category": ["Healthcare", "Environment", "Finance", "Quants"],
            "Description": [
                "AI model predicting biosynthetic pathways for hard-to-make chiral molecules used in medicine.",
                "Low-cost carbon nanotube sensor array mapping particulate pollution vector changes across urban centers.",
                "Temporal Graph Neural Network modeling liquidity cascades and risk contagion inside order-books.",
                "Automated multi-agent execution pipeline simulating decentralized liquidation parameters under historical tail-risk scenarios."
            ],
            "StudentName": ["Alexei Manuel", "Sarah Jenkins", "Ethan Zhao", "Rohan Mehta"],
            "Github URL": ["https://github.com/aalxi/ChiralAI", "https://github.com/sj/ecosense", "https://github.com/ez/alphaspread", "https://github.com/rm/defirisk"],
            "Youtube URL": ["https://youtube.com/watch?v=mock1", "https://youtube.com/watch?v=mock2", "https://youtube.com/watch?v=mock3", "https://youtube.com/watch?v=mock4"],
            "Project Category": ["AI + Healthcare", "Hardware + Climate", "AI + Finance", "AI + Quants"]
        }
        PROJECTS_DF = pd.DataFrame(mock_data)
    else:
        PROJECTS_DF = pd.read_csv(CSV_FILE_PATH)

    for col in PROJECTS_DF.columns:
        PROJECTS_DF[col] = PROJECTS_DF[col].fillna("").astype(str).str.strip()

    logger.info("Building enriched multi-dimensional text anchors for historical benchmarks...")
    enriched_docs = []
    for idx, row in PROJECTS_DF.iterrows():
        title = row.get("Title", "")
        desc = row.get("Description", "")
        cat = row.get("Project Category", "")
        category = row.get("Category", "")
        
        dense_doc = (
            f"TITLE: {title} | DOMAIN: {cat} | SECTOR: {category}\n"
            f"CORE BLUEPRINT DESCRIPTION: {desc}\n"
            f"PROBLEM SOLVED: Granular localized structural or computational friction and systemic data anomalies.\n"
            f"ADMISSIONS NARRATIVE: Demonstrates high domain agency, rigorous data engineering feature maps, and practical execution parameters over generic buzzwords."
        )
        enriched_docs.append(dense_doc)
        PROJECTS_DF.at[idx, "search_doc"] = dense_doc

    logger.info("Generating corpus vector matrix space...")
    PROJECT_EMBEDDINGS = EMBEDDING_MODEL.encode(enriched_docs, convert_to_numpy=True)
    
    await TALLY_SERVICE.fetch_all_submissions()
    asyncio.create_task(cache_refresh_scheduler())

# ---------------------------------------------------------------------------
# HIGH-FIDELITY PRE-GENERATION RETRIEVAL INTERFACE
# ---------------------------------------------------------------------------
def execute_dense_retrieval(query_context: str, top_k: int = 15) -> List[Dict[str, Any]]:
    if PROJECTS_DF is None or PROJECTS_DF.empty or EMBEDDING_MODEL is None or PROJECT_EMBEDDINGS is None:
        return []
    
    query_vec = EMBEDDING_MODEL.encode([query_context], convert_to_numpy=True)
    scores = np.dot(PROJECT_EMBEDDINGS, query_vec.T).flatten()
    norms = np.linalg.norm(PROJECT_EMBEDDINGS, axis=1) * np.linalg.norm(query_vec)
    similarities = np.divide(scores, norms, out=np.zeros_like(scores), where=norms != 0)
    
    top_indices = similarities.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        row = PROJECTS_DF.iloc[idx]
        results.append({
            "idx": int(idx),
            "title": row.get("Title"),
            "student_name": row.get("StudentName"),
            "description": row.get("Description"),
            "github_url": row.get("Github URL"),
            "youtube_url": row.get("Youtube URL"),
            "project_category": row.get("Project Category"),
            "similarity_score": float(similarities[idx]),
            "search_doc": row.get("search_doc")
        })
    return results

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm > 0 else 0.0

# ---------------------------------------------------------------------------
# MULTI-AGENT SYNTHESIZED PATTERN GENERATORS (GROUP MODE MATRIX)
# ---------------------------------------------------------------------------
def resolve_and_score_profile(inp: ParticipantInput) -> Dict[str, Any]:
    tally = TALLY_SERVICE.lookup_student(inp.email) or {}
    resolved = {
        "full_name": tally.get("full_name", inp.email.split("@")[0].capitalize()),
        "email": inp.email.lower().strip(),
        "grade": str(inp.grade) if inp.grade else tally.get("grade", "11"),
        "city": tally.get("city", "Unknown"),
        "future_goal": tally.get("future_goal", "Advanced Engineering Systems"),
        "major": tally.get("major", "Computer Science"),
        "project_themes": tally.get("project_themes", []),
        "programming_experience": tally.get("programming_experience", "Intermediate"),
        "ai_ml_experience": tally.get("ai_ml_experience", "Beginner"),
        "programming_languages": tally.get("programming_languages", ""),
        "club_information": tally.get("club_information", "N/A"),
        "toughest_project": tally.get("toughest_project", ""),
        "existing_project_idea": inp.proposed_idea or tally.get("existing_project_idea", "")
    }
    if inp.interest_areas:
        manual = [t.strip() for t in inp.interest_areas.split(",") if t.strip()]
        resolved["project_themes"] = [t for t in manual if t in VALID_THEMES]
    return resolved

def compile_identity_graph(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_themes = [t for p in profiles for t in p["project_themes"]]
    unique_themes = list(set(all_themes))
    
    experience_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
    floor_score = 3
    for p in profiles:
        score = min(experience_map.get(p["programming_experience"].lower(), 2), experience_map.get(p["ai_ml_experience"].lower(), 1))
        if score < floor_score:
            floor_score = score
            
    calibrated_diff = "Intermediate" if floor_score == 1 else "Advanced"
    
    return {
        "combined_interests": unique_themes if unique_themes else ["AI + General Engineering"],
        "grade_distribution": {f"{g}th": [p["grade"] for p in profiles].count(g) for g in set(p["grade"] for p in profiles)},
        "experience_distribution": {p["full_name"]: f"Prog: {p['programming_experience']}, AI: {p['ai_ml_experience']}" for p in profiles},
        "shared_future_goals": [f"{p['full_name']}: {p['future_goal']}" for p in profiles],
        "combined_project_history": [f"{p['full_name']}: {p['toughest_project']}" for p in profiles],
        "combined_existing_ideas": [f"{p['full_name']}: {p['existing_project_idea']}" for p in profiles],
        "calibrated_group_difficulty": calibrated_diff,
        "complementary_skills_summary": {
            "combined_languages": list(set([lang.strip() for p in profiles for lang in p["programming_languages"].split(",") if lang.strip()])),
            "experience_range": f"Floor Baseline Level Matrix Score: {floor_score}"
        },
        "shared_interests": list(set.intersection(*map(set, [p["project_themes"] for p in profiles if p["project_themes"]]))) if len(profiles) > 1 else unique_themes
    }

# ---------------------------------------------------------------------------
# IDENTITY-FIRST DEEP LLM CORE INSIGHT PROMPT
# ---------------------------------------------------------------------------
def generate_master_reasoning_prompt(student_matrix: str, historical_pool: str, track_rules: str) -> str:
    return (
        "You are an Elite Research Director, Systems Architect, and Admissions Strategist at BetterMind Labs.\n"
        "Your role dictates that legendary high school engineering projects originate from a profound, Identity-First design engine, "
        "NOT surface-level text queries.\n\n"
        "EXECUTE INTERNAL REASONING PIPELINE PROCESS (STAGES 1 - 7):\n"
        "1. Extract Student Intelligence Profile signals (Domain, Quant Signals, Initiative, Entrepreneurship Core).\n"
        "2. Isolate a precise Primary Archetype (e.g., Financial Systems Builder, Quant Researcher, Climate Scientist, AI Founder, Healthcare Innovator, Policy Technologist).\n"
        "3. Mine the provided Historical Student Benchmarks dataset to identify foundational engineering patterns.\n"
        "4. Construct clear, domain-specific Project Opportunity maps.\n"
        "5. Generate EXACTLY 3 custom project structures representing: Research Track, Engineering Track, Product/Impact Track.\n"
        "6. Execute internal validation loops: ensure Domain, Career, and Narrative affinity scores are high.\n"
        "7. Apply the Admissions Filter Gate: Reject generic clones, basic chatbots, resume builders, study tools, or volunteer platform apps unless deeply customized by the profile.\n\n"
        "CRITICAL DOMAIN CONVERGENCE CONSTRAINT:\n"
        "All 3 tracks MUST operate inside the same primary resolved student domain (e.g., if the candidate is structured around Quant Finance, "
        "the Research track must target algorithmic market features, the Engineering track must evaluate high-performance computational pipelines, "
        "and the Product track must target SME or private capital structures). NEVER switch fields across tracks.\n\n"
        "TECHNICAL BOUNDARY DEFINITIONS:\n"
        "- Projects must be built to be completed within 18-20 hours of work over 3 weeks.\n"
        "- Every blueprint must embed a clear synthetic data pipeline strategy or specify public clean datasets (e.g., Kaggle, NOAA, SEC filings).\n"
        "- Every blueprint must feature an interactive framework UI model via Streamlit or Gradio for local validation loops.\n\n"
        "STRICT OUTPUT VALIDATION SCHEMA:\n"
        "Return exclusively a valid, parseable JSON block matching the structure below. Do not wrap with conversational prose.\n\n"
        "{\n"
        '  "ideas": [\n'
        "    {\n"
        '      "title": "[Track Specific Unique Technical Title]",\n'
        '      "problem_statement": "[Specific real-world granular technical friction parameters]. 3-WEEK MILESTONE BREAKDOWN: Week 1: [Specific local data pipeline / data initialization]. Week 2: [Specific model training or algorithmic optimization]. Week 3: [Streamlit/Gradio configuration and testing].",\n'
        '      "research_question": "[Precise testable inquiry checking parameter variations, hyperparameters, or mathematical changes]",\n'
        '      "why_it_matters": "INTERACTIVE DEMO STRATEGY: [Detailed description of interface widgets and real-time model evaluation triggers]. IMPACT: [Specific systemic value statement].",\n'
        '      "difficulty": "[Intermediate, Advanced, or Ambitious]",\n'
        '      "college_application_value": "[Deep narrative summary detailing why an elite university admissions committee will recognize high intellectual agency and technical curiosity].",\n'
        '      "project_strength_score": {\n'
        '        "research_depth": 8,\n'
        '        "real_world_impact": 9,\n'
        '        "technical_depth": 8,\n'
        '        "college_application_strength": 9\n'
        '      },\n'
        '      "matched_historical_project_title": "[Exact title string match extracted from the provided historical baseline context pool that shares structural alignment]",\n'
        '      "historical_match_justification": "[Profound analysis explaining problem, research, or execution intersection vectors between this new generation and previous success blueprints]"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"--- ACTIVE STUDENT IDENTITY CONFIGURATION MATRIX ---\n{student_matrix}\n\n"
        f"--- HISTORICAL REFERENCE BLUEPRINT DATASET POOL ---\n{historical_pool}\n\n"
        f"--- TARGET SEGMENT BOUNDARY RULES ---\n{track_rules}"
    )

# ---------------------------------------------------------------------------
# CORE REST FRAMEWORK ROUTING PORTS
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running", "active_records": len(TALLY_SERVICE.cache)}

@app.post("/generate", response_model=GenerateIdeasResponse)
def generate_ideas_external(profile: StudentProfile):
    if not groq_client or not EMBEDDING_MODEL:
        raise HTTPException(status_code=503, detail="Required AI Compute Engine or Core Embedder missing.")
        
    try:
        tally = TALLY_SERVICE.lookup_student(profile.email) or {}
        
        # Resolve and score student metrics
        resolved_matrix = {
            "grade": profile.grade, "city": profile.city, "declared_themes": profile.interest_areas,
            "toughest_completed_project": profile.toughest_project, "proposed_direction": profile.proposed_idea or "None",
            "college_aspiration": profile.preferred_college or "Top-Tier Engineering",
            "intended_major": tally.get("major", "Computer Science"),
            "career_trajectory": tally.get("future_goal", "Systems Engineering"),
            "programming_floor": tally.get("programming_experience", "Intermediate"),
            "ai_ml_floor": tally.get("ai_ml_experience", "Beginner"),
            "extracurriculars": tally.get("club_information", "N/A")
        }
        
        # Dense query construction for pre-generation mining
        search_query = (
            f"Themes: {profile.interest_areas} Major: {resolved_matrix['intended_major']} "
            f"Goal: {resolved_matrix['career_trajectory']} History: {profile.toughest_project} "
            f"Idea: {profile.proposed_idea or ''}"
        )
        historical_pool = execute_dense_retrieval(search_query, top_k=15)
        
        formatted_pool_text = "\n".join([
            f"Historical Project Reference Match:\n- Title: {p['title']}\n- Domain: {p['project_category']}\n- Context: {p['search_doc']}\n"
            for p in historical_pool
        ])
        
        sys_prompt = generate_master_reasoning_prompt(
            student_matrix=json.dumps(resolved_matrix, indent=2),
            historical_pool=formatted_pool_text,
            track_rules="Individual Mode Configuration: Generate 3 track variations inside the single matching domain."
        )
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sys_prompt}],
            response_format={"type": "json_object"},
            temperature=0.45,
            timeout=20.0
        )
        
        parsed_output = json.loads(response.choices[0].message.content)
        raw_ideas = parsed_output.get("ideas", [])
        
        processed_ideas: List[ProjectIdea] = []
        for item in raw_ideas[:3]:
            # Structural re-ranking layer verification
            matched_title = item.get("matched_historical_project_title", "").strip().lower()
            matching_historical_record = next((p for p in historical_pool if p["title"].lower() == matched_title), None)
            
            similar_projects_list = []
            if matching_historical_record:
                similar_projects_list.append(SimilarProject(
                    title=matching_historical_record["title"],
                    student_name=matching_historical_record["student_name"],
                    description=matching_historical_record["description"],
                    github_url=matching_historical_record["github_url"],
                    youtube_url=matching_historical_record["youtube_url"],
                    project_category=matching_historical_record["project_category"],
                    similarity_score=matching_historical_record["similarity_score"],
                    matching_reason=item.get("historical_match_justification", "Shares core conceptual problem architecture.")
                ))
            
            # Pad similar projects slot using dense retrieval layer scores to maintain schema requirements
            for fallback in historical_pool[:3]:
                if len(similar_projects_list) >= 3:
                    break
                if matching_historical_record and fallback["title"] == matching_historical_record["title"]:
                    continue
                similar_projects_list.append(SimilarProject(
                    title=fallback["title"],
                    student_name=fallback["student_name"],
                    description=fallback["description"],
                    github_url=fallback["github_url"],
                    youtube_url=fallback["youtube_url"],
                    project_category=fallback["project_category"],
                    similarity_score=fallback["similarity_score"],
                    matching_reason="Aligned technological footprint matching student context vectors."
                ))

            scores = item.get("project_strength_score", {})
            processed_ideas.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Advanced Context Framework"),
                problem_statement=item.get("problem_statement", "Detailed problem bounds undefined."),
                research_question=item.get("research_question", "Inquiry tracking variant adjustments."),
                why_it_matters=item.get("why_it_matters", "Establishes systemic structural agency."),
                difficulty=item.get("difficulty", "Advanced"),
                college_application_value=item.get("college_application_value", "Demonstrates deep non-cliché intellectual initiative."),
                similar_projects=similar_projects_list[:3],
                project_strength_score=ProjectStrengthScore(
                    research_depth=scores.get("research_depth", 8),
                    real_world_impact=scores.get("real_world_impact", 8),
                    technical_depth=scores.get("technical_depth", 8),
                    college_application_strength=scores.get("college_application_strength", 9)
                )
            ))
            
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        SESSIONS[session_id] = {
            "source": "external", "profile": profile.model_dump(),
            "ideas": [i.model_dump() for i in processed_ideas], "niche_ideas": []
        }
        return GenerateIdeasResponse(session_id=session_id, ideas=processed_ideas)
    except Exception as e:
        logger.error(f"Critical execution error inside /generate pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-bml", response_model=GenerateIdeasResponse)
def generate_ideas_bettermind(request: BettermindGenerateRequest):
    if request.mode != "bettermind":
        raise HTTPException(status_code=400, detail="Invalid application execution mode parameter.")
    if len(request.participants) + 1 > 3:
        raise HTTPException(status_code=400, detail="Group matrix configuration limits exceeded (Max 3 members).")
    if not groq_client or not EMBEDDING_MODEL:
        raise HTTPException(status_code=503, detail="Required Core Model configurations unassigned.")

    try:
        resolved_profiles = [resolve_and_score_profile(request.primary_student)]
        for p in request.participants:
            resolved_profiles.append(resolve_and_score_profile(p))

        # Build synthesized graphs and optimization boundary clusters
        group_identity_graph = compile_identity_graph(resolved_profiles)
        
        # Dense query mapping from unified cluster vectors
        search_query = " ".join(group_identity_graph.get("combined_interests", [])) + " " + " ".join(group_identity_graph.get("combined_existing_ideas", []))
        historical_pool = execute_dense_retrieval(search_query, top_k=15)
        
        formatted_pool_text = "\n".join([
            f"Historical Project Reference Match:\n- Title: {p['title']}\n- Domain: {p['project_category']}\n- Context: {p['search_doc']}\n"
            for p in historical_pool
        ])

        sys_prompt = generate_master_reasoning_prompt(
            student_matrix=json.dumps(group_identity_graph, indent=2),
            historical_pool=formatted_pool_text,
            track_rules=(
                "Group Mode Cluster Rules: Synthesize an optimized collaborative intersection architecture framework. "
                "Ensure separate execution paths (Data Engineering vs Algorithmic Modeling vs Interface Prototyping) "
                "exist across the 3 tracks while locking all tracks within the primary unified cluster domain space."
            )
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sys_prompt}],
            response_format={"type": "json_object"},
            temperature=0.42,
            timeout=20.0
        )

        parsed_output = json.loads(response.choices[0].message.content)
        raw_ideas = parsed_output.get("ideas", [])

        processed_ideas: List[ProjectIdea] = []
        for item in raw_ideas[:3]:
            matched_title = item.get("matched_historical_project_title", "").strip().lower()
            matching_historical_record = next((p for p in historical_pool if p["title"].lower() == matched_title), None)

            similar_projects_list = []
            if matching_historical_record:
                similar_projects_list.append(SimilarProject(
                    title=matching_historical_record["title"],
                    student_name=matching_historical_record["student_name"],
                    description=matching_historical_record["description"],
                    github_url=matching_historical_record["github_url"],
                    youtube_url=matching_historical_record["youtube_url"],
                    project_category=matching_historical_record["project_category"],
                    similarity_score=matching_historical_record["similarity_score"],
                    matching_reason=item.get("historical_match_justification", "Shares collaborative engineering footprints.")
                ))

            for fallback in historical_pool[:3]:
                if len(similar_projects_list) >= 3:
                    break
                if matching_historical_record and fallback["title"] == matching_historical_record["title"]:
                    continue
                similar_projects_list.append(SimilarProject(
                    title=fallback["title"],
                    student_name=fallback["student_name"],
                    description=fallback["description"],
                    github_url=fallback["github_url"],
                    youtube_url=fallback["youtube_url"],
                    project_category=fallback["project_category"],
                    similarity_score=fallback["similarity_score"],
                    matching_reason="Aligned infrastructure vector tracking cluster interest metrics."
                ))

            scores = item.get("project_strength_score", {})
            processed_ideas.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Synthesized Group Architecture"),
                problem_statement=item.get("problem_statement", "Collaborative problem parameter undefined."),
                research_question=item.get("research_question", "Inquiry tracking group optimization models."),
                why_it_matters=item.get("why_it_matters", "Establishes distributed system configuration rules."),
                difficulty=group_identity_graph["calibrated_group_difficulty"],
                college_application_value=item.get("college_application_value", "Highlights collaborative infrastructure engineering and modular component control."),
                similar_projects=similar_projects_list[:3],
                project_strength_score=ProjectStrengthScore(
                    research_depth=scores.get("research_depth", 8),
                    real_world_impact=scores.get("real_world_impact", 8),
                    technical_depth=scores.get("technical_depth", 8),
                    college_application_strength=scores.get("college_application_strength", 9)
                )
            ))

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        SESSIONS[session_id] = {
            "source": "bettermind",
            "participants": [p.model_dump() for p in request.participants],
            "resolved_profiles": resolved_profiles,
            "group_profile": group_identity_graph,
            "ideas": [i.model_dump() for i in processed_ideas],
            "niche_ideas": []
        }
        return GenerateIdeasResponse(session_id=session_id, ideas=processed_ideas)
    except Exception as e:
        logger.error(f"Critical execution error inside /generate-bml pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/niche-down", response_model=GenerateIdeasResponse)
def niche_down_endpoint(payload: NicheDownRequest):
    if payload.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Target tracking session identifier missing.")
    
    session_data = SESSIONS[payload.session_id]
    all_ideas = session_data.get("ideas", []) + session_data.get("niche_ideas", [])
    base_idea = next((item for item in all_ideas if item["id"] == payload.idea_id), None)
    
    if not base_idea:
        raise HTTPException(status_code=400, detail="Target baseline project index node invalid.")

    if not groq_client:
        raise HTTPException(status_code=503, detail="AI Inference engine client unassigned.")

    system_prompt = (
        "You are an Elite Research Director at BetterMind Labs specializing in advanced academic hyper-specialization.\n"
        "Your task is to take a chosen project blueprint and generate exactly 3 highly refined, localized iterations "
        "that pivot into specialized sub-problem contexts. Do NOT provide cosmetic variations.\n\n"
        "Format output strictly inside a parseable JSON block matching this schema:\n"
        "{\n"
        '  "ideas": [\n'
        '    { "title": "[Niche Technical Title]", "problem_statement": "[Hyper-localized technical problem bounds]. 3-WEEK MILESTONE BREAKDOWN: Week 1: [Data pipeline instantiation]. Week 2: [Model training/tuning]. Week 3: [Interactive deployment checks].", "research_question": "[Precise testable academic inquiry]", "why_it_matters": "INTERACTIVE DEMO STRATEGY: [Widget configuration parameters]. IMPACT: [Systemic localized reduction of friction].", "difficulty": "Ambitious", "college_application_value": "[Deep narrative summary of intellectual curiosity]", "project_strength_score": {"research_depth":9,"real_world_impact":8,"technical_depth":8,"college_application_strength":9} }\n'
        '  ]\n'
        "}"
    )
    user_prompt = f"Refine and hyper-specialize this concept: Title: {base_idea.get('title')}. Problem Space: {base_idea.get('problem_statement')}. Research Inquiry: {base_idea.get('research_question')}."

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.45,
            timeout=15.0
        )
        parsed = json.loads(response.choices[0].message.content)
        raw_niche = parsed.get("ideas", [])
        
        processed: List[ProjectIdea] = []
        for item in raw_niche[:3]:
            scores = item.get("project_strength_score", {})
            processed.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Specialized Variant Core"),
                problem_statement=item.get("problem_statement", base_idea.get("problem_statement")),
                research_question=item.get("research_question", base_idea.get("research_question")),
                why_it_matters=item.get("why_it_matters", base_idea.get("why_it_matters")),
                difficulty="Ambitious",
                college_application_value=item.get("college_application_value", base_idea.get("college_application_value")),
                similar_projects=base_idea.get("similar_projects", []),
                project_strength_score=ProjectStrengthScore(
                    research_depth=scores.get("research_depth", 9),
                    real_world_impact=scores.get("real_world_impact", 8),
                    technical_depth=scores.get("technical_depth", 8),
                    college_application_strength=scores.get("college_application_strength", 9)
                )
            ))
        
        while len(processed) < 3:
            processed.append(processed[0])
            
        session_data["niche_ideas"] = [i.model_dump() for i in processed]
        return GenerateIdeasResponse(session_id=payload.session_id, ideas=processed)
    except Exception as e:
        logger.error(f"Error inside niche-down execution pipeline loops: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
def get_session_state(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Active context tracking index mismatch.")
    return SESSIONS[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)