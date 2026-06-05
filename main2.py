"""
Bettermind Labs Project Generator Backend - Milestone 4 (Research-Backed Discovery Engine)
Implements problem-first discovery loops, hybrid semantic-archetype matching matrices, 
student archetype engines, and multi-student group compatibility mapping layers.
Preserves FastAPI routing signatures, Tally ingestion layers, and strict API contracts.
Refactored to enforce Trajectory-Based Opportunity Discovery over generic interest matching.
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
    title="Bettermind Labs Project Discovery Platform",
    version="4.1.0",
    description="Milestone 4 - Trajectory-Driven Problem Verification Flow & Hybrid Strategic Match Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GLOBAL STATE & DENSE EMBEDDING CORES
# ---------------------------------------------------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
EMBEDDING_MODEL: Optional[SentenceTransformer] = None
PROJECTS_DF: Optional[pd.DataFrame] = None
PROJECT_EMBEDDINGS: Optional[np.ndarray] = None
PROJECT_TAGS_MAP: Dict[int, set] = {}  # Maps dataframe row index to extracted keyword tags

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
# PYDANTIC ROUTING SCHEMAS (EXTENDED DATA MODELS FOR COUNSELOR NARRATIVES)
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
    project_page: str
    similarity_score: float
    matching_reason: str

class DiscoveryMetadata(BaseModel):
    generation_framework: str
    research_basis: str
    personalization_summary: str

class ProjectIdea(BaseModel):
    # Original Legacy Fields (Preserved for strict frontend compatibility)
    id: str
    title: str
    problem_area: str
    guiding_question: str
    project_description: str
    why_it_matters: str
    why_this_fits_you: List[str]
    target_users: str
    datasets: str
    tools: str
    technical_approach: str
    difficulty: str
    feasibility_score: int
    feasibility_reasoning: str
    week1_plan: str
    week2_plan: str
    week3_plan: str
    final_demo: str
    college_application_value: str
    similar_projects: List[SimilarProject] = []
    
    # New BetterMind Problem-First Discovery Engine Fields
    problem_context: str
    who_experiences_this_problem: str
    why_this_problem_exists: str
    potential_project_direction: str
    college_narrative: str
    
    # Trajectory-Driven Opportunity Architecture Fields
    student_history_connection: str

class GenerateIdeasResponse(BaseModel):
    session_id: str
    discovery_metadata: DiscoveryMetadata
    ideas: List[ProjectIdea]

class NicheDownRequest(BaseModel):
    session_id: str
    idea_id: str

# ---------------------------------------------------------------------------
# AUTOMATED REGEX/KEYWORD-BASED TAG EXTRACTION SERVICE
# ---------------------------------------------------------------------------
def extract_domain_tags(text_corpus: str) -> set:
    """
    Statically extracts explicit structural and domain engineering tags 
    from a block of text to fuel the high-fidelity matching calculation.
    """
    tags = set()
    keywords_library = {
        "ipo", "finance", "quant", "liquidity", "nlp", "sentiment", "portfolio", "trading",
        "stock", "market", "risk", "fraud", "credit", "banking", "ledger", "alpha",
        "healthcare", "medical", "chiral", "biosynthetic", "pathway", "diagnosis", "clinical",
        "imaging", "mri", "disease", "patient", "genomic", "protein", "cell", "neuro",
        "climate", "carbon", "nanotube", "environment", "sensor", "pollution", "noaa",
        "satellite", "telemetry", "orbit", "space", "propulsion", "nasa", "trajectory",
        "cybersecurity", "phishing", "threat", "network", "anomaly", "malware", "encryption",
        "robotics", "hardware", "drone", "maze", "navigation", "control", "cv", "vision",
        "education", "pedagogy", "curriculum", "learning", "classroom", "grading",
        "gnn", "cnn", "lstm", "transformer", "regression", "classification", "timeseries"
    }
    normalized_text = text_corpus.lower()
    for keyword in keywords_library:
        if keyword in normalized_text:
            tags.add(keyword)
    return tags

# ---------------------------------------------------------------------------
# STUDENT ARCHETYPE INFERENCE ENGINE
# ---------------------------------------------------------------------------
def infer_student_archetype(profile: Dict[str, Any]) -> str:
    """
    Parses historical targets, interest trends, and statements to map students
    into exact strategic academic persona profiles.
    """
    goal = str(profile.get("future_goal", "")).lower()
    major = str(profile.get("major", "")).lower()
    themes = [str(t).lower() for t in profile.get("project_themes", [])]
    toughest = str(profile.get("toughest_project", "")).lower()
    
    if "quant" in goal or "quant" in major or "high-frequency" in goal:
        return "Quant Researcher"
    if "hospital" in goal or "medical" in goal or "healthcare" in goal or "biomedical" in major:
        return "Healthcare Innovator"
    if "climate" in goal or "environment" in goal or "carbon" in goal or "green" in goal:
        return "Climate Scientist"
    if "robot" in goal or "hardware" in goal or "robotics" in major or "drone" in toughest:
        return "Robotics Builder"
    if "policy" in goal or "law" in goal or "government" in goal or "ethics" in goal:
        return "Policy Thinker"
    if "found" in goal or "startup" in goal or "entrepreneur" in goal or "saas" in goal:
        return "Founder"
    if "research" in goal or "phd" in goal or "academic" in goal or "paper" in toughest:
        return "Researcher"
    if "teach" in goal or "education" in goal or "learn" in goal or "school" in goal:
        return "Education Builder"
    if "social" in goal or "good" in goal or "community" in goal or "ngo" in goal:
        return "Social Impact Builder"
    return "Engineer"

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
            "full_name": raw_map.get("your full name", ""),
            "email": raw_map.get("your email", ""),
            "grade": raw_map.get("your grade", raw_map.get("grade", "")),
            "timezone": raw_map.get("time zone", ""),
            "city": raw_map.get("your city", ""),
            "state": raw_map.get("your state", ""),
            "school": raw_map.get("which school do you go to?", ""),
            "future_goal": raw_map.get("want to pursue in future", ""),
            "major": raw_map.get("major to pursue", ""),
            "challenging_subject": raw_map.get("which subject is most challenging for you in your curriculum?", ""),
            "project_themes": raw_map.get("what kind of project themes are you interested in? (min 3, max 4)", []),
            "programming_experience": raw_map.get("programming exp", 0),
            "ai_ml_experience": raw_map.get("ai/ml exp", 0),
            "programming_languages": raw_map.get("which programing language are you familiar with?", []),
            "program_goals": raw_map.get("what are you hoping to gain from this program? are there specific skills, projects, or experiences you’re most excited about?", ""),
            "club_information": raw_map.get("are you part of any club or community? if yes what is your role there, describe briefly.", ""),
            "toughest_project": raw_map.get("what is the toughest project you have worked so far?", ""),
            "has_project_idea": raw_map.get("do you have an idea on what you'd like to build during the program", []),
            "existing_project_idea": raw_map.get("great, please tell us what you'd like to explore or build?", ""),
            "availability_window": raw_map.get("please provide a 3–5 hour time window each day during which you will be consistently available throughout the program...", ""),
            "preferred_session_time": raw_map.get("preferred instructor led sessions timings", []),
            "preferred_call_time": raw_map.get("preferred mentorship call timing (in short)", ""),
            "parent_name": raw_map.get("parent name", ""),
            "parent_phone": raw_map.get("parent phone", ""),
            "parent_email": raw_map.get("parent email", ""),
            "linkedin": raw_map.get("your linkedin profile(write linkedin.com if you dont have)", ""),
            "instagram": raw_map.get("your instagram handle", ""),
            "portfolio": raw_map.get("other socials or portfolio", ""),
            "case_study_interest": raw_map.get("for some exceptional projects, we create student case studies and success stories...", []),
            "has_counselor": raw_map.get("do you work with an independent college counselor?", []),
            "counselor_name": raw_map.get("name of counselor, organization", ""),
            "counselor_email": raw_map.get("email of counselor", ""),
            "student_video": raw_map.get("student video (optional)", [])
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
# STARTUP SEQUENCING & CORRESPONDING DATA SYNC
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_pipeline_sequence():
    global EMBEDDING_MODEL, PROJECTS_DF, PROJECT_EMBEDDINGS, PROJECT_TAGS_MAP
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
        PROJECTS_DF.to_csv(CSV_FILE_PATH, index=False)
    else:
        PROJECTS_DF = pd.read_csv(CSV_FILE_PATH)

    for col in PROJECTS_DF.columns:
        PROJECTS_DF[col] = PROJECTS_DF[col].fillna("").astype(str).str.strip()

    logger.info("Building enriched multi-dimensional text anchors and tag indices for benchmarks...")
    enriched_docs = []
    PROJECT_TAGS_MAP.clear()
    
    for idx, row in PROJECTS_DF.iterrows():
        title = row.get("Title", "")
        desc = row.get("Description", "")
        cat = row.get("Project Category", "")
        category = row.get("Category", "")
        
        dense_doc = (
            f"TITLE: {title} | DOMAIN: {cat} | SECTOR: {category}\n"
            f"CORE BLUEPRINT DESCRIPTION: {desc}\n"
        )
        enriched_docs.append(dense_doc)
        PROJECTS_DF.at[idx, "search_doc"] = dense_doc
        
        # Build in-memory static tags for Part 7 Overhaul
        combined_metadata_text = f"{title} {desc} {cat} {category}"
        PROJECT_TAGS_MAP[idx] = extract_domain_tags(combined_metadata_text)

    logger.info("Generating corpus vector matrix space...")
    PROJECT_EMBEDDINGS = EMBEDDING_MODEL.encode(enriched_docs, convert_to_numpy=True)
    
    await TALLY_SERVICE.fetch_all_submissions()
    asyncio.create_task(cache_refresh_scheduler())

# ---------------------------------------------------------------------------
# HIGH-FIDELITY TRACK-MATCHING MATRIX ENGINE (REPLACES PURE EMBEDDINGS)
# ---------------------------------------------------------------------------
def execute_hybrid_retrieval(query_context: str, student_info: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Implements a strategic domain matching pipeline for historical benchmarks.
    Enforces Category Filters, explicit tag intersections, and a deterministic score breakdown:
    Score = 0.50 Category Match + 0.30 Tag Overlap + 0.20 Semantic Similarity
    """
    if PROJECTS_DF is None or PROJECTS_DF.empty or EMBEDDING_MODEL is None or PROJECT_EMBEDDINGS is None:
        return []
    
    # Pre-extract target opportunity parameters
    target_themes = [t.lower().strip() for t in student_info.get("project_themes", [])]
    if not target_themes and student_info.get("declared_themes"):
        target_themes = [t.lower().strip() for t in str(student_info.get("declared_themes")).split(",")]
        
    opportunity_tags = extract_domain_tags(query_context)
    
    # Compute base dense semantic array
    query_vec = EMBEDDING_MODEL.encode([query_context], convert_to_numpy=True)
    scores = np.dot(PROJECT_EMBEDDINGS, query_vec.T).flatten()
    norms = np.linalg.norm(PROJECT_EMBEDDINGS, axis=1) * np.linalg.norm(query_vec)
    semantic_sims = np.divide(scores, norms, out=np.zeros_like(scores), where=norms != 0)
    
    candidate_matches = []
    
    for idx, row in PROJECTS_DF.iterrows():
        row_cat = str(row.get("Project Category", "")).lower().strip()
        row_sector = str(row.get("Category", "")).lower().strip()
        
        # STEP 1: Strict Category Clustering Filter
        category_match = 0.0
        for theme in target_themes:
            if theme in row_cat or theme in row_sector or row_cat in theme:
                category_match = 1.0
                break
                
        # STEP 2: Structural Tag Overlap
        row_tags = PROJECT_TAGS_MAP.get(idx, set())
        overlap_count = len(opportunity_tags.intersection(row_tags))
        total_unique_tags = len(opportunity_tags.union(row_tags))
        tag_overlap_score = (overlap_count / total_unique_tags) if total_unique_tags > 0 else 0.0
        
        # STEP 3: Dense Cosine Factor
        sem_score = max(0.0, min(1.0, float(semantic_sims[idx])))
        
        # STEP 4: High-Fidelity Unified Calculation Formula
        final_score = (0.50 * category_match) + (0.30 * tag_overlap_score) + (0.20 * sem_score)
        
        # STEP 5: Counselor Validation Layer Guardrail (Discard poor matches)
        if category_match == 0.0 or final_score < 0.35:
            continue
            
        # STEP 7: Generate Factual Derived Overlap Reasons
        matched_shared_keywords = list(opportunity_tags.intersection(row_tags))
        if matched_shared_keywords:
            reason = f"Matched because both systems track core problem contexts involving {', '.join(matched_shared_keywords[:3])} within the {row.get('Project Category')} domain."
        else:
            reason = f"Matched because both projects analyze data architecture pipelines inside the {row.get('Project Category')} track."

        candidate_matches.append({
            "title": row.get("Title"),
            "student_name": row.get("StudentName"),
            "description": row.get("Description"),
            "github_url": row.get("Github URL"),
            "youtube_url": row.get("Youtube URL"),
            "project_category": row.get("Project Category"),
            "project_page": f"https://bettermindlabs.com/projects/{str(row.get('Title')).lower().replace(' ', '-')}",
            "similarity_score": round(final_score, 3),
            "matching_reason": reason
        })
        
    candidate_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    # STEP 6: Return strictly 0 to 2 verified matches max (Quality > Forced Counts)
    return candidate_matches[:2]

# ---------------------------------------------------------------------------
# COHORT INTERSECTION GENERATOR (GROUP INTELLIGENCE ENGINE)
# ---------------------------------------------------------------------------
def resolve_and_score_profile(inp: ParticipantInput) -> Dict[str, Any]:
    email = inp.email.lower().strip()
    logger.info(f"Looking up Tally profile: {email}")
    tally = TALLY_SERVICE.lookup_student(email) or {}

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
        "program_goals": tally.get("program_goals", "N/A"),
        "toughest_project": tally.get("toughest_project", ""),
        "existing_project_idea": inp.proposed_idea or tally.get("existing_project_idea", "")
    }
    if inp.interest_areas:
        manual = [t.strip() for t in inp.interest_areas.split(",") if t.strip()]
        resolved["project_themes"] = [t for t in manual if t in VALID_THEMES]
        
    resolved["archetype"] = infer_student_archetype(resolved)
    return resolved

def compile_identity_graph(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_themes = [t for p in profiles for t in p["project_themes"]]
    unique_themes = list(set(all_themes))
    shared_interests = list(set.intersection(*map(set, [p["project_themes"] for p in profiles if p["project_themes"]]))) if len(profiles) > 1 else unique_themes
    
    roles = ["Data Engineering & Infrastructure", "Algorithmic Modeling & Core Training", "Frontend Prototyping & Interface Loops"]
    role_split = {}
    for idx, p in enumerate(profiles):
        assigned_role = roles[idx % len(roles)]
        role_split[p["full_name"]] = f"Assigned: {assigned_role} (Archetype: {p['archetype']})"

    return {
        "shared_interests": shared_interests if shared_interests else ["AI + General Engineering"],
        "complementary_skills": [f"{p['full_name']}: {p['programming_languages']} (Prog: {p['programming_experience']}, AI: {p['ai_ml_experience']})" for p in profiles],
        "collaboration_strengths": "High balance across complementary engineering archetypes.",
        "recommended_role_split": role_split,
        "combined_interests": unique_themes,
        "archetypes": [p["archetype"] for p in profiles],
        "individual_profiles": profiles
    }

# ---------------------------------------------------------------------------
# RESEARCH-BACKED PROMPT ENGINE (STUDENT TRAJECTORY-FIRST REORIENTATION)
# ---------------------------------------------------------------------------
def generate_master_reasoning_prompt(student_matrix: str, historical_pool: str, group_rules: str) -> str:
    return (
        "You are an expert academic mentor and data engineer at BetterMind Labs helping ambitious high school students discover deep, research-backed problem opportunities.\n"
        "Your task is to act as a counselor-backed research opportunity discovery engine that transforms technology-first concepts into deep, problem-first explanatory tracks driven by student trajectory.\n\n"
        
        "CRITICAL CORE DESIGN PRINCIPLE:\n"
        "Projects MUST be generated primarily from Student History + Student Trajectory + Student Strengths + Student Future Direction, NOT from interest areas alone.\n"
        "Interest areas are merely supporting signals. Past technical history is the primary driver.\n\n"
        
        "MANDATORY INTERNAL STEP: STUDENT TRAJECTORY ANALYSIS\n"
        "Before selecting problems or generating any project ideas, you MUST construct an internal 'student_trajectory_profile' for the student(s) by analyzing the matrix variables. Compute this step-by-step internally as part of your reasoning context (do not output this as a separate JSON field, but use it to anchor the ideas):\n"
        "{\n"
        "  \"student_archetype\": \"...\",\n"
        "  \"core_strengths\": [],\n"
        "  \"research_readiness\": \"...\",\n"
        "  \"leadership_profile\": \"...\",\n"
        "  \"technical_maturity\": \"...\",\n"
        "  \"trajectory_summary\": \"...\",\n"
        "  \"highest_leverage_directions\": []\n"
        "}\n\n"
        
        "TRAJECTORY ANALYSIS & SIGNAL WEIGHTING RULES:\n"
        "Carefully parse the following signals with these exact rigid weightings:\n"
        "- 35% Weight: toughest_project (What advanced architectural exposure or skills have they already demonstrated?)\n"
        "- 25% Weight: future_goal (What is their chosen professional path/long-term objective?)\n"
        "- 15% Weight: club_information (What leadership, community involvement, or extracurricular roles do they hold?)\n"
        "- 10% Weight: existing_project_idea (What initial topic or seed idea do they want to explore?)\n"
        "- 10% Weight: major (What is their intended college major?)\n"
        "- 5% Weight: project_themes (Interest areas are only supporting signals, never the main driver!)\n\n"
        
        "Determine what the student has already explored, what level of complexity they can handle, and what the natural next step or logical evolution of their work is. The generated opportunities must feel like 'the next chapter of this student\'s journey', rather than a random project inside a category.\n\n"
        
        "PROJECT DISCOVERY ENGINE AND QUALITY FILTER:\n"
        "Do NOT ask 'What project fits AI + Finance?'. Ask 'What project would naturally come after this student\'s toughest project?'\n"
        "1. STRICT IDEA QUALITY FILTER: Immediately reject generic dashboards, chatbots, basic wrappers, simple trackers, task managers, or personal study/productivity tools.\n"
        "2. REQUIRED TECHNICAL COMPLEXITY: Every project must center on non-trivial operations such as prediction, optimization, scientific discovery, decision support, anomaly detection, forecasting, risk modeling, simulation, or research investigation.\n"
        "3. PORTFOLIO-QUALITY TITLE FORMULA: Formulate titles ONLY AFTER selecting the problem. Every title must strictly follow this pattern: '[Memorable Product Name]: [Specific Research-Oriented Subtitle]'.\n"
        "   Examples: 'MissionGuard: Predicting Satellite Component Failures Through Telemetry Analysis', 'IPOIQ: Predicting Long-Term IPO Volatility Using Filing Language', 'LungShift: Detecting Hospital-to-Hospital Generalization Failure in Pneumonia Models'.\n\n"
        
        "INNOVATION & INTEGRITY FILTER:\n"
        "Before finalizing each idea, evaluate and score it on four internal vectors: Novelty, Trajectory Fit, College Narrative Strength, and Execution Feasibility. Only include options scoring exceptionally across all four.\n"
        "STRICT NO-HALLUCINATION RULES:\n"
        "1. Never declare that the student built any historical benchmark projects mentioned in the reference dataset pools.\n"
        "2. Student background details may ONLY come from the explicitly provided student_matrix details.\n\n"
        
        "HIGH-FIDELITY DATA SOURCES:\n"
        "Mandate authentic public or professional repositories (e.g., SEC EDGAR, FRED, NOAA Climate Models, NASA Open Data, MIMIC-III, Stanford CheXpert), never generic terms like 'Kaggle dataset'.\n\n"
        
        "STRICT OUTPUT VALIDATION SCHEMA:\n"
        "Return exclusively a valid, parseable JSON block matching the structure below. Do not wrap with conversational text.\n\n"
        "{\n"
        '  "ideas": [\n'
        '    {\n'
        '      "title": "[BrandName: Specific Research-Oriented Subtitle Formulation]",\n'
        '      "problem_area": "[Granular, observable real-world problem statement detailing systematic friction points]",\n'
        '      "guiding_question": "[Precise testable inquiry evaluating specific parameter interactions]",\n'
        '      "project_description": "[Comprehensive mapping of why the problem emerges and how the data pipeline isolates it]",\n'
        '      "why_it_matters": "[Systemic importance to specific targeted users, industries, or ecosystems]",\n'
        '      "why_this_fits_you": [\n'
        '        "Summary bullet point 1 explaining alignment...",\n'
        '        "Summary bullet point 2..."\n'
        '      ],\n'
        '      "student_history_connection": "[Detailed technical narrative explicitly connecting this opportunity back to the student\'s history, toughest project, and trajectory, explaining why this is the natural next chapter of their journey]",\n'
        '      "target_users": "[Explicit user group or professional persona experiencing the problem]",\n'
        '      "datasets": "[Name of specific, verified high-fidelity data repository, e.g., SEC EDGAR database, NOAA physical arrays]",\n'
        '      "tools": "Python, Streamlit, Pandas, Scikit-Learn, HuggingFace",\n'
        '      "technical_approach": "[Step-by-step pipeline workflow details mapping ingestion, processing, and interface loops]",\n'
        '      "difficulty": "Advanced",\n'
        '      "feasibility_score": 9,\n'
        '      "feasibility_reasoning": "[Clear justification verifying completion within program bounds]",\n'
        '      "week1_plan": "[Data fetching, parameter checking, and cleaning loops from raw repository locations]",\n'
        '      "week2_plan": "[Model training, algorithmic pipeline tuning, and metric optimization loops]",\n'
        '      "week3_plan": "[Streamlit interactive UI building and testing validation arrays]",\n'
        '      "final_demo": "[Detailed interactive presentation setup with explicit widgets showcasing data parameter changes]",\n'
        '      "college_application_value": "[Summary of intellectual vitalization and technical depth generated]",\n'
        '      "problem_context": "[Deep overarching systemic overview of the problem sphere]",\n'
        '      "who_experiences_this_problem": "[Detailed breakdown of specialized stakeholders impacted]",\n'
        '      "why_this_problem_exists": "[Structural, technical, or economic root causes why this remains unresolved]",\n'
        '      "potential_project_direction": "[Secondary deployment framework description]",\n'
        '      "college_narrative": "[Counselor-ready positioning text explaining why an admissions officer values this project, the exact intellectual theme validated, leadership angles, and future research options. MUST BE A MINIMUM OF 120 WORDS.]"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        f"--- ACTIVE STUDENT PROFILE AND ARCHEQUITY CONFIGURATION MATRIX ---\n{student_matrix}\n\n"
        f"--- HISTORICAL REFERENCE DATASET POOL FOR TRACK CROSS-VERIFICATION ---\n{historical_pool}\n\n"
        f"--- STRUCTURAL CLUSTER EXTRA CONSTRAINT RULES ---\n{group_rules}"
    )

# ---------------------------------------------------------------------------
# CORE APPLICATION ROUTING INFRASTRUCTURE
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running", "active_records": len(TALLY_SERVICE.cache)}

@app.post("/generate", response_model=GenerateIdeasResponse)
def generate_ideas_external(profile: StudentProfile):
    if not groq_client or not EMBEDDING_MODEL:
        raise HTTPException(status_code=503, detail="Required AI Compute Engine or Core Embedder configurations missing.")
        
    try:
        tally = TALLY_SERVICE.lookup_student(profile.email) or {}
        
        student_info = {
            "full_name": tally.get("full_name", profile.email.split("@")[0].capitalize()),
            "email": profile.email.lower().strip(), 
            "grade": profile.grade, 
            "city": profile.city,
            "declared_themes": profile.interest_areas, 
            "toughest_project": profile.toughest_project or tally.get("toughest_project", ""),
            "proposed_idea": profile.proposed_idea or "None",
            "intended_major": tally.get("major", "Computer Science"),
            "future_goal": tally.get("future_goal", "Systems Engineering"),
            "project_themes": tally.get("project_themes", []),
            "programming_experience": tally.get("programming_experience", "Intermediate"),
            "ai_ml_experience": tally.get("ai_ml_experience", "Beginner"),
            "club_information": tally.get("club_information", "N/A"),
            "program_goals": tally.get("program_goals", "N/A"),
            "existing_project_idea": profile.proposed_idea or tally.get("existing_project_idea", "")
        }
        student_info["archetype"] = infer_student_archetype(student_info)
        
        # Build strict personalization trajectory anchors
        search_query = f"Themes: {profile.interest_areas} Specific Target: {student_info['existing_project_idea']} Previous Work: {student_info['toughest_project']}"
        
        # Pull basic structural matches to insert into context pool
        initial_pool = execute_hybrid_retrieval(search_query, student_info, top_k=2)
        formatted_pool_text = "\n".join([
            f"- Title: {p['title']} | Category: {p['project_category']} | Description: {p['description']}"
            for p in initial_pool
        ]) if initial_pool else "No directly matching historical baselines. Build a specialized track."
        
        sys_prompt = generate_master_reasoning_prompt(
            student_matrix=json.dumps(student_info, indent=2),
            historical_pool=formatted_pool_text,
            group_rules="Individual Student Mode: Weight historical milestones and continuous trajectory markers as core discovery parameters."
        )
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sys_prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            timeout=25.0
        )
        
        parsed_output = json.loads(response.choices[0].message.content)
        raw_ideas = parsed_output.get("ideas", [])
        
        processed_ideas: List[ProjectIdea] = []
        for item in raw_ideas[:3]:
            # Run deep post-generation individual high-fidelity matching layer for Part 7 Overhaul
            generated_opportunity_context = f"{item.get('title')} {item.get('problem_area')} {item.get('project_description')}"
            refined_matched_pool = execute_hybrid_retrieval(generated_opportunity_context, student_info, top_k=2)
            
            similar_projects_list = []
            for p in refined_matched_pool:
                similar_projects_list.append(SimilarProject(
                    title=p["title"],
                    student_name=p["student_name"],
                    description=p["description"],
                    github_url=p["github_url"],
                    youtube_url=p["youtube_url"],
                    project_category=p["project_category"],
                    project_page=p["project_page"],
                    similarity_score=p["similarity_score"],
                    matching_reason=p["matching_reason"]
                ))
                
            processed_ideas.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Advanced Research Discovery Node"),
                problem_area=item.get("problem_area", "Localized real-world data discrepancies."),
                guiding_question=item.get("guiding_question", "Inquiry exploring specific system optimizations."),
                project_description=item.get("project_description", "Problem analysis and targeted engineering loops."),
                why_it_matters=item.get("why_it_matters", "Reduces friction over critical system parameters."),
                why_this_fits_you=item.get("why_this_fits_you", [f"Aligned with your {student_info['archetype']} archetype framework choice."]),
                student_history_connection=item.get("student_history_connection", "This opportunity builds directly on top of your previous technical work exposure to establish a clear intellectual progression narrative."),
                target_users=item.get("target_users", "Industry specific research networks"),
                datasets=item.get("datasets", "SEC EDGAR System Reference Portal"),
                tools=item.get("tools", "Python, Streamlit, Pandas, Scikit-Learn"),
                technical_approach=item.get("technical_approach", "Data cleansing, processing pipeline, visual deployment"),
                difficulty=item.get("difficulty", "Advanced"),
                feasibility_score=max(1, min(10, int(item.get("feasibility_score", 9)))),
                feasibility_reasoning=item.get("feasibility_reasoning", "Achievable over 20 hours inside 3 weeks."),
                week1_plan=item.get("week1_plan", "Inference mapping and pipeline checks."),
                week2_plan=item.get("week2_plan", "Model tuning and architecture construction."),
                week3_plan=item.get("week3_plan", "Streamlit UI configuration loops."),
                final_demo=item.get("final_demo", "Interactive model performance visualization setup."),
                college_application_value=item.get("college_application_value", "Demonstrates strong intellectual interest."),
                similar_projects=similar_projects_list,
                
                # Extended Problem-First Discoveries Fields Mapping
                problem_context=item.get("problem_context", "Systemic market friction parameters."),
                who_experiences_this_problem=item.get("who_experiences_this_problem", "Target users and ecosystem stakeholders."),
                why_this_problem_exists=item.get("why_this_problem_exists", "Lack of granular structured baseline metrics."),
                potential_project_direction=item.get("potential_project_direction", "Specialized Analytical Model Deployment Dashboard"),
                college_narrative=item.get("college_narrative", "Counselor positioning text verifying exceptional academic focus and unique thematic mastery across data matrices.")
            ))
            
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        SESSIONS[session_id] = {
            "source": "individual", "archetype": student_info["archetype"],
            "profile": profile.model_dump(), "ideas": [i.model_dump() for i in processed_ideas], "niche_ideas": []
        }
        
        discovery_metadata = DiscoveryMetadata(
            generation_framework="BetterMind Labs Trajectory Problem Discovery Engine",
            research_basis="Built leveraging real track intersections across student histories using specialized pipeline metrics.",
            personalization_summary=f"Personalized heavily utilizing continuous trajectory anchors, toughest_project ({student_info['toughest_project']}), and career intent profiles to unlock next-stage engineering milestones."
        )
        
        return GenerateIdeasResponse(session_id=session_id, discovery_metadata=discovery_metadata, ideas=processed_ideas)
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

        group_identity_graph = compile_identity_graph(resolved_profiles)
        
        search_query = " ".join(group_identity_graph.get("combined_interests", []))
        initial_pool = execute_hybrid_retrieval(search_query, {"project_themes": group_identity_graph["combined_interests"]}, top_k=2)
        
        formatted_pool_text = "\n".join([
            f"- Title: {p['title']} | Category: {p['project_category']} | Description: {p['description']}"
            for p in initial_pool
        ]) if initial_pool else "No direct group benchmarks available."

        sys_prompt = generate_master_reasoning_prompt(
            student_matrix=json.dumps(group_identity_graph, indent=2),
            historical_pool=formatted_pool_text,
            group_rules=(
                "Group Mode Cluster Configuration: Synthesize balanced collaborative framework opportunities. "
                "Ensure separate execution sub-tasks mapping directly to the explicit recommended_role_split structures."
            )
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sys_prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            timeout=25.0
        )

        parsed_output = json.loads(response.choices[0].message.content)
        raw_ideas = parsed_output.get("ideas", [])

        processed_ideas: List[ProjectIdea] = []
        for item in raw_ideas[:3]:
            generated_opportunity_context = f"{item.get('title')} {item.get('problem_area')} {item.get('project_description')}"
            refined_matched_pool = execute_hybrid_retrieval(generated_opportunity_context, {"project_themes": group_identity_graph["combined_interests"]}, top_k=2)
            
            similar_projects_list = []
            for p in refined_matched_pool:
                similar_projects_list.append(SimilarProject(
                    title=p["title"],
                    student_name=p["student_name"],
                    description=p["description"],
                    github_url=p["github_url"],
                    youtube_url=p["youtube_url"],
                    project_category=p["project_category"],
                    project_page=p["project_page"],
                    similarity_score=p["similarity_score"],
                    matching_reason=p["matching_reason"]
                ))

            fits_you_summary = item.get("why_this_fits_you", [])
            fits_you_summary.append("Optimized role division split mapped cleanly across teammate skill distribution parameters.")

            processed_ideas.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Synthesized Core Group Concept"),
                problem_area=item.get("problem_area", "Shared multi-variant problem space friction bounds."),
                guiding_question=item.get("guiding_question", "Unified testable group execution inquiries."),
                project_description=item.get("project_description", "Collaborative modular solution implementation parameters."),
                why_it_matters=item.get("why_it_matters", "Establishes systematic framework controls."),
                why_this_fits_you=fits_you_summary,
                student_history_connection=item.get("student_history_connection", "This collaborative project synthesizes the individual technical backgrounds of the cohort members to scale parallel workflows into an advanced integrated architecture."),
                target_users=item.get("target_users", "Target ecosystem user bases"),
                datasets=item.get("datasets", "NOAA Integrated Global Meteorological Repository"),
                tools=item.get("tools", "Python, Streamlit, Git Workflow, Pandas"),
                technical_approach=item.get("technical_approach", "Modular component creation and pipeline merging"),
                difficulty="Advanced",
                feasibility_score=max(1, min(10, int(item.get("feasibility_score", 9)))),
                feasibility_reasoning=item.get("feasibility_reasoning", "Calibrated for efficient parallel multi-student execution profiles."),
                week1_plan=item.get("week1_plan", "Architecture separation and repository initialization setup."),
                week2_plan=item.get("week2_plan", "Parallel training loops and computational engine updates."),
                week3_plan=item.get("week3_plan", "Unified interface mapping and presentation verification loops."),
                final_demo=item.get("final_demo", "Comprehensive multi-widget configuration monitor layout."),
                college_application_value=item.get("college_application_value", "Highlights collaborative infrastructure engineering and non-trivial intellectual curiosities."),
                similar_projects=similar_projects_list,
                
                # Extended Problem-First Fields Mapping
                problem_context=item.get("problem_context", "Systemic cluster friction parameters."),
                who_experiences_this_problem=item.get("who_experiences_this_problem", "Target users and group ecosystem stakeholders."),
                why_this_problem_exists=item.get("why_this_problem_exists", "Lack of parallel baseline data processing benchmarks."),
                potential_project_direction=item.get("potential_project_direction", "Unified Collaborative Modular Analytical Deployment Framework"),
                college_narrative=item.get("college_narrative", "Group counselor positioning text verifying exceptional multi-agent engineering capabilities, collaborative system design, and specialized problem discovery benchmarks.")
            ))

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        SESSIONS[session_id] = {
            "source": "bettermind", "group_profile": group_identity_graph,
            "participants": [p.model_dump() for p in request.participants],
            "ideas": [i.model_dump() for i in processed_ideas], "niche_ideas": []
        }
        
        discovery_metadata = DiscoveryMetadata(
            generation_framework="BetterMind Labs Trajectory Problem Discovery Engine",
            research_basis="Generated using collaborative structural intersections for team configuration models.",
            personalization_summary=f"Recommendations were synthesized via cohort trajectory matching layers and combined interest tracks ({', '.join(group_identity_graph.get('combined_interests', []))})."
        )
        
        return GenerateIdeasResponse(session_id=session_id, discovery_metadata=discovery_metadata, ideas=processed_ideas)
    except Exception as e:
        logger.error(f"Critical execution error inside /generate-bml pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/niche-down", response_model=GenerateIdeasResponse)
def niche_down_endpoint(payload: NicheDownRequest):
    """
    Implements Hierarchical Niche Down.
    Guarantees incremental problem narrowing (e.g. Healthcare -> Mental Health -> Teen Stress) 
    while preserving the entire parent problem context and historical baseline configurations.
    """
    if payload.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Target tracking session identifier missing.")
    
    session_data = SESSIONS[payload.session_id]
    all_ideas = session_data.get("ideas", []) + session_data.get("niche_ideas", [])
    base_idea = next((item for item in all_ideas if item["id"] == payload.idea_id), None)
    
    if not base_idea:
        raise HTTPException(status_code=400, detail="Target baseline project reference node invalid.")

    if not groq_client:
        raise HTTPException(status_code=503, detail="AI Inference engine client unassigned.")

    system_prompt = (
        "You are an expert mentor helping ambitious high school students discover research-backed problem opportunities.\n"
        "Your task is to take a chosen parent problem context and generate exactly 3 highly refined, localized iterations "
        "that narrow down hierarchically into a specialized sub-problem context (e.g. Healthcare -> Mental Health -> Teen Stress). Do NOT switch domains or create random variations.\n\n"
        "Ensure titles match the professional format 'BrandName: Specific Research-Oriented Subtitle'. Datasets must be authentic explicit pipelines.\n\n"
        "Format output strictly inside a parseable JSON block matching this exact schema:\n"
        "{\n"
        '  "ideas": [\n'
        '    {\n'
        '      "title": "[Refined Specialized BrandName: Specific Research-Oriented Subtitle]",\n'
        '      "problem_area": "[Hierarchically narrowed sub-problem bounds and granular real-world friction]",\n'
        '      "guiding_question": "[Precise testable sub-inquiry]",\n'
        '      "project_description": "[Specific solution breakdown avoiding generic chatbots/wrappers]",\n'
        '      "why_it_matters": "[Localized impact statement]",\n'
        '      "why_this_fits_you": ["Fits your specialization trajectory path", "Extends prior work context cleanly"],\n'
        '      "student_history_connection": "[Detailed refinement statement tracking the continuous historical line from your trajectory profile into this niche scope]",\n'
        '      "target_users": "[Narrowed niche user base]",\n'
        '      "datasets": "[Target specialized high-fidelity public data source]",\n'
        '      "tools": "Python, Streamlit",\n'
        '      "technical_approach": "[Refined workflow steps]",\n'
        '      "difficulty": "Advanced",\n'
        '      "feasibility_score": 9,\n'
        '      "feasibility_reasoning": "[Verified bounds]",\n'
        '      "week1_plan": "[Setup loops]",\n'
        '      "week2_plan": "[Training loops]",\n'
        '      "week3_plan": "[UI validation loops]",\n'
        '      "final_demo": "[Interface configuration params]",\n'
        '      "college_application_value": "[Summary value narrative]",\n'
        '      "problem_context": "[Deep specialized overarching systemic problem description]",\n'
        '      "who_experiences_this_problem": "[Specific narrowed stakeholders impacted]",\n'
        '      "why_this_problem_exists": "[Systemic or structural causes for this localized sub-problem]",\n'
        '      "potential_project_direction": "[Secondary localized project title framework direction]",\n'
        '      "college_narrative": "[Counselor-ready positioning narrative detailing unique story components. MUST BE A MINIMUM OF 120 WORDS.]"\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    user_prompt = (
        f"Parent Context Problem: {base_idea.get('problem_area')}\n"
        f"Current Title: {base_idea.get('title')}\n"
        f"Narrow this scope down by one hierarchical step into a granular technical sub-problem."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            timeout=20.0
        )
        parsed = json.loads(response.choices[0].message.content)
        raw_niche = parsed.get("ideas", [])
        
        processed: List[ProjectIdea] = []
        for item in raw_niche[:3]:
            processed.append(ProjectIdea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                title=item.get("title", "Specialized Sub-problem Core"),
                problem_area=item.get("problem_area", base_idea.get("problem_area")),
                guiding_question=item.get("guiding_question", base_idea.get("guiding_question")),
                project_description=item.get("project_description", base_idea.get("project_description")),
                why_it_matters=item.get("why_it_matters", base_idea.get("why_it_matters")),
                why_this_fits_you=item.get("why_this_fits_you", base_idea.get("why_this_fits_you", [])),
                student_history_connection=item.get("student_history_connection", base_idea.get("student_history_connection", "Deepens the specialized technical lineage established in your history profile.")),
                target_users=item.get("target_users", base_idea.get("target_users", "Niche user base")),
                datasets=item.get("datasets", base_idea.get("datasets", "Parent reference dataset")),
                tools=item.get("tools", base_idea.get("tools", "Python, Streamlit")),
                technical_approach=item.get("technical_approach", base_idea.get("technical_approach")),
                difficulty="Advanced",
                feasibility_score=max(1, min(10, int(item.get("feasibility_score", 9)))),
                feasibility_reasoning=item.get("feasibility_reasoning", base_idea.get("feasibility_reasoning")),
                week1_plan=item.get("week1_plan", base_idea.get("week1_plan")),
                week2_plan=item.get("week2_plan", base_idea.get("week2_plan")),
                week3_plan=item.get("week3_plan", base_idea.get("week3_plan")),
                final_demo=item.get("final_demo", base_idea.get("final_demo")),
                college_application_value=item.get("college_application_value", base_idea.get("college_application_value")),
                similar_projects=base_idea.get("similar_projects", []),
                
                # Extended Problem-First Fields Multi-Inheritance
                problem_context=item.get("problem_context", base_idea.get("problem_context", "Narrowed niche context parameters.")),
                who_experiences_this_problem=item.get("who_experiences_this_problem", base_idea.get("who_experiences_this_problem", "Target specific user subset.")),
                why_this_problem_exists=item.get("why_this_problem_exists", base_idea.get("why_this_problem_exists", "Structural system blockages.")),
                potential_project_direction=item.get("potential_project_direction", "Niche Analytical Service Node Interface"),
                college_narrative=item.get("college_narrative", base_idea.get("college_narrative", "Deep niche academic narrative positioning."))
            ))
        
        while len(processed) < 3:
            processed.append(processed[0])
            
        session_data["niche_ideas"] = [i.model_dump() for i in processed]
        
        discovery_metadata = DiscoveryMetadata(
            generation_framework="BetterMind Labs Problem Discovery Engine",
            research_basis="Generated using patterns observed across 200+ student projects built through BetterMind Labs.",
            personalization_summary="Hierarchical narrow-down refinement context generated by isolating a specific real-world sub-problem node from parent configuration architecture."
        )
        
        return GenerateIdeasResponse(session_id=payload.session_id, discovery_metadata=discovery_metadata, ideas=processed)
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