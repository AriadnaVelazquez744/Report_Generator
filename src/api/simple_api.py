"""
Simplified FastAPI backend that works with JSON files instead of a database.
It exposes the endpoints expected by the frontend (auth, session, reports).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.data.loader import ArticleLoader
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
from src.recommendation.matcher import NewsMatcher
from src.recommendation.report_generator import ReportGenerator
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.vectorizer import NewsVectorizer, UserProfileVectorizer
from src.summarization.summarizer import PersonalizedSummarizer, TextRankSummarizer

logger = logging.getLogger(__name__)

# Paths and constants
ROOT_DIR = Path(__file__).resolve().parents[2]
USERS_FILE = ROOT_DIR / "Data_users" / "users.json"
SESSION_FILE = ROOT_DIR / "current_session.json"
ARTICLES_DIR = ROOT_DIR / "Data_articles"
PDF_OUTPUT_DIR = ROOT_DIR / "reportes_pdf"

# Locks for thread safety
_users_lock = threading.Lock()
_session_lock = threading.Lock()
_articles_lock = threading.Lock()

# Cached components
_articles_cache: List[Dict[str, Any]] = []
_news_vectorizer: Optional[NewsVectorizer] = None
_profile_vectorizer: Optional[UserProfileVectorizer] = None
_profile_manager: Optional[UserProfileManager] = None
_matcher: Optional[NewsMatcher] = None
_report_generator: Optional[ReportGenerator] = None
_text_preprocessor = TextPreprocessor(use_spacy=False)
_regex_annotator = RegexAnnotator()

try:
    _spacy_nlp = spacy.load("es_core_news_sm")
except Exception:  # pragma: no cover - optional dependency
    _spacy_nlp = None
    logger.warning("spaCy spanish model not found; entities will be empty.")


def _ensure_base_files() -> None:
    """Create users/session files if they do not exist."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": []}, indent=2), encoding="utf-8")

    if not SESSION_FILE.exists():
        SESSION_FILE.write_text(
            json.dumps(
                {
                    "user_id": "",
                    "messages": [],
                    "created_at": "",
                    "updated_at": "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_users() -> List[Dict[str, Any]]:
    _ensure_base_files()
    data = _read_json(USERS_FILE, {"users": []})
    users = data.get("users", [])
    return users if isinstance(users, list) else []


def _save_users(users: List[Dict[str, Any]]) -> None:
    with _users_lock:
        _write_json(USERS_FILE, {"users": users})


def _sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """Return user data without the password hash."""
    safe = dict(user)
    safe.pop("password", None)
    return safe


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def _create_empty_session(user_id: str) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    return {"user_id": user_id, "messages": [], "created_at": now, "updated_at": now}


def _load_session() -> Dict[str, Any]:
    _ensure_base_files()
    data = _read_json(SESSION_FILE, _create_empty_session(""))
    if not isinstance(data, dict):
        return _create_empty_session("")
    return data


def _save_session(session: Dict[str, Any]) -> None:
    with _session_lock:
        _write_json(SESSION_FILE, session)


def _ensure_profile_components() -> None:
    """
    Initialize only the components needed for user profile creation.
    Does NOT load articles - only creates vectorizers for profile processing.
    Uses generic Spanish text to initialize the vectorizer vocabulary.
    """
    global _news_vectorizer, _profile_vectorizer, _profile_manager

    if _profile_manager and _profile_vectorizer and _news_vectorizer:
        return

    with _articles_lock:
        if _profile_manager and _profile_vectorizer and _news_vectorizer:
            return

        # Create a basic vectorizer with relaxed parameters for profile-only use
        # Use min_df=1 instead of 2 since we have few generic texts
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        basic_vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            min_df=1,  # Allow terms that appear in just 1 document (needed for small vocab)
            max_df=0.95,
            sublinear_tf=True,
            norm='l2',
        )
        
        # Fit with expanded generic Spanish text to initialize vocabulary
        # Repeat terms to ensure they appear multiple times
        generic_texts = [
            "política economía internacional derechos humanos tecnología deportes cultura salud medio ambiente educación ciencia sociedad",
            "política economía internacional derechos humanos tecnología deportes cultura salud medio ambiente educación ciencia sociedad noticias",
            "noticias información actualidad periodismo reportaje análisis opinión política economía",
            "gobierno estado país ciudad región mundo global internacional política",
            "tecnología ciencia educación salud medio ambiente cultura deportes sociedad",
            "derechos humanos gobierno estado país ciudad región mundo global",
        ]
        basic_vectorizer.fit(generic_texts)
        
        # Wrap in NewsVectorizer-compatible object
        vectorizer = NewsVectorizer(max_features=4000, ngram_range=(1, 2))
        vectorizer.vectorizer = basic_vectorizer
        vectorizer.fitted = True

        profile_vectorizer = UserProfileVectorizer(vectorizer)
        profile_manager = UserProfileManager(profile_vectorizer)

        _news_vectorizer = vectorizer
        _profile_vectorizer = profile_vectorizer
        _profile_manager = profile_manager

        logger.info("Profile components initialized (no articles loaded).")


def _ensure_recommendation_stack() -> None:
    """
    Lazily load and cache articles, vectorizers, and generators.
    Runs once and is shared across requests.
    """
    global _articles_cache, _news_vectorizer, _profile_vectorizer
    global _profile_manager, _matcher, _report_generator

    if _articles_cache and _news_vectorizer and _profile_manager and _matcher:
        return

    with _articles_lock:
        if _articles_cache and _news_vectorizer and _profile_manager and _matcher:
            return

        if not ARTICLES_DIR.exists():
            raise HTTPException(status_code=500, detail="Articles directory not found.")

        loader = ArticleLoader(str(ARTICLES_DIR))
        raw_articles = loader.load_all_articles(recursive=True)
        if not raw_articles:
            raise HTTPException(status_code=500, detail="No articles found to process.")

        processed_articles: List[Dict[str, Any]] = []
        clean_texts: List[str] = []

        for idx, article in enumerate(raw_articles):
            text = article.get("text", "") or ""
            clean_tokens = _text_preprocessor.preprocess(text, return_tokens=True)
            clean_text = " ".join(clean_tokens)
            annotations = _regex_annotator.annotate(text)

            processed = {
                **article,
                "id": idx,
                "clean_text": clean_text,
                "categories": annotations.get("categories", []),
                "entidades": annotations.get("entities", []),
            }
            processed_articles.append(processed)
            clean_texts.append(clean_text)

        vectorizer = NewsVectorizer(max_features=4000, ngram_range=(1, 2))
        article_matrix = vectorizer.fit_transform0(clean_texts)
        for i, article in enumerate(processed_articles):
            article["vector"] = article_matrix[i].tolist()

        profile_vectorizer = UserProfileVectorizer(vectorizer)
        profile_manager = UserProfileManager(profile_vectorizer)
        matcher = NewsMatcher.from_articles(processed_articles, vectorizer=vectorizer)

        base_summarizer = TextRankSummarizer(language="spanish")
        personalized_summarizer = PersonalizedSummarizer(base_summarizer)
        report_generator = ReportGenerator(personalized_summarizer)

        _articles_cache = processed_articles
        _news_vectorizer = vectorizer
        _profile_vectorizer = profile_vectorizer
        _profile_manager = profile_manager
        _matcher = matcher
        _report_generator = report_generator

        logger.info("Articles and recommendation stack initialized (%s articles).", len(_articles_cache))


# Request models
class LoginRequest(BaseModel):
    name: str
    password: str


class RegisterRequest(BaseModel):
    password: str
    name: str
    selected_categories: List[str] = Field(default_factory=list)
    additional_interests: str = ""


class SessionMessageRequest(BaseModel):
    user_id: str
    type: Literal["request", "report"]
    content: str
    report_data: Optional[Dict[str, Any]] = None


class ClearSessionRequest(BaseModel):
    user_id: str


class RecommendationRequest(BaseModel):
    profile_text: str
    max_articles: int = 10


class PDFRequest(BaseModel):
    report: Dict[str, Any]
    user_name: Optional[str] = None


app = FastAPI(title="Report Generator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    _ensure_base_files()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# User management
@app.get("/api/users")
def list_users() -> Dict[str, Any]:
    users = [_sanitize_user(u) for u in _load_users()]
    return {"users": users}


@app.post("/api/users/login")
def login(payload: LoginRequest) -> Dict[str, Any]:
    users = _load_users()
    for user in users:
        if user.get("name") == payload.name and user.get("password") == payload.password:
            return {"user": user}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/users/register")
def register(payload: RegisterRequest) -> Dict[str, Any]:
    # Only initialize profile components, not the full recommendation stack
    _ensure_profile_components()
    users = _load_users()

    # Check if user with same name already exists
    if any(u.get("name") == payload.name for u in users):
        raise HTTPException(status_code=400, detail="User with this name already exists")

    profile_text_parts = []
    if payload.selected_categories:
        profile_text_parts.append("Intereses seleccionados: " + ", ".join(payload.selected_categories))
    additional_interests_clean = (payload.additional_interests or "").strip()
    if additional_interests_clean:
        profile_text_parts.append("Detalle adicional: " + additional_interests_clean)
    profile_text = ". ".join(profile_text_parts) if profile_text_parts else "Usuario sin detalles"

    def _dummy_interest_pipeline(text: str) -> Dict[str, Any]:
        # Placeholder: replace with real processing later
        text_clean = (text or "").strip()
        if not text_clean:
            return {"processed_interests": "", "keywords": []}
        return {"processed_interests": text_clean, "keywords": text_clean.split()[:5]}

    processed_interests = _dummy_interest_pipeline(payload.additional_interests or "")

    # Initialize profile components (doesn't load articles)
    _ensure_profile_components()
    
    profile_data = _profile_manager.create_profile(profile_text, nlp=_spacy_nlp)
    merged_categories = sorted(set(payload.selected_categories + profile_data.get("categories", [])))
    profile_vector = _profile_vectorizer.vectorize_profile(profile_text, merged_categories).tolist()

    now = datetime.utcnow().isoformat()
    # Generate internal user number (not shown to user)
    number = f"user_{int(time.time()*1000)}"

    new_user = {
        "number": number,
        "password": payload.password,
        "name": payload.name,
        "profile_text": profile_text,
        "categories": merged_categories,
        "entities": profile_data.get("entities", []),
        "vector": profile_vector,
        "created_at": now,
        "updated_at": now,
        "additional_interests": processed_interests,
    }

    users.append(new_user)
    _save_users(users)
    return {"user": new_user}


# Session management
@app.get("/api/session")
def get_session(user_id: str) -> Dict[str, Any]:
    session = _load_session()
    if session.get("user_id") != user_id:
        session = _create_empty_session(user_id)
        _save_session(session)
    return session


@app.post("/api/session/message")
def add_session_message(payload: SessionMessageRequest) -> Dict[str, Any]:
    session = _load_session()
    if session.get("user_id") != payload.user_id:
        session = _create_empty_session(payload.user_id)

    message = {
        "type": payload.type,
        "content": payload.content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if payload.report_data is not None:
        message["report_data"] = payload.report_data

    session["messages"].append(message)
    session["updated_at"] = datetime.utcnow().isoformat()
    _save_session(session)
    return {"ok": True, "session": session}


@app.post("/api/session/clear")
def clear_session(payload: ClearSessionRequest) -> Dict[str, Any]:
    session = _create_empty_session(payload.user_id)
    _save_session(session)
    return session


# Recommendation and reports
@app.post("/recommendations/generate")
def generate_recommendations(payload: RecommendationRequest) -> Dict[str, Any]:
    _ensure_recommendation_stack()

    profile_data = _profile_manager.create_profile(payload.profile_text, nlp=_spacy_nlp)
    matches = _matcher.match_articles(
        profile_data,
        _articles_cache,
        top_k=payload.max_articles,
    )
    report = _report_generator.generate_report(matches, profile_data, max_articles=payload.max_articles)
    return {"report": report}


@app.post("/reports/generate-pdf")
def generate_pdf(payload: PDFRequest):
    _ensure_recommendation_stack()
    if not payload.report:
        raise HTTPException(status_code=400, detail="Report payload is required.")

    user_slug = _slugify(payload.user_name or "usuario")
    filename = PDF_OUTPUT_DIR / f"reporte_{user_slug}_{int(time.time())}.pdf"
    ok = _report_generator.generate_pdf(payload.report, str(filename), payload.user_name)

    if not ok or not filename.exists():
        raise HTTPException(status_code=500, detail="Could not generate PDF.")

    return FileResponse(
        path=filename,
        media_type="application/pdf",
        filename=filename.name,
    )


