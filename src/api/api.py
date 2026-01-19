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
from typing import Any, Dict, List, Literal, Optional, Tuple

import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import os

from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
from src.process.profile_process import create_complete_user_profile
from src.process.report_formatter import generate_report_from_user_query
from src.recommendation.matcher import NewsMatcher
from src.recommendation.report_generator import ReportGenerator
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.vectorizer import NewsVectorizer, UserProfileVectorizer
from src.summarization.summarizer import PersonalizedSummarizer, TextRankSummarizer, ModelSummarizer
from src.summarization.model.config import Config
from src.summarization.model.vocabulary import Vocabulary
from src.summarization.model.constant import MAX_LEN_SRC,\
                                            MAX_LEN_TGT,MAX_VOCAB_SIZE,HIDDEN_SIZE,EMBEDDING_SIZE,\
                                            NUM_DEC_LAYERS,NUM_ENC_LAYERS, USE_GPU,UNK_TOKEN,BEAM_SIZE,\
                                            IS_COVERAGE,IS_PGEN,DROPOUT_RATIO,BIDIRECTIONAL,DEVICE,DECODING_STRATEGY,GPU_ID,\
                                            PAD_TOKEN,START_DECODING,END_DECODING,CHECKPOINT_VOCABULARY_DIR,DATA_DIR,VOCAB_NAME

logger = logging.getLogger(__name__)

# Paths and constants
ROOT_DIR = Path(__file__).resolve().parents[2]
USERS_FILE = ROOT_DIR / "Data" / "Data_users" / "users.json"
SESSION_FILE = ROOT_DIR / "Data" / "current_session.json"
ARTICLES_DIR = ROOT_DIR / "Data" / "Data_articles"
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
_text_preprocessor: Optional[TextPreprocessor] = None
_regex_annotator: Optional[RegexAnnotator] = None
_summarizer: Optional[ModelSummarizer] = None

_spacy_nlp: Optional[Any] = None

def _load_spacy_model() -> Optional[Any]:
    """Carga el modelo de spaCy de forma lazy."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        _spacy_nlp = spacy.load("es_core_news_sm")
    except Exception:
        logger.warning("spaCy spanish model not found; entities will be empty.")
    return _spacy_nlp


def _try_auto_update_news() -> None:
    """
    Intenta ejecutar la actualización automática de noticias al iniciar la API.
    Se ejecuta en background sin bloquear el inicio del servidor.
    """
    import threading
    import importlib.util
    from pathlib import Path
    
    def _run_update():
        try:
            update_script = ROOT_DIR / "src" / "process" / "news_update_&_process.py"
            if not update_script.exists():
                return
            
            spec = importlib.util.spec_from_file_location("news_update_process", update_script)
            if not spec or not spec.loader:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "run_news_update"):
                module.run_news_update()
        except Exception as e:
            logger.debug(f"No se pudo ejecutar actualización automática: {e}")
    
    # Ejecutar en thread separado para no bloquear el startup
    thread = threading.Thread(target=_run_update, daemon=True)
    thread.start()


def _ensure_summarizer() -> Optional[ModelSummarizer]:
    global _summarizer
    if _summarizer:
        return _summarizer
    print("Cargando vocabulario...")
    vocab = Vocabulary(
        CREATE_VOCABULARY=False,  
        PAD_TOKEN=PAD_TOKEN,
        UNK_TOKEN=UNK_TOKEN,
        START_DECODING=START_DECODING,
        END_DECODING=END_DECODING,
        MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,
        CHECKPOINT_VOCABULARY_DIR=CHECKPOINT_VOCABULARY_DIR,
        DATA_DIR=DATA_DIR,
        VOCAB_NAME=VOCAB_NAME
    )
    vocab.build_vocabulary()
    print(f"Vocabulario cargado: {vocab.total_size()} palabras")
    
    config = Config(
        max_vocab_size=vocab.total_size(),
        src_len=MAX_LEN_SRC,
        tgt_len=MAX_LEN_TGT,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_enc_layers=NUM_ENC_LAYERS,
        num_dec_layers=NUM_DEC_LAYERS,
        use_gpu=USE_GPU,
        is_pgen=IS_PGEN,
        is_coverage=IS_COVERAGE,
        dropout_ratio=DROPOUT_RATIO,
        bidirectional=BIDIRECTIONAL,
        device=DEVICE,
        decoding_strategy=DECODING_STRATEGY,
        beam_size=BEAM_SIZE,
        gpu_id=GPU_ID
    )
    
    model_path = ROOT_DIR / "src" / "summarization" / "finetune_last-v2.pt" # solo cambiar el nombre
    if model_path.exists():
        try:
            _summarizer = ModelSummarizer(str(model_path),vocab=vocab,config=config)
            logger.info("ModelSummarizer initialized.")
        except Exception as e:
            logger.error(f"Error initializing ModelSummarizer: {e}")
    else:
        logger.warning(f"Model file not found at {model_path}")
    
    return _summarizer


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
    # else:
    #     try:
    #         with open(SESSION_FILE, 'r', encoding='utf-8') as f:
    #             session = json.load(f)
    #             user_id = session.get("user_id")
    #             clean_session = _create_empty_session(user_id)
    #             _save_session(clean_session)
    #     except json.JSONDecodeError as e:
    #         logger.error(f"Error decodificando JSON en {file_path}: {e}")

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
    # Hora local para que el frontend muestre hora correcta del sistema
    now = datetime.now().isoformat()
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


def _ensure_profile_components() -> tuple[UserProfileVectorizer, UserProfileManager]:
    """
    Inicializa solo los componentes necesarios para crear perfiles de usuario.
    No carga artículos, solo crea vectorizadores para procesamiento de perfiles.
    Retorna los componentes listos para usar.
    """
    global _news_vectorizer, _profile_vectorizer, _profile_manager

    if _profile_manager and _profile_vectorizer and _news_vectorizer:
        return _profile_vectorizer, _profile_manager

    with _articles_lock:
        if _profile_manager and _profile_vectorizer and _news_vectorizer:
            return _profile_vectorizer, _profile_manager

        from sklearn.feature_extraction.text import TfidfVectorizer
        
        basic_vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2',
        )
        
        generic_texts = [
            "política economía internacional derechos humanos tecnología deportes cultura salud medio ambiente educación ciencia sociedad",
            "política economía internacional derechos humanos tecnología deportes cultura salud medio ambiente educación ciencia sociedad noticias",
            "noticias información actualidad periodismo reportaje análisis opinión política economía",
            "gobierno estado país ciudad región mundo global internacional política",
            "tecnología ciencia educación salud medio ambiente cultura deportes sociedad",
            "derechos humanos gobierno estado país ciudad región mundo global",
        ]
        basic_vectorizer.fit(generic_texts)
        
        vectorizer = NewsVectorizer(max_features=4000, ngram_range=(1, 2))
        vectorizer.vectorizer = basic_vectorizer
        vectorizer.fitted = True

        profile_vectorizer = UserProfileVectorizer(vectorizer)
        profile_manager = UserProfileManager(profile_vectorizer)

        _news_vectorizer = vectorizer
        _profile_vectorizer = profile_vectorizer
        _profile_manager = profile_manager

        logger.info("Profile components initialized.")
        return profile_vectorizer, profile_manager


def _ensure_matching_components() -> tuple[UserProfileVectorizer, NewsMatcher]:
    """
    Inicializa profile_vectorizer y matcher necesarios para generate_report_from_user_query.
    Carga artículos con preprocesamiento completo para el matcheo.
    Se ejecuta una sola vez y se comparte entre requests.
    
    OPTIMIZACIÓN: Usa cache binario de artículos procesados para evitar leer ~48k JSONs.
    Retorna: (profile_vectorizer, matcher)
    """
    global _profile_vectorizer, _matcher, _news_vectorizer
    import pickle

    print("Inicializando componentes de matching...")

    if _profile_vectorizer and _matcher:
        print("✅ Componentes ya cargados en memoria")
        return _profile_vectorizer, _matcher

    with _articles_lock:
        if _profile_vectorizer and _matcher:
            return _profile_vectorizer, _matcher

        if not ARTICLES_DIR.exists():
            raise HTTPException(status_code=500, detail="Articles directory not found.")
            
        vectorizer_path = ROOT_DIR / "Data" / "vectorizer.pkl"
        articles_cache_path = ROOT_DIR / "Data" / "articles_cache.pkl"
        state_path = ROOT_DIR / "Data" / "articles_state.json"
        current_version = "1.1"
        vectorizer = None
        should_refit = True
        
        # 1. Intentar cargar vectorizer existente
        if vectorizer_path.exists():
            try:
                print(f"Cargando vectorizador desde {vectorizer_path}...")
                loaded_vec = NewsVectorizer.load(str(vectorizer_path))
                loaded_version = getattr(loaded_vec, 'version', '1.0')
                if loaded_version == current_version:
                    vectorizer = loaded_vec
                    should_refit = False
                    print(f"✅ Vectorizador cargado (v{loaded_version})")
                else:
                    print(f"⚠️ Versión mismatch ({loaded_version} vs {current_version}). Re-entrenando...")
            except Exception as e:
                print(f"⚠️ Error cargando vectorizador: {e}. Se creará uno nuevo.")

        #  Intentar usar cache de artículos procesados
        if not should_refit and articles_cache_path.exists() and state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state_data = json.load(f)
                    cached_count = state_data.get("count", -1)
                    cached_version = state_data.get("version", "")
                
                # Validación rápida: contar archivos JSON en vez de cargarlos todos
                from src.data.loader import ArticleLoader
                loader = ArticleLoader(str(ARTICLES_DIR))
                current_count = loader.count_articles(recursive=True)
                
                if cached_count == current_count and cached_version == current_version:
                    print(f"✅ Cache válido ({current_count} artículos). Cargando desde cache...")
                    with open(articles_cache_path, 'rb') as f:
                        processed_articles = pickle.load(f)
                    print(f"✅ {len(processed_articles)} artículos cargados desde cache")
                    
                    # Crear componentes y retornar inmediatamente
                    profile_vectorizer = UserProfileVectorizer(vectorizer)
                    matcher = NewsMatcher.from_articles(processed_articles, vectorizer=vectorizer)
                    _profile_vectorizer = profile_vectorizer
                    _matcher = matcher
                    _news_vectorizer = vectorizer
                    logger.info("Matching components initialized from cache (%s articles).", len(processed_articles))
                    return _profile_vectorizer, _matcher
                else:
                    print(f"⚠️ Cache inválido (disco: {current_count} vs cache: {cached_count}). Recargando...")
            except Exception as e:
                print(f"⚠️ Error leyendo cache: {e}. Recargando artículos...")

        # 3. Cargar artículos desde JSONs (solo si es necesario)
        from src.data.loader import ArticleLoader
        from src.recommendation.vectorize_articles import vectorize_articles_directory
        
        print("Cargando artículos desde disco...")
        loader = ArticleLoader(str(ARTICLES_DIR))
        raw_articles = loader.load_all_articles(recursive=True)
        
        if not raw_articles:
            vectorizer = NewsVectorizer(max_features=4000, ngram_range=(1, 2), use_lemmatization=True)
            vectorizer.version = current_version 
            vectorizer.fit0(["dummy text"])
            _news_vectorizer = vectorizer
            _profile_vectorizer = UserProfileVectorizer(vectorizer)
            _matcher = NewsMatcher(vectorizer=vectorizer)
            return _profile_vectorizer, _matcher

        # 4. Re-entrenar vectorizador si es necesario
        if should_refit:
            print("Entrenando nuevo vectorizador...")
            processed_articles: List[Dict[str, Any]] = []
            clean_texts: List[str] = []

            for idx, article in enumerate(raw_articles):
                clean_text = article.get('preprocessing', {}).get('cleaned', "")
                annotations = article.get('regex_annotations', {})
                processed = {
                    **article,
                    "id": idx,
                    "clean_text": clean_text,
                    "categories": annotations.get("categories", []),
                    "entidades": annotations.get("entities", []),
                    "date": article.get("source_metadata", {}).get("date")
                }
                processed_articles.append(processed)
                clean_texts.append(clean_text)

            vectorizer = NewsVectorizer(max_features=4000, ngram_range=(1, 2), use_lemmatization=True)
            article_matrix = vectorizer.fit_transform0(clean_texts)
            vectorizer.version = current_version
            vectorizer.save(str(vectorizer_path))
            print(f"✅ Vectorizador guardado en {vectorizer_path}")
            
            print("💾 Actualizando vectores en JSONs...")
            article_dirs = [d for d in ARTICLES_DIR.iterdir() if d.is_dir() and d.name.startswith("Data_articles")]
            total_updated = 0
            for adir in article_dirs:
                count = vectorize_articles_directory(str(adir), vectorizer)
                total_updated += count
            print(f"✅ {total_updated} artículos actualizados")
            
            for i, article in enumerate(processed_articles):
                article["vector"] = article_matrix[i].tolist()
        else:
            # Procesar artículos usando vectores existentes
            print("📊 Procesando artículos con vectores existentes...")
            processed_articles = []
            for idx, article in enumerate(raw_articles):
                clean_text = article.get('preprocessing', {}).get('cleaned', "")
                annotations = article.get('regex_annotations', {})
                processed = {
                    **article,
                    "id": idx,
                    "clean_text": clean_text,
                    "categories": annotations.get("categories", []),
                    "entidades": annotations.get("entities", []),
                    "date": article.get("source_metadata", {}).get("date"),
                    "vector": article.get('vector', [])
                }
                processed_articles.append(processed)

        # Guardar cache de artículos procesados
        try:
            print("💾 Guardando cache de artículos...")
            with open(articles_cache_path, 'wb') as f:
                pickle.dump(processed_articles, f)
            with open(state_path, "w") as f:
                json.dump({"count": len(processed_articles), "version": current_version}, f)
            print(f"✅ Cache guardado ({len(processed_articles)} artículos)")
        except Exception as e:
            logger.warning(f"No se pudo guardar cache: {e}")

        print("✅ Vectorizador listo")

        profile_vectorizer = UserProfileVectorizer(vectorizer)
        matcher = NewsMatcher.from_articles(processed_articles, vectorizer=vectorizer)

        _profile_vectorizer = profile_vectorizer
        _matcher = matcher
        _news_vectorizer = vectorizer

        logger.info("Matching components initialized (%s articles loaded).", len(processed_articles))


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


class UserInputRecommendationRequest(BaseModel):
    """Request para generar recomendaciones desde input del usuario"""
    user_id: str
    user_input: str
    max_articles: int = 10
    input_weight: float = 0.7
    prioritize_recent: bool = True


class PDFRequest(BaseModel):
    report: Dict[str, Any]
    user_name: Optional[str] = None
    user_query: Optional[str] = None
    custom_path: Optional[str] = None
    browser_mode: bool = False


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
    # Ejecutar actualización automática de noticias en background (best-effort)
    _try_auto_update_news()
    _ensure_matching_components()
    _ensure_summarizer()


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
            # session = _create_empty_session(user.get("number"))
            # _save_session(session)
            return {"user": user}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/users/register")
def register(payload: RegisterRequest) -> Dict[str, Any]:
    profile_vectorizer, profile_manager = _ensure_profile_components()
    users = _load_users()

    # Check if user with same name already exists
    if any(u.get("name") == payload.name for u in users):
        raise HTTPException(status_code=400, detail="User with this name already exists")

    nlp = _load_spacy_model()
    profile_data = create_complete_user_profile(
        selected_categories=payload.selected_categories,
        additional_interests=payload.additional_interests or "",
        profile_manager=profile_manager,
        profile_vectorizer=profile_vectorizer,
        nlp=nlp
    )

    now = datetime.now().isoformat()
    # Generate internal user number (not shown to user)
    number = f"user_{int(time.time()*1000)}"

    new_user = {
        "number": number,
        "password": payload.password,
        "name": payload.name,
        "profile_text": profile_data["profile_text"],
        "categories": profile_data["categories"],
        "entities": profile_data["entities"],
        "vector": profile_data["vector"],
        "created_at": now,
        "updated_at": now,
        "additional_interests": profile_data["additional_interests"],
    }

    users.append(new_user)
    _save_users(users)
    # session = _create_empty_session(number)
    # _save_session(session)
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
        "timestamp": datetime.now().isoformat(),  # hora local
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
@app.post("/recommendations/generate-text-report")
def generate_text_report(payload: UserInputRecommendationRequest) -> Dict[str, Any]:
    """
    Genera un reporte en texto plano usando generate_report_from_user_query().
    Devuelve texto plano para mostrar en el scroller del frontend.
    """
    print("se hizo la llamada")
    t_start = time.monotonic()
    logger.info("=" * 80)
    logger.info("generate-text-report | REQUEST RECEIVED")
    logger.info("generate-text-report | user_id=%s", payload.user_id)
    logger.info("generate-text-report | user_input length=%d", len(payload.user_input))
    logger.info("generate-text-report | max_articles=%d", payload.max_articles)

    print("loggers information trows")

    nlp = _load_spacy_model()

    print("inicializado nlp")

    users_file_path = str(USERS_FILE)
    print(payload.json())
    print("establecida dirección del archivo de usuario")
    
    try:
        # Generar reporte en texto plano usando la función de report_formatter
        structured_report, text_report = generate_report_from_user_query(
            user_id=payload.user_id,
            user_query=payload.user_input,
            profile_vectorizer=_profile_vectorizer,
            matcher=_matcher,
            nlp=nlp,
            max_articles=payload.max_articles,
            users_file_path=users_file_path,
            summarizer=_summarizer
        )

        print("se generó el reporte a partir de la query del usuario")
        
        logger.info("generate-text-report | completed in %.2fs total", time.monotonic() - t_start)
        
        return {
            "structured_report": structured_report,
            "text_report": text_report,
            "status": "success"
        }
        
    except Exception as e:
        logger.error("generate-text-report | error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error generating text report: {str(e)}")


@app.get("/reports/download-pdf")
def download_pdf(filename: str):
    """Endpoint para descargar un PDF generado previamente"""
    file_path = PDF_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.post("/reports/generate-pdf")
def generate_pdf(payload: PDFRequest):
    # Inicializar report generator si no existe
    global _report_generator
    if not _report_generator:
        _report_generator = ReportGenerator()
        logger.info("Report generator initialized.")
    
    report_generator = _report_generator
    if not payload.report:
        raise HTTPException(status_code=400, detail="Report payload is required.")

    # Determinar ruta de salida
    if payload.custom_path:
        filename = Path(payload.custom_path)
        # Asegurar que tenga extensión .pdf
        if not filename.suffix.lower() == '.pdf':
            filename = filename.with_suffix('.pdf')
    else:
        user_slug = _slugify(payload.user_name or "usuario")
        filename = PDF_OUTPUT_DIR / f"reporte_{user_slug}_{int(time.time())}.pdf"
    
    # Generar PDF con nueva estructura
    ok = report_generator.generate_pdf(
        report=payload.report,
        output_path=str(filename),
        user_name=payload.user_name,
        user_query=payload.user_query
    )

    if not ok or not filename.exists():
        raise HTTPException(status_code=500, detail="Could not generate PDF.")

    if payload.browser_mode:
        # Devolver PDF para visualización en navegador
        return FileResponse(
            path=filename,
            media_type="application/pdf",
            filename=filename.name,
        )
    else:
        # Devolver información del archivo generado
        return {
            "success": True,
            "filename": filename.name,
            "path": str(filename),
            "size": filename.stat().st_size
        }


