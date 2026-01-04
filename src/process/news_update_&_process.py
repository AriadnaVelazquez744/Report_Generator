"""
Pipeline para actualizar automáticamente la base de noticias.

Pasos:
1) Verifica la fecha de la última actualización; si han pasado ≥3 días continúa.
2) Comprueba conexión a internet.
3) Descarga mensajes nuevos de Telegram desde la última fecha.
4) Procesa URLs de teleSUR y descarga artículos.
5) Ejecuta el flujo NLP sobre los artículos recién descargados.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Asegurar rutas para imports locales
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scraper.scraper import Scraper  # type: ignore
from telegram.extract_data_tg import ScraperT  # type: ignore

from src.nlp.preprocessing import TextPreprocessor, detect_language
from src.nlp.pos_analyzer import POSAnalyzer
from src.nlp.grammar_analyzer import GrammarAnalyzer
from src.nlp.regex_annotator import RegexAnnotator
from src.nlp.relation_extractor import RelationExtractor
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.recommendation.vectorize_articles import vectorize_articles_directory
from src.recommendation.vectorizer import NewsVectorizer

load_dotenv()


def _get_env(key: str, default: str | None = None) -> Optional[str]:
    """Lee variables de entorno en minúsculas o mayúsculas."""
    return os.environ.get(key) or os.environ.get(key.upper()) or default


# Configuración y rutas
TELEGRAM_GROUP = _get_env("tg_group_username", "teleSURtv")
TG_API_ID = _get_env("api_id")
TG_API_HASH = _get_env("api_hash")

DATA_DIR = PROJECT_ROOT / "Data"
TELEGRAM_DATA_DIR = DATA_DIR / TELEGRAM_GROUP
DATA_ARTICLES_ROOT = PROJECT_ROOT / "Data" / "Data_articles"
STATE_FILE = DATA_ARTICLES_ROOT / "last_update.json"


# Utilidades básicas
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_last_update() -> Optional[datetime]:
    """Lee la última fecha de actualización desde disco."""
    if not STATE_FILE.exists():
        return None
    try:
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "last_update" in data:
            return datetime.fromisoformat(data["last_update"])
    except Exception as exc:  # pragma: no cover - ruta defensiva
        print(f"No se pudo leer {STATE_FILE}: {exc}")
    return None


def save_last_update(ts: datetime) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as fh:
        json.dump({"last_update": ts.isoformat()}, fh, ensure_ascii=False, indent=2)


def needs_update(last_update: Optional[datetime], threshold_days: int = 3) -> bool:
    if last_update is None:
        return True
    return (utcnow() - last_update) >= timedelta(days=threshold_days)


def has_internet_connection(test_url: str = "https://www.google.com", timeout: int = 5) -> bool:
    try:
        requests.get(test_url, timeout=timeout)
        return True
    except Exception:
        return False


def next_articles_dir(root: Path = DATA_ARTICLES_ROOT) -> Path:
    """Crea y devuelve la próxima carpeta Data_articlesN."""
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("Data_articles"):
            suffix = path.name.replace("Data_articles", "")
            try:
                num = int(suffix) if suffix else 1
            except ValueError:
                continue
            existing.append(num)
    next_num = (max(existing) + 1) if existing else 1
    new_dir = root / f"Data_articles{next_num}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


# Descarga de mensajes y artículos
def fetch_telegram_messages(days_to_fetch: int) -> None:
    """Ejecuta la descarga de mensajes del grupo en un rango de días."""
    if not TG_API_ID or not TG_API_HASH:
        print("⚠️ No se configuraron api_id/api_hash en .env; se omite la descarga de Telegram.")
        return

    scraper_t = ScraperT(
        group_username=TELEGRAM_GROUP,
        api_id=TG_API_ID,
        api_hash=TG_API_HASH,
        max_workers=5,
    )
    print(f"⏬ Descargando mensajes de los últimos {days_to_fetch} días...")
    asyncio.run(
        scraper_t.extract_group_sms(
            limit=None,
            extract_all=False,
            n=days_to_fetch,
            batch_size=150,
        )
    )


def scrape_telesur_articles(output_dir: Path) -> List[Dict]:
    """Extrae URLs de teleSUR desde los JSON de Telegram y descarga artículos."""
    scraper = Scraper(max_workers=5)
    print(f"📰 Procesando URLs de teleSUR desde {TELEGRAM_DATA_DIR}")
    scraped = scraper.scrape_urls_from_data(
        json_file_path=str(TELEGRAM_DATA_DIR),
        output_dir=str(output_dir),
    )
    return scraped


# Procesamiento NLP
def load_article(path: Path) -> Dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def persist_article(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def update_article(path: Path, updates: Dict) -> Dict:
    data = load_article(path)
    data.update(updates)
    persist_article(path, data)
    return data


def process_nlp_for_dir(articles_dir: Path) -> None:
    """Replica el flujo de test_nlp.ipynb sobre una carpeta de artículos."""
    article_paths = sorted(articles_dir.glob("*.json"))
    skip_files = {"all_articles_with_metadata.json", "articles_index.json"}
    article_paths = [p for p in article_paths if p.name not in skip_files]

    if not article_paths:
        print(f"No se encontraron artículos en {articles_dir}")
        return

    preprocessor = TextPreprocessor(use_spacy=False, remove_stopwords=True)
    pos_analyzer = POSAnalyzer()
    grammar_analyzer = GrammarAnalyzer()
    regex_annotator = RegexAnnotator()
    try:
        relation_extractor: RelationExtractor | None = RelationExtractor()
    except ValueError as exc:
        relation_extractor = None
        print(f"⚠️ No se podrán extraer relaciones: {exc}")
    sentiment_analyzer = SentimentAnalyzer()

    for path in article_paths:
        article = load_article(path)
        text = article.get("text", "")
        if not text:
            continue

        lang = detect_language(text)
        preprocessing = preprocessor.preprocess_full(text)

        pos_info = pos_analyzer.analyze(text)
        pos_patterns = pos_analyzer.get_top_patterns(pos_info, n=5)

        grammar_info = grammar_analyzer.analyze(text)
        regex_annotations = regex_annotator.annotate(text)

        if relation_extractor:
            rel_info = relation_extractor.extract(text)
        else:
            rel_info = {"entidades": [], "relaciones": [], "error": "Modelo spaCy no disponible"}

        sentiment = sentiment_analyzer.analyze(text)

        updates = {
            "language": lang,
            "preprocessing": {
                "cleaned": preprocessing["cleaned"],
                "tokens": preprocessing["tokens"],
                "token_count": preprocessing["token_count"],
                "sentence_count": preprocessing["sentence_count"],
            },
            "pos_analysis": {
                "tag_freq": pos_info.get("tag_freq", {}),
                "bigram_freq": pos_info.get("bigram_freq", {}),
                "trigram_freq": pos_info.get("trigram_freq", {}),
                "top_patterns": pos_patterns,
            },
            "grammar_analysis": grammar_info,
            "regex_annotations": regex_annotations,
            "knowledge_graph": rel_info,
            "sentiment": sentiment,
        }

        update_article(path, updates)
        print(f"✓ Procesado NLP: {path.name}")


# Pipeline principal
def run_news_update() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TG_API_ID or not TG_API_HASH:
        print("⚠️ No se configuraron api_id/api_hash; se omite actualización automática.")
        return

    last_update = load_last_update()
    if not needs_update(last_update):
        print("⏸ No es necesario actualizar: la última actualización fue hace menos de 3 días.")
        return

    if not has_internet_connection():
        print("❌ Sin conexión a internet. Reintenta cuando haya red disponible.")
        return

    # Días a consultar en Telegram: al menos 3, o desde la última fecha.
    if last_update:
        days_to_fetch = max(1, (utcnow() - last_update).days + 1)
    else:
        days_to_fetch = 3

    fetch_telegram_messages(days_to_fetch)

    new_articles_dir = next_articles_dir()
    scraped_articles = scrape_telesur_articles(new_articles_dir)

    if not scraped_articles:
        print("⚠️ No se descargaron artículos nuevos; no se ejecutará NLP.")
        return

    process_nlp_for_dir(new_articles_dir)
    
    # Vectorizar artículos recién procesados
    print(f"🔄 Vectorizando nuevos artículos...")
    
    vectorizer_path = PROJECT_ROOT / "Data" / "vectorizer.pkl"
    if vectorizer_path.exists():
        try:
            vectorizer = NewsVectorizer.load(str(vectorizer_path))
            print(f"✅ Vectorizador maestro cargado (v{getattr(vectorizer, 'version', 'unknown')})")
            
            # Usar transform (no fit!) para mantener consistencia
            vectorized_count = vectorize_articles_directory(
                directory_path=str(new_articles_dir),
                news_vectorizer=vectorizer
            )
            print(f"✅ {vectorized_count} nuevos artículos vectorizados")
            
        except Exception as e:
            print(f"❌ Error al cargar/usar vectorizador maestro: {e}")
            print("⚠️ Los nuevos artículos no tendrán vectores compatibles hasta que se reinicie la API.")
    else:
        print("⚠️ No se encontró vectorizador maestro (vectorizer.pkl).")
        print("ℹ️ Ejecute la API primero para inicializar el sistema de recomendación.")
    
    save_last_update(utcnow())
    print(f"✅ Actualización completa. Nuevos artículos en: {new_articles_dir}")


if __name__ == "__main__":
    run_news_update()
