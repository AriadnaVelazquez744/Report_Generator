"""
Pipeline de procesamiento de perfiles de usuario
Procesa texto del perfil, extrae entidades, categorías y vectoriza preferencias
"""
import logging
from typing import Dict, List, Optional, Any
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.vectorizer import UserProfileVectorizer

logger = logging.getLogger(__name__)


def build_profile_text(selected_categories: List[str], additional_interests: str = "") -> str:
    """
    Construye el texto del perfil a partir de categorías seleccionadas e intereses adicionales.
    
    Args:
        selected_categories: Lista de categorías seleccionadas por el usuario
        additional_interests: Texto con intereses adicionales del usuario
        
    Returns:
        Texto del perfil combinado
    """
    profile_text_parts = []
    
    if selected_categories:
        profile_text_parts.append("Intereses seleccionados: " + ", ".join(selected_categories))
    
    additional_interests_clean = (additional_interests or "").strip()
    if additional_interests_clean:
        profile_text_parts.append("Detalle adicional: " + additional_interests_clean)
    
    profile_text = ". ".join(profile_text_parts) if profile_text_parts else "Usuario sin detalles"
    
    return profile_text


def process_additional_interests(additional_interests: str) -> Dict[str, Any]:
    """
    Procesa los intereses adicionales del usuario usando preprocesamiento y anotación.
    
    Args:
        additional_interests: Texto con intereses adicionales
        
    Returns:
        Diccionario con intereses procesados y keywords extraídas
    """
    text_clean = (additional_interests or "").strip()
    if not text_clean:
        return {
            "processed_interests": "",
            "keywords": []
        }
    
    # Preprocesar texto para extraer keywords relevantes
    text_preprocessor = TextPreprocessor(use_spacy=False)
    tokens = text_preprocessor.preprocess(text_clean, return_tokens=True)
    
    # Filtrar tokens muy cortos y obtener keywords más relevantes
    keywords = [token for token in tokens if len(token) > 2][:10]  # Top 10 keywords
    
    # Anotar para detectar categorías adicionales
    annotator = RegexAnnotator()
    annotations = annotator.annotate(text_clean)
    detected_categories = annotations.get("categories", [])
    
    return {
        "processed_interests": text_clean,
        "keywords": keywords,
        "detected_categories": detected_categories
    }


def process_user_profile(
    profile_text: str,
    selected_categories: List[str],
    profile_manager: UserProfileManager,
    profile_vectorizer: UserProfileVectorizer,
    nlp=None
) -> Dict[str, Any]:
    """
    Procesa el perfil completo del usuario: extrae entidades, categorías y vectoriza.
    
    Similar al procesamiento en main.py pero adaptado para el registro de usuarios.
    
    Args:
        profile_text: Texto del perfil del usuario
        selected_categories: Categorías seleccionadas explícitamente por el usuario
        profile_manager: Instancia de UserProfileManager
        profile_vectorizer: Instancia de UserProfileVectorizer
        nlp: Modelo de spaCy para extracción de entidades (opcional)
        
    Returns:
        Diccionario con todos los datos del perfil procesado:
        - profile_text: Texto original del perfil
        - preprocessed_text: Texto preprocesado
        - categories: Categorías detectadas (combinadas de selección y detección)
        - entities: Entidades extraídas con spaCy
        - vector: Vector TF-IDF del perfil
        - annotations: Anotaciones completas del regex annotator
    """
    # Crear perfil base usando UserProfileManager (similar a main.py)
    profile_data = profile_manager.create_profile(profile_text, nlp=nlp)
    
    # Combinar categorías seleccionadas con las detectadas automáticamente
    detected_categories = profile_data.get("categories", [])
    merged_categories = sorted(set(selected_categories + detected_categories))
    
    # Vectorizar el perfil con las categorías combinadas
    # Esto refuerza los términos relevantes en el vector
    profile_vector = profile_vectorizer.vectorize_profile(
        profile_text, 
        merged_categories
    )
    
    # Construir resultado completo con todos los campos necesarios
    result = {
        "profile_text": profile_text,
        "preprocessed_text": profile_data.get("preprocessed_text", ""),
        "categories": merged_categories,
        "entities": profile_data.get("entities", []),
        "vector": profile_vector.tolist(),
        "annotations": profile_data.get("annotations", {})
    }
    
    return result


def create_complete_user_profile(
    selected_categories: List[str],
    additional_interests: str,
    profile_manager: UserProfileManager,
    profile_vectorizer: UserProfileVectorizer,
    nlp=None
) -> Dict[str, Any]:
    """
    Crea un perfil completo de usuario desde cero.
    
    Combina todas las funciones anteriores para procesar el perfil completo:
    1. Construye el texto del perfil
    2. Procesa intereses adicionales
    3. Procesa el perfil completo con extracción de entidades y vectorización
    
    Args:
        selected_categories: Lista de categorías seleccionadas
        additional_interests: Texto con intereses adicionales
        profile_manager: Instancia de UserProfileManager
        profile_vectorizer: Instancia de UserProfileVectorizer
        nlp: Modelo de spaCy (opcional)
        
    Returns:
        Diccionario con todos los datos del perfil procesado, incluyendo:
        - profile_text: Texto del perfil
        - categories: Categorías combinadas
        - entities: Entidades extraídas
        - vector: Vector del perfil
        - additional_interests: Intereses procesados con keywords
    """
    # 1. Construir texto del perfil
    profile_text = build_profile_text(selected_categories, additional_interests)
    
    # 2. Procesar intereses adicionales
    processed_interests = process_additional_interests(additional_interests)
    
    # 3. Procesar perfil completo
    profile_data = process_user_profile(
        profile_text=profile_text,
        selected_categories=selected_categories,
        profile_manager=profile_manager,
        profile_vectorizer=profile_vectorizer,
        nlp=nlp
    )
    
    # 4. Combinar resultados
    result = {
        **profile_data,
        "additional_interests": processed_interests
    }
    
    return result

