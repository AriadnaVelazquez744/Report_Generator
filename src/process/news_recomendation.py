"""
Pipeline de procesamiento de recomendaciones de reportes
Procesa el input del usuario, lo combina con su perfil y genera recomendaciones personalizadas
"""
import logging
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.vectorizer import UserProfileVectorizer
from src.recommendation.matcher import NewsMatcher
from src.recommendation.vectorize_articles import load_articles_with_vectors

logger = logging.getLogger(__name__)


def find_user_profile_by_id(user_id: str, users_file_path: str = "Data/Data_users/users.json") -> Optional[Dict[str, Any]]:
    """
    Busca el perfil de un usuario por su ID en el archivo users.json.
    
    Args:
        user_id: ID del usuario a buscar
        users_file_path: Ruta al archivo users.json
        
    Returns:
        Perfil del usuario si se encuentra, None si no
    """
    try:
        with open(users_file_path, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        for user in users_data.get('users', []):
            if user.get('number') == user_id:
                return user
        
        logger.warning(f"Usuario con ID '{user_id}' no encontrado")
        return None
    except Exception as e:
        logger.error(f"Error buscando usuario por ID: {e}")
        return None


def get_sorted_article_directories(base_path: str = "Data/Data_articles") -> List[str]:
    """
    Obtiene los directorios Data_articles<n> ordenados de mayor a menor n.
    
    Args:
        base_path: Ruta base donde buscar los directorios
        
    Returns:
        Lista de rutas completas ordenadas de mayor a menor
    """
    try:
        if not os.path.exists(base_path):
            return []
        
        article_dirs = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith("Data_articles"):
                # Extraer el número del directorio
                suffix = item.replace("Data_articles", "")
                if suffix.isdigit():
                    article_dirs.append((int(suffix), item_path))
        
        # Ordenar por número de mayor a menor
        article_dirs.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in article_dirs]
    except Exception as e:
        logger.error(f"Error obteniendo directorios de artículos: {e}")
        return []


def process_user_input(
    user_input: str,
    text_processor: TextPreprocessor,
    annotator: RegexAnnotator,
    nlp=None
) -> Dict[str, Any]:
    """
    Procesa el input del usuario desde la casilla de texto.
    Similar al procesamiento de artículos en main.py pero para texto del usuario.
    
    Args:
        user_input: Texto ingresado por el usuario en la casilla
        text_processor: Instancia de TextPreprocessor
        annotator: Instancia de RegexAnnotator
        nlp: Modelo de spaCy para extracción de entidades (opcional)
        
    Returns:
        Diccionario con:
        - clean_text: Texto preprocesado
        - categories: Categorías detectadas
        - entities: Entidades extraídas
        - preprocessed_tokens: Tokens preprocesados
    """
    if not user_input or not user_input.strip():
        return {
            'clean_text': '',
            'categories': [],
            'entities': [],
            'preprocessed_tokens': []
        }
    
    try:
        # Preprocesar texto (similar a process_single_article en main.py)
        clean_tokens = text_processor.preprocess(user_input, return_tokens=True)
        clean_text = ' '.join(clean_tokens)
        
        # Procesar con spaCy si está disponible
        current_ents = []
        if nlp:
            doc = nlp(clean_text)
            current_ents = [{'text': e.text, 'label': e.label_} for e in doc.ents]
        
        # Anotar con regex para detectar categorías
        annotations = annotator.annotate(clean_text)
        categories = annotations.get('categories', [])
        
        return {
            'clean_text': clean_text,
            'categories': categories,
            'entities': current_ents,
            'preprocessed_tokens': clean_tokens
        }
    except Exception as e:
        logger.error(f"Error procesando input del usuario: {e}")
        return {
            'clean_text': user_input,
            'categories': [],
            'entities': [],
            'preprocessed_tokens': []
        }


def combine_user_profile_with_input(
    user_profile: Dict[str, Any],
    processed_input: Dict[str, Any],
    profile_vectorizer: UserProfileVectorizer,
    input_weight: float = 0.7
) -> Dict[str, Any]:
    """
    Combina el perfil del usuario con el input actual, dando más peso al input.
    
    Args:
        user_profile: Perfil completo del usuario (con vector, categorías, entidades)
        processed_input: Input procesado del usuario
        profile_vectorizer: Vectorizador de perfiles
        input_weight: Peso del input actual (0.0-1.0). Default 0.7 significa 70% input, 30% perfil
        
    Returns:
        Perfil combinado con:
        - profile_text: Texto combinado
        - categories: Categorías combinadas (input tiene prioridad)
        - entities: Entidades combinadas
        - vector: Vector combinado ponderado
    """
    # Combinar texto: input primero, luego perfil base
    input_text = processed_input.get('clean_text', '')
    base_profile_text = user_profile.get('profile_text', '')
    
    # El texto combinado pondera más el input
    combined_text = f"{input_text} {base_profile_text}"
    
    # Combinar categorías: input tiene prioridad
    input_categories = processed_input.get('categories', [])
    base_categories = user_profile.get('categories', [])
    # Unir y eliminar duplicados, manteniendo orden (input primero)
    combined_categories = list(dict.fromkeys(input_categories + base_categories))
    
    # Combinar entidades: input tiene prioridad
    input_entities = processed_input.get('entities', [])
    base_entities = user_profile.get('entities', [])
    # Unir entidades, evitando duplicados exactos
    seen_entities = set()
    combined_entities = []
    
    # Primero agregar entidades del input
    for ent in input_entities:
        ent_key = (ent.get('text', '').lower(), ent.get('label', ''))
        if ent_key not in seen_entities:
            combined_entities.append(ent)
            seen_entities.add(ent_key)
    
    # Luego agregar entidades del perfil base
    for ent in base_entities:
        ent_key = (ent.get('text', '').lower(), ent.get('label', ''))
        if ent_key not in seen_entities:
            combined_entities.append(ent)
            seen_entities.add(ent_key)
    
    # Combinar vectores con ponderación
    input_vector = profile_vectorizer.vectorize_profile(
        input_text,
        categories=input_categories
    )
    base_vector = np.array(user_profile.get('vector', []))
    
    # Si los vectores tienen dimensiones diferentes, usar solo el input
    if len(base_vector) == 0 or len(base_vector) != len(input_vector):
        combined_vector = input_vector
    else:
        # Combinar con ponderación: input_weight para input, (1-input_weight) para perfil
        combined_vector = (
            input_weight * input_vector + 
            (1 - input_weight) * base_vector
        )
        # Normalizar para mantener escala similar
        norm = np.linalg.norm(combined_vector)
        if norm > 0:
            combined_vector = combined_vector / norm * np.linalg.norm(input_vector)
    
    return {
        'profile_text': combined_text,
        'preprocessed_text': processed_input.get('clean_text', ''),
        'categories': combined_categories,
        'entities': combined_entities,
        'vector': combined_vector.tolist(),
        'original_profile': user_profile,
        'input_data': processed_input
    }


def find_relevant_articles(
    combined_profile: Dict[str, Any],
    articles: List[Dict[str, Any]],
    matcher: NewsMatcher,
    top_k: int = 10,
    prioritize_recent: bool = True,
    timeout_sec: float = 60.0,
    start_time: Optional[float] = None
) -> List[Tuple[Dict[str, Any], float, Dict[str, Any]]]:
    """
    Encuentra los artículos más relevantes usando el matcher.
    Prioriza artículos más actuales si prioritize_recent=True.
    
    Args:
        combined_profile: Perfil combinado del usuario
        articles: Lista de artículos disponibles
        matcher: Instancia de NewsMatcher
        top_k: Número de artículos a retornar
        prioritize_recent: Si True, da más peso a artículos recientes
        
    Returns:
        Lista de tuplas (artículo, score, detalles) ordenadas por relevancia
    """
    import time
    
    min_score = 0.15
    start_time = start_time or time.monotonic()
    
    def _article_recency_key(article: Dict[str, Any]) -> float:
        """
        Intenta derivar una clave de recencia:
        1) Extrae número de carpeta Data_articlesNN de _file_path
        2) Usa timestamp del archivo si existe
        3) Usa id negativo (para que ids más altos sean primero)
        """
        path = article.get("_file_path", "")
        number_hint = -1
        for token in str(path).split(os.sep):
            if token.startswith("Data_articles"):
                suffix = token.replace("Data_articles", "")
                if suffix.isdigit():
                    number_hint = int(suffix)
                    break
        mtime = -1.0
        try:
            if path and os.path.exists(path):
                mtime = os.path.getmtime(path)
        except Exception:
            pass
        article_id = article.get("id", 0)
        # Prioridad: número de carpeta, luego mtime, luego id
        return (number_hint, mtime, article_id)
    
    # Ordenar artículos para revisar primero los más nuevos
    ordered_articles = sorted(
        articles,
        key=_article_recency_key,
        reverse=True  # más nuevos primero
    )
    
    # Recorremos manualmente para respetar timeout
    candidate_results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
    max_candidates = top_k * 2 if prioritize_recent else top_k
    
    for article in ordered_articles:
        # Timeout hard-stop
        if time.monotonic() - start_time > timeout_sec:
            break
        
        score, details = matcher.calculate_score(combined_profile, article)
        if score >= min_score:
            candidate_results.append((article, score, details))
            if len(candidate_results) >= max_candidates:
                # Tenemos suficientes candidatos; seguimos por si mejora con recencia pero respetando timeout
                continue
    
    # Ordenar por score descendente
    matches = sorted(candidate_results, key=lambda x: x[1], reverse=True)
    
    if not prioritize_recent or len(matches) <= top_k:
        # Si no priorizamos recencia o ya tenemos suficientes, retornar directamente
        return matches[:top_k]
    
    # Si priorizamos recencia, reordenar considerando el time_score
    # El matcher ya calcula time_score, así que podemos usarlo para reordenar
    scored_matches = []
    for article, score, details in matches:
        time_score = details.get('time_score', 0.5)
        # Aumentar ligeramente el score de artículos recientes
        # Multiplicar por un factor que favorece recencia
        recency_boost = 1.0 + (time_score - 0.5) * 0.3  # Boost de hasta 15% para artículos muy recientes
        adjusted_score = score * recency_boost
        scored_matches.append((article, adjusted_score, details, score))
    
    # Ordenar por score ajustado
    scored_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar top_k con el formato original
    result = []
    for article, adjusted_score, details, original_score in scored_matches[:top_k]:
        # Mantener el score original en details pero usar el ajustado para ordenamiento
        details['original_score'] = original_score
        details['adjusted_score'] = adjusted_score
        result.append((article, original_score, details))
    
    return result


def find_relevant_articles_with_time_strategy(
    combined_profile: Dict[str, Any],
    matcher: NewsMatcher,
    base_articles_path: str = "Data/Data_articles",
    top_k: int = 10,
    high_relevance_threshold: float = 0.7,
    min_high_relevance_count: int = 7,
    initial_timeout_sec: float = 60.0,
    extended_timeout_sec: float = 30.0
) -> Tuple[List[Tuple[Dict[str, Any], float, Dict[str, Any]]], int]:
    """
    Busca artículos relevantes con estrategia de tiempo basada en los requisitos:
    1. Empieza por Data_articles de mayor n
    2. Después de 60s, revisa si hay >= 7 altamente coincidentes
    3. Si hay, se detiene y devuelve top 10
    4. Si no, busca 30s más y devuelve top 10 con lo encontrado
    
    Args:
        combined_profile: Perfil combinado del usuario
        matcher: Instancia de NewsMatcher
        base_articles_path: Ruta base de los directorios Data_articles
        top_k: Número de artículos a retornar
        high_relevance_threshold: Umbral para considerar "altamente coincidente"
        min_high_relevance_count: Mínimo de artículos altamente coincidentes
        initial_timeout_sec: Tiempo inicial de búsqueda
        extended_timeout_sec: Tiempo adicional de búsqueda
        
    Returns:
        Tupla con:
        - Lista de tuplas (artículo, score, detalles) ordenadas por relevancia
        - Número total de artículos revisados
    """
    start_time = time.monotonic()
    all_results: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
    articles_reviewed = 0
    
    # Obtener directorios ordenados de mayor a menor
    article_dirs = get_sorted_article_directories(base_articles_path)
    
    if not article_dirs:
        logger.warning("No se encontraron directorios de artículos")
        return [], 0
    
    # Fase 1: Búsqueda inicial de 60 segundos
    logger.info(f"Iniciando búsqueda inicial de {initial_timeout_sec}s en {len(article_dirs)} directorios")
    
    for dir_path in article_dirs:
        # Cargar artículos con vectores pre-calculados
        try:
            articles = load_articles_with_vectors(dir_path)
            
            # Procesar artículos de este directorio
            for article in articles:
                articles_reviewed += 1
                # Verificar timeout inicial
                if time.monotonic() - start_time > initial_timeout_sec:
                    break
                
                score, details = matcher.calculate_score(combined_profile, article)
                if score > 0.15:  # Umbral mínimo básico
                    all_results.append((article, score, details))
            
            # Verificar si ya tenemos suficientes artículos altamente coincidentes
            high_relevance_count = sum(1 for _, score, _ in all_results if score >= high_relevance_threshold)
            
            if time.monotonic() - start_time >= initial_timeout_sec:
                logger.info(f"Tiempo inicial de {initial_timeout_sec}s alcanzado")
                logger.info(f"Artículos altamente coincidentes encontrados: {high_relevance_count}")
                
                if high_relevance_count >= min_high_relevance_count:
                    logger.info(f"Criterio cumplido: {high_relevance_count} >= {min_high_relevance_count}. Deteniendo búsqueda.")
                    # Ordenar y devolver top k
                    all_results.sort(key=lambda x: x[1], reverse=True)
                    return all_results[:top_k], articles_reviewed
                else:
                    logger.info(f"Criterio no cumplido. Extendiendo búsqueda por {extended_timeout_sec}s más.")
                    break
        except Exception as e:
            logger.error(f"Error procesando directorio {dir_path}: {e}")
            continue
    
    # Fase 2: Búsqueda extendida de 30 segundos si no se cumplió el criterio
    extended_start_time = time.monotonic()
    total_timeout = initial_timeout_sec + extended_timeout_sec
    
    logger.info(f"Iniciando búsqueda extendida hasta {total_timeout}s total")
    
    # Continuar buscando desde donde nos quedamos
    for dir_path in article_dirs:
        # Cargar artículos con vectores pre-calculados
        try:
            articles = load_articles_with_vectors(dir_path)
            
            # Evitar duplicados
            article_ids = [a[0].get('id', '') for a in all_results]
            articles = [a for a in articles if a.get('id', '') not in article_ids]
            
            for article in articles:
                articles_reviewed += 1
                # Verificar timeout total
                if time.monotonic() - start_time > total_timeout:
                    logger.info(f"Tiempo total de {total_timeout}s alcanzado")
                    break
                
                score, details = matcher.calculate_score(combined_profile, article)
                if score > 0.15:
                    all_results.append((article, score, details))
            
            if time.monotonic() - start_time > total_timeout:
                break
        except Exception as e:
            logger.error(f"Error procesando directorio {dir_path}: {e}")
            continue
    
    # Ordenar y devolver top k
    all_results.sort(key=lambda x: x[1], reverse=True)
    final_count = min(len(all_results), top_k)
    logger.info(f"Búsqueda completada. Devolviendo {final_count} artículos más relevantes")
    logger.info(f"Total de artículos revisados: {articles_reviewed}")
    
    return all_results[:top_k], articles_reviewed


def generate_report_recommendations(
    user_input: str,
    user_id: str,
    text_processor: TextPreprocessor,
    annotator: RegexAnnotator,
    profile_vectorizer: UserProfileVectorizer,
    matcher: NewsMatcher,
    nlp=None,
    input_weight: float = 0.7,
    top_k: int = 10,
    users_file_path: str = "Data/Data_users/users.json"
) -> Dict[str, Any]:
    """
    Pipeline completo para generar recomendaciones de reporte basado en ID de usuario.
    
    Este método orquesta todo el proceso:
    1. Busca el perfil del usuario por ID
    2. Procesa el input del usuario
    3. Lo combina con el perfil del usuario (dando más peso al input)
    4. Busca artículos relevantes con estrategia de tiempo optimizada
    5. Retorna el ranking de top_k artículos
    
    Args:
        user_input: Texto ingresado por el usuario
        user_id: ID del usuario para buscar su perfil
        text_processor: Instancia de TextPreprocessor
        annotator: Instancia de RegexAnnotator
        profile_vectorizer: Instancia de UserProfileVectorizer
        matcher: Instancia de NewsMatcher
        nlp: Modelo de spaCy (opcional)
        input_weight: Peso del input actual vs perfil base (0.0-1.0)
        top_k: Número de artículos a retornar
        users_file_path: Ruta al archivo de usuarios
        
    Returns:
        Diccionario con:
        - matches: Lista de tuplas (artículo, score, detalles)
        - combined_profile: Perfil combinado usado para la búsqueda
        - processed_input: Input procesado del usuario
        - user_profile: Perfil encontrado del usuario
        - search_stats: Estadísticas de la búsqueda
    """
    # 1. Buscar perfil del usuario por ID
    user_profile = find_user_profile_by_id(user_id, users_file_path)
    if not user_profile:
        return {
            'error': f'Usuario con ID "{user_id}" no encontrado',
            'matches': [],
            'combined_profile': None,
            'processed_input': None,
            'user_profile': None,
            'search_stats': {}
        }
    
    # 2. Procesar input del usuario
    processed_input = process_user_input(
        user_input,
        text_processor,
        annotator,
        nlp=nlp
    )
    
    # 3. Combinar con perfil del usuario
    combined_profile = combine_user_profile_with_input(
        user_profile,
        processed_input,
        profile_vectorizer,
        input_weight=input_weight
    )
    
    # 4. Buscar artículos relevantes con estrategia de tiempo
    search_start_time = time.monotonic()
    matches, articles_reviewed = find_relevant_articles_with_time_strategy(
        combined_profile,
        matcher,
        top_k=top_k
    )
    search_duration = time.monotonic() - search_start_time
    
    # 5. Preparar estadísticas de búsqueda
    high_relevance_count = sum(1 for _, score, _ in matches if score >= 0.7)
    search_stats = {
        'search_duration_seconds': round(search_duration, 2),
        'total_matches_found': len(matches),
        'high_relevance_matches': high_relevance_count,
        'articles_reviewed': articles_reviewed,
        'user_id': user_id,
        'strategy_used': 'time_based_priority_search'
    }
    
    return {
        'matches': matches,
        'combined_profile': combined_profile,
        'processed_input': processed_input,
        'user_profile': user_profile,
        'search_stats': search_stats,
        'top_k': top_k
    }

