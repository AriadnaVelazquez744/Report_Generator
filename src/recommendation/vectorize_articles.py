"""
Módulo para vectorizar artículos y guardar vectores TF-IDF en los archivos JSON
Mejora el tiempo de búsqueda evitando recalcular vectores en cada consulta

Uso simple:
    from src.recommendation.vectorize_articles import vectorize_articles_directory
    from src.recommendation.vectorizer import NewsVectorizer
    
    vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
    count = vectorize_articles_directory("data_example", vectorizer)
"""
import os
import json
import logging
from typing import List, Dict, Optional, Any
import numpy as np
import tqdm

from src.recommendation.vectorizer import NewsVectorizer

logger = logging.getLogger(__name__)


def load_articles_with_vectors(directory_path: str) -> List[Dict[str, Any]]:
    """
    Carga artículos desde un directorio, asegurando que tengan vectores pre-calculados.
    Si un artículo no tiene vector, lo omite (ya debería estar vectorizado).
    
    Args:
        directory_path: Ruta al directorio con artículos JSON
        
    Returns:
        Lista de artículos que tienen vectores pre-calculados
    """
    articles = []
    
    if not os.path.exists(directory_path):
        logger.warning(f"Directorio no existe: {directory_path}")
        return []
    
    try:
        for filename in sorted(os.listdir(directory_path)):
            if filename.endswith('.json') and filename.startswith('article_'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                        
                        # Verificar que tiene vector pre-calculado
                        if 'vector' in article and isinstance(article.get('vector'), list):
                            article['_file_path'] = file_path
                            # Extract date if missing
                            if 'date' not in article and 'source_metadata' in article:
                                article['date'] = article['source_metadata'].get('date')
                            articles.append(article)
                        else:
                            logger.debug(f"Artículo sin vector: {filename}")
                except Exception as e:
                    logger.warning(f"Error cargando {file_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error procesando directorio {directory_path}: {e}")
        return []
    
    logger.info(f"✅ {len(articles)} artículos cargados con vectores desde {directory_path}")
    return articles


def extract_clean_text_from_article(article: Dict[str, Any]) -> Optional[str]:
    """
    Extrae el texto limpio de un artículo usando preprocessing existente
    Evita cualquier reprocesamiento innecesario
    
    Args:
        article: Diccionario del artículo
        
    Returns:
        Texto limpio, o None si no se puede obtener
    """
    try:
        # Usar preprocessing existente si está disponible
        preprocessing = article.get('preprocessing', {})
        if preprocessing and 'cleaned' in preprocessing:
            clean_text = preprocessing.get('cleaned', '')
            if clean_text:
                return clean_text
        
        # Fallback: usar texto original si no hay preprocessing
        text = article.get('text', '')
        if text:
            clean_text = text.lower()
            clean_text = ' '.join(clean_text.split())
            return clean_text
        
        logger.warning(f"No se pudo extraer texto de: {article.get('_file_path', 'unknown')}")
        return None
    except Exception as e:
        logger.error(f"Error extrayendo texto: {e}")
        return None


def vectorize_articles_directory(
    directory_path: str,
    news_vectorizer: NewsVectorizer
) -> int:
    """
    Vectoriza todos los artículos en un directorio y guarda los vectores en los archivos JSON
    
    Uso:
        from src.recommendation.vectorize_articles import vectorize_articles_directory
        from src.recommendation.vectorizer import NewsVectorizer
        
        vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
        count = vectorize_articles_directory("data_example", vectorizer)
    
    Args:
        directory_path: Ruta al directorio con artículos JSON
        news_vectorizer: Vectorizador TF-IDF ya ajustado
        
    Returns:
        Número de artículos vectorizados exitosamente
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directorio no existe: {directory_path}")
        return 0
    
    # Cargar artículos
    articles = []
    article_files = sorted([f for f in os.listdir(directory_path) 
                           if f.endswith('.json') and f.startswith('article_')])
    
    if not article_files:
        logger.error(f"No se encontraron artículos en: {directory_path}")
        return 0
    
    logger.info(f"Cargando {len(article_files)} artículos desde {directory_path}...")
    
    for filename in article_files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
                article['_file_path'] = file_path
                articles.append(article)
        except Exception as e:
            logger.warning(f"Error cargando {file_path}: {e}")
            continue
    
    if not articles:
        logger.error("No se cargaron artículos")
        return 0
    
    logger.info(f"✅ {len(articles)} artículos cargados")
    
    # Extraer textos limpios
    logger.info(f"Extrayendo textos limpios...")
    clean_texts = []
    articles_with_text = []
    
    for article in tqdm.tqdm(articles, desc="Extrayendo textos"):
        clean_text = extract_clean_text_from_article(article)
        if clean_text:
            clean_texts.append(clean_text)
            articles_with_text.append(article)
        else:
            logger.warning(f"No se pudo extraer texto de: {article.get('_file_path', 'unknown')}")
    
    if not clean_texts:
        logger.error("No hay textos limpios para vectorizar")
        return 0
    
    logger.info(f"✅ {len(clean_texts)} artículos listos para vectorizar")
    
    # Vectorizar
    logger.info(f"Vectorizando {len(clean_texts)} artículos con TF-IDF...")
    article_matrix = news_vectorizer.fit_transform0(clean_texts)
    logger.info(f"✅ Matriz de artículos: {article_matrix.shape}")
    
    # Guardar vectores en archivos
    logger.info(f"Guardando vectores en archivos JSON...")
    saved_count = 0
    
    for i, article in enumerate(tqdm.tqdm(articles_with_text, desc="Guardando vectores")):
        try:
            file_path = article.get('_file_path')
            if not file_path:
                logger.warning("Artículo sin ruta de archivo")
                continue
            
            # Agregar vector y metadata
            article['vector'] = article_matrix[i].tolist()
            article['vectorization_metadata'] = {
                'vectorizer_type': 'TF-IDF',
                'vector_dimension': len(article_matrix[i]),
                'processing_timestamp': __import__('datetime').datetime.now().isoformat()
            }
            
            # Guardar archivo
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
            saved_count += 1
            logger.debug(f"✅ Guardado: {file_path}")
        
        except Exception as e:
            logger.error(f"Error guardando artículo {file_path}: {e}")
            continue
    
    logger.info(f"✅ {saved_count}/{len(articles_with_text)} artículos vectorizados y guardados")
    return saved_count
