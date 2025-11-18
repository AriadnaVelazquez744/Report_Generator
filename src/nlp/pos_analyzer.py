"""
Análisis de POS tags y n-gramas de tags
"""
import logging
from typing import List, Dict
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk

logger = logging.getLogger(__name__)

# Descargar recursos si no están disponibles
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class POSAnalyzer:
    """Analizador de POS tags y n-gramas"""
    
    def __init__(self):
        """Inicializa el analizador POS"""
        pass
    
    def analyze(self, text: str, language: str = 'spanish') -> Dict:
        """
        Analiza POS tags de un texto
        
        Args:
            text: Texto a analizar
            language: Idioma del texto
            
        Returns:
            Diccionario con información de POS tags
        """
        if not text:
            return {
                'tags': [],
                'tag_freq': {},
                'bigrams': [],
                'trigrams': [],
                'bigram_freq': {},
                'trigram_freq': {}
            }
        
        try:
            # Tokenizar
            tokens = word_tokenize(text, language=language)
            tokens = [t for t in tokens if len(t) > 1]
            
            # Obtener POS tags
            tagged = pos_tag(tokens)
            tags = [tag for word, tag in tagged]
            
            # Frecuencia de tags
            tag_freq = Counter(tags)
            
            # Bigramas y trigramas de tags
            tag_bigrams = list(ngrams(tags, 2))
            tag_trigrams = list(ngrams(tags, 3))
            
            bigram_freq = Counter(tag_bigrams)
            trigram_freq = Counter(tag_trigrams)
            
            return {
                'tags': tags,
                'tag_freq': dict(tag_freq),
                'bigrams': tag_bigrams,
                'trigrams': tag_trigrams,
                'bigram_freq': {f"{bg[0]}-{bg[1]}": count for bg, count in bigram_freq.items()},
                'trigram_freq': {f"{tg[0]}-{tg[1]}-{tg[2]}": count for tg, count in trigram_freq.items()}
            }
        except Exception as e:
            logger.error(f"Error analizando POS tags: {e}")
            return {
                'tags': [],
                'tag_freq': {},
                'bigrams': [],
                'trigrams': [],
                'bigram_freq': {},
                'trigram_freq': {}
            }
    
    def get_top_patterns(self, pos_results: Dict, n: int = 10) -> Dict:
        """
        Obtiene los patrones más frecuentes
        
        Args:
            pos_results: Resultados del análisis POS
            n: Número de patrones a retornar
            
        Returns:
            Diccionario con top patrones
        """
        top_tags = dict(Counter(pos_results.get('tag_freq', {})).most_common(n))
        top_bigrams = dict(Counter(pos_results.get('bigram_freq', {})).most_common(n))
        top_trigrams = dict(Counter(pos_results.get('trigram_freq', {})).most_common(n))
        
        return {
            'top_tags': top_tags,
            'top_bigrams': top_bigrams,
            'top_trigrams': top_trigrams
        }



