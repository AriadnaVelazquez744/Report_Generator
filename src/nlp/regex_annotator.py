"""
Sistema de anotación con expresiones regulares
"""
import logging
from typing import List, Dict, Set

from regex_patterns import regex_extraccion
from preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class RegexAnnotator:
    """Anotador de texto usando expresiones regulares"""
    
    def __init__(self, use_lemmatization: bool = True):
        """Inicializa el anotador con los patrones compilados"""
        self.patterns = regex_extraccion
        self.use_lemmatization = use_lemmatization
        self.preprocessor = None

        if self.use_lemmatization:
            try:
                # No eliminar stopwords aquí para conservar el máximo contexto semántico
                self.preprocessor = TextPreprocessor(use_spacy=True, remove_stopwords=False)
                if not self.preprocessor.nlp:
                    # spaCy no disponible, desactivar lematización
                    self.use_lemmatization = False
                    self.preprocessor = None
                    logger.warning("RegexAnnotator: spaCy no disponible, se desactiva la lematización")
            except Exception as e:
                logger.warning(f"RegexAnnotator: no se pudo inicializar TextPreprocessor, se desactiva la lematización: {e}")
                self.use_lemmatization = False
                self.preprocessor = None
    
    def annotate(self, text: str) -> Dict:
        """
        Anota un texto con las categorías detectadas
        
        Args:
            text: Texto a anotar
            
        Returns:
            Diccionario con categorías detectadas y detalles
        """
        if not text:
            return {
                'categories': [],
                'matches': [],
                'match_count': 0
            }
        
        # Texto base en minúsculas para matching
        text_for_match = text.lower()

        # Opcionalmente, ampliar el texto con una versión lematizada
        if self.use_lemmatization and self.preprocessor and self.preprocessor.nlp:
            try:
                # Usar remove_noise para limpiar sin perder estructura
                cleaned = self.preprocessor.remove_noise(text)
                doc = self.preprocessor.nlp(cleaned)
                lemmas = []
                for token in doc:
                    if token.is_space or token.is_punct:
                        continue
                    lemma = (token.lemma_ or token.text).lower()
                    if lemma:
                        lemmas.append(lemma)
                if lemmas:
                    # Añadir los lemas al final del texto base para mejorar el recall
                    text_for_match = text_for_match + " " + " ".join(lemmas)
            except Exception as e:
                logger.error(f"RegexAnnotator: error durante la lematización, se continúa sin lemas: {e}")
        
        
        categories_detected: Set[str] = set()
        matches: List[Dict] = []
        texts_seen: Set[str] = set()
        
        # Iterar sobre todas las categorías y patrones
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                # Buscar todas las coincidencias sobre el texto ampliado
                for match in pattern.finditer(text_for_match):
                    text_match = match.group().strip()
                    
                    # Evitar duplicados exactos
                    if text_match in texts_seen:
                        continue
                    
                    texts_seen.add(text_match)
                    categories_detected.add(category)
                    
                    matches.append({
                        'category': category,
                        'matched_text': text_match,
                        'position': match.span(),
                        'start': match.start(),
                        'end': match.end()
                    })
                    
                    # Solo la primera coincidencia por patrón
                    break
        
        return {
            'categories': list(categories_detected),
            'matches': matches,
            'match_count': len(matches)
        }
    
    def annotate_batch(self, texts: List[str]) -> List[Dict]:
        """
        Anota múltiples textos
        
        Args:
            texts: Lista de textos a anotar
            
        Returns:
            Lista de resultados de anotación
        """
        results = []
        for text in texts:
            result = self.annotate(text)
            results.append(result)
        return results
    
    def get_category_stats(self, annotations: List[Dict]) -> Dict[str, int]:
        """
        Obtiene estadísticas de categorías de una lista de anotaciones
        
        Args:
            annotations: Lista de resultados de anotación
            
        Returns:
            Diccionario con conteo de categorías
        """
        stats = {}
        for annotation in annotations:
            for category in annotation.get('categories', []):
                stats[category] = stats.get(category, 0) + 1
        return stats



