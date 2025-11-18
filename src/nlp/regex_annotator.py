"""
Sistema de anotación con expresiones regulares
"""
import logging
from typing import List, Dict, Set
from src.nlp.regex_patterns import regex_extraccion

logger = logging.getLogger(__name__)


class RegexAnnotator:
    """Anotador de texto usando expresiones regulares"""
    
    def __init__(self):
        """Inicializa el anotador con los patrones compilados"""
        self.patterns = regex_extraccion
    
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
        
        # Convertir a minúsculas para matching
        text_lower = text.lower()
        
        categories_detected: Set[str] = set()
        matches: List[Dict] = []
        texts_seen: Set[str] = set()
        
        # Iterar sobre todas las categorías y patrones
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                # Buscar todas las coincidencias
                for match in pattern.finditer(text_lower):
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



