"""
Análisis de sentimiento usando NLTK VADER
"""
import logging
from typing import Dict, Optional
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Inicializar VADER
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()


class SentimentAnalyzer:
    """Analizador de sentimiento para texto en español"""
    
    def __init__(self):
        """Inicializa el analizador de sentimiento"""
        self.analyzer = sia
    
    def analyze(self, text: str) -> Dict:
        """
        Analiza el sentimiento de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con scores de sentimiento y clasificación
        """
        if not text:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'label': 'neutral'
            }
        
        try:
            # VADER funciona mejor con texto en inglés, pero puede dar resultados útiles
            # para español también
            scores = self.analyzer.polarity_scores(text)
            
            # Clasificar según el score compuesto
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'compound': compound,
                'pos': scores['pos'],
                'neu': scores['neu'],
                'neg': scores['neg'],
                'label': label
            }
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'label': 'neutral'
            }
    
    def analyze_batch(self, texts: list) -> list:
        """
        Analiza el sentimiento de múltiples textos
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            Lista de resultados de análisis
        """
        return [self.analyze(text) for text in texts]



