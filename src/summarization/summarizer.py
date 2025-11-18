"""
Generación de resúmenes extractivos y abstractivos
"""
import logging
from typing import List, Optional
from nltk.tokenize import sent_tokenize
import numpy as np

logger = logging.getLogger(__name__)


class TextRankSummarizer:
    """Resumidor extractivo usando algoritmo TextRank simplificado"""
    
    def __init__(self, language: str = 'spanish'):
        """
        Inicializa el resumidor
        
        Args:
            language: Idioma del texto
        """
        self.language = language
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Genera un resumen extractivo del texto
        
        Args:
            text: Texto a resumir
            num_sentences: Número de oraciones en el resumen
            
        Returns:
            Texto resumido
        """
        if not text:
            return ""
        
        try:
            # Tokenizar en oraciones
            sentences = sent_tokenize(text, language=self.language)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Calcular scores simples basados en longitud y posición
            # (versión simplificada de TextRank)
            scores = []
            for i, sentence in enumerate(sentences):
                # Score basado en posición (primeras oraciones más importantes)
                position_score = 1.0 / (i + 1)
                # Score basado en longitud (oraciones medianas más importantes)
                length_score = 1.0 / (1.0 + abs(len(sentence.split()) - 15))
                # Score combinado
                score = position_score * 0.6 + length_score * 0.4
                scores.append((score, i, sentence))
            
            # Ordenar por score y tomar las mejores
            scores.sort(reverse=True)
            top_sentences = sorted(scores[:num_sentences], key=lambda x: x[1])
            
            # Reconstruir texto manteniendo orden original
            summary = ' '.join([sent for _, _, sent in top_sentences])
            
            return summary
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            # Fallback: retornar primeras oraciones
            sentences = sent_tokenize(text, language=self.language)
            return ' '.join(sentences[:num_sentences])


class PersonalizedSummarizer:
    """Resumidor personalizado que destaca aspectos relevantes al perfil"""
    
    def __init__(self, base_summarizer: TextRankSummarizer):
        """
        Inicializa el resumidor personalizado
        
        Args:
            base_summarizer: Resumidor base
        """
        self.base_summarizer = base_summarizer
    
    def summarize_for_profile(
        self,
        text: str,
        user_categories: List[str],
        num_sentences: int = 3
    ) -> str:
        """
        Genera un resumen personalizado destacando aspectos relevantes
        
        Args:
            text: Texto a resumir
            user_categories: Categorías de interés del usuario
            num_sentences: Número de oraciones en el resumen
            
        Returns:
            Texto resumido personalizado
        """
        if not text:
            return ""
        
        try:
            sentences = sent_tokenize(text, language='spanish')
            
            if len(sentences) <= num_sentences:
                return text
            
            # Calcular relevancia de cada oración
            scores = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Score base
                position_score = 1.0 / (i + 1)
                length_score = 1.0 / (1.0 + abs(len(sentence.split()) - 15))
                base_score = position_score * 0.4 + length_score * 0.3
                
                # Bonus por categorías relevantes
                relevance_bonus = 0.0
                if user_categories:
                    for category in user_categories:
                        # Buscar palabras clave de la categoría en la oración
                        category_keywords = category.lower().split('_')
                        for keyword in category_keywords:
                            if keyword in sentence_lower:
                                relevance_bonus += 0.1
                
                final_score = base_score + relevance_bonus * 0.3
                scores.append((final_score, i, sentence))
            
            # Ordenar y seleccionar mejores oraciones
            scores.sort(reverse=True)
            top_sentences = sorted(scores[:num_sentences], key=lambda x: x[1])
            
            summary = ' '.join([sent for _, _, sent in top_sentences])
            return summary
        except Exception as e:
            logger.error(f"Error generando resumen personalizado: {e}")
            return self.base_summarizer.summarize(text, num_sentences)



