"""
Módulo de preprocesamiento de texto
Limpieza, tokenización, lematización y eliminación de stopwords
"""
import re
import logging
from typing import List, Optional, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

logger = logging.getLogger(__name__)

# Descargar recursos de NLTK si no están disponibles
# Nota: Es mejor ejecutar scripts/setup_nltk.py primero
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass  # Se descargará cuando se ejecute setup_nltk.py

try:
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass  # Se descargará cuando se ejecute setup_nltk.py

# Cargar stopwords en español
try:
    STOPWORDS_ES = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS_ES = set(stopwords.words('spanish'))

# Cargar modelo de spaCy
try:
    nlp_spacy = spacy.load("es_core_news_lg")
except OSError:
    logger.warning("Modelo es_core_news_lg no encontrado. Intentando con es_core_news_sm")
    try:
        nlp_spacy = spacy.load("es_core_news_sm")
    except OSError:
        logger.error("No se encontró ningún modelo de spaCy en español. Por favor instálalo con: python -m spacy download es_core_news_lg")
        nlp_spacy = None


class TextPreprocessor:
    """Preprocesador de texto para español"""
    
    def __init__(self, use_spacy: bool = True, remove_stopwords: bool = True):
        """
        Inicializa el preprocesador
        
        Args:
            use_spacy: Si True, usa spaCy para lematización (más preciso pero más lento)
            remove_stopwords: Si True, elimina stopwords
        """
        self.use_spacy = use_spacy and nlp_spacy is not None
        self.remove_stopwords = remove_stopwords
        self.nlp = nlp_spacy if self.use_spacy else None
        
        if self.use_spacy:
            logger.info("Usando spaCy para preprocesamiento")
        else:
            logger.info("Usando NLTK para preprocesamiento")
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto básico
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not isinstance(text, str):
            return ""
        
        # Eliminar patrones de "LEA TAMBIÉN:"
        # Ejemplo: "LEA TAMBIÉN:Argelia mediará con países africanos para resolver conflicto en RD de Congo"
        text = re.sub(r'LEA\s+TAMBIÉN\s*[:.].*?(?=\.|$)', '', text, flags=re.IGNORECASE)

        
        # Convertir a minúsculas
        text = text.lower()
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str, language: str = 'spanish') -> List[str]:
        """
        Tokeniza el texto en palabras
        
        Args:
            text: Texto a tokenizar
            language: Idioma para tokenización
            
        Returns:
            Lista de tokens
        """
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text, language=language)
            return tokens
        except Exception as e:
            logger.error(f"Error en tokenización: {e}")
            return []
    
    def tokenize_sentences(self, text: str, language: str = 'spanish') -> List[str]:
        """
        Tokeniza el texto en oraciones
        
        Args:
            text: Texto a tokenizar
            language: Idioma para tokenización
            
        Returns:
            Lista de oraciones
        """
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text, language=language)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error en tokenización de oraciones: {e}")
            return []
    
    def remove_stopwords_tokens(self, tokens: List[str]) -> List[str]:
        """
        Elimina stopwords de una lista de tokens
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Lista de tokens sin stopwords
        """
        if not self.remove_stopwords:
            return tokens
        
        return [t for t in tokens if t.lower() not in STOPWORDS_ES]
    
    def lemmatize(self, text: str) -> List[str]:
        """
        Lematiza el texto usando spaCy
        
        Args:
            text: Texto a lematizar
            
        Returns:
            Lista de lemas
        """
        if not text or not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
            return lemmas
        except Exception as e:
            logger.error(f"Error en lematización: {e}")
            return []
    
    def preprocess(self, text: str, return_tokens: bool = True) -> List[str]:
        """
        Preprocesa el texto completo: limpia, tokeniza, elimina stopwords y lematiza
        
        Args:
            text: Texto a preprocesar
            return_tokens: Si True, retorna lista de tokens; si False, retorna string
            
        Returns:
            Lista de tokens preprocesados o string preprocesado
        """
        if not text:
            return [] if return_tokens else ""
        
        # Limpiar texto
        cleaned = self.clean_text(text)
        
        if self.use_spacy:
            # Usar spaCy para lematización
            lemmas = self.lemmatize(cleaned)
            if self.remove_stopwords:
                lemmas = [l for l in lemmas if l.lower() not in STOPWORDS_ES]
            # Filtrar tokens muy cortos
            tokens = [t for t in lemmas if len(t) > 1]
        else:
            # Usar NLTK
            tokens = self.tokenize(cleaned)
            if self.remove_stopwords:
                tokens = self.remove_stopwords_tokens(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_full(self, text: str) -> Dict:
        """
        Preprocesa el texto y retorna información completa
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Diccionario con información de preprocesamiento
        """
        if not text:
            return {
                'original': '',
                'cleaned': '',
                'tokens': [],
                'sentences': [],
                'lemmas': [],
                'token_count': 0,
                'sentence_count': 0
            }
        
        cleaned = self.clean_text(text)
        sentences = self.tokenize_sentences(text)
        tokens = self.preprocess(text, return_tokens=True)
        
        result = {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'sentences': sentences,
            'token_count': len(tokens),
            'sentence_count': len(sentences)
        }
        
        if self.use_spacy:
            result['lemmas'] = tokens  # Ya son lemas si usamos spaCy
        else:
            result['lemmas'] = tokens  # En este caso son tokens normales
        
        return result


def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto (simple, basado en palabras comunes)
    
    Args:
        text: Texto a analizar
        
    Returns:
        Código de idioma ('es', 'en', 'unknown')
    """
    if not text:
        return 'unknown'
    
    text_lower = text.lower()
    
    # Palabras comunes en español
    spanish_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'muy', 'años', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esas', 'esos', 'estas', 'estos', 'estas', 'estos'}
    
    # Palabras comunes en inglés
    english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
    
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    spanish_count = len(words.intersection(spanish_words))
    english_count = len(words.intersection(english_words))
    
    if spanish_count > english_count and spanish_count > 3:
        return 'es'
    elif english_count > spanish_count and english_count > 3:
        return 'en'
    else:
        return 'unknown'


def is_spanish(text: str) -> bool:
    """
    Verifica si el texto es en español
    
    Args:
        text: Texto a verificar
        
    Returns:
        True si es español, False en caso contrario
    """
    return detect_language(text) == 'es'

