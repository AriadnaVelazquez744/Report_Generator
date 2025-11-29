"""
Vectorización de noticias y perfiles de usuario
"""
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class NewsVectorizer:
    """Vectorizador de noticias usando TF-IDF mejorado"""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Inicializa el vectorizador con parámetros optimizados
        
        Args:
            max_features: Número máximo de features
            ngram_range: Rango de n-gramas
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Ya se eliminan en preprocesamiento
            lowercase=True,
            # Parámetros adicionales para mejor representación:
            min_df=2,  # Ignorar términos que aparecen en menos de 2 documentos
            max_df=0.85,  # Ignorar términos que aparecen en más del 85% de documentos
            sublinear_tf=True,  # Usar 1 + log(tf) en vez de tf (reduce impacto de frecuencias altas)
            norm='l2',  # Normalización L2 para cosine similarity
        )
        self.fitted = False
    
    def fit0(self, texts: List[str]):
        """
        Ajusta el vectorizador con textos
        
        Args:
            texts: Lista de textos preprocesados
        """
        self.vectorizer.fit(texts)
        self.fitted = True
    
    def transform0(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos a vectores
        
        Args:
            texts: Lista de textos preprocesados
            
        Returns:
            Matriz de vectores
        """
        if not self.fitted:
            raise ValueError("El vectorizador debe ser ajustado primero con fit()")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform0(self, texts: List[str]) -> np.ndarray:
        """
        Ajusta y transforma textos
        
        Args:
            texts: Lista de textos preprocesados
            
        Returns:
            Matriz de vectores
        """
        self.fitted = True
        return self.vectorizer.fit_transform(texts).toarray()
    
    def vectorize_article(self, text: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Vectoriza un artículo individual
        
        Args:
            text: Texto del artículo
            metadata: Metadatos adicionales (sección, tags, categorías)
            
        Returns:
            Vector del artículo
        """
       
        if not self.fitted:
            raise ValueError("El vectorizador debe ser ajustado primero con fit()")
        
        # Si hay metadatos, podrían agregarse como features adicionales
        # Por ahora, solo retornamos el vector TF-IDF
        
        vector = self.transform0([text])[0]

        return vector
    
    def get_feature_names(self) -> List[str]:
        """Obtiene los nombres de las features"""
        if not self.fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary(self) -> Dict:
        """Obtiene el vocabulario del vectorizador"""
        if not self.fitted:
            return {}
        return self.vectorizer.vocabulary_
    
    def set_vocabulary(self, vocabulary: Dict):
        """Establece el vocabulario del vectorizador sin reentrenar"""
        self.vectorizer.vocabulary_ = vocabulary
        self.fitted = True
    
    def save(self, filepath: str):
        """Guarda el vectorizador completo en un archivo pickle"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    @classmethod
    def load(cls, filepath: str, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Carga un vectorizador desde un archivo pickle"""
        import pickle
        instance = cls(max_features=max_features, ngram_range=ngram_range)
        with open(filepath, 'rb') as f:
            instance.vectorizer = pickle.load(f)
        instance.fitted = True
        return instance
    
    def to_dict(self) -> Dict:
        """Serializa el vectorizador a un diccionario JSON-compatible"""
        import pickle
        import base64
        if not self.fitted:
            return {}
        # Serializar el vectorizador completo a bytes y luego a base64
        vectorizer_bytes = pickle.dumps(self.vectorizer)
        return {
            'vectorizer_b64': base64.b64encode(vectorizer_bytes).decode('ascii'),
            'max_features': self.vectorizer.max_features,
            'ngram_range': self.vectorizer.ngram_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Deserializa el vectorizador desde un diccionario"""
        import pickle
        import base64
        if not data or 'vectorizer_b64' not in data:
            return None
        
        # Decodificar de base64 y deserializar
        vectorizer_bytes = base64.b64decode(data['vectorizer_b64'])
        vectorizer_obj = pickle.loads(vectorizer_bytes)
        
        # Crear instancia y asignar
        instance = cls(
            max_features=data.get('max_features', 5000),
            ngram_range=tuple(data.get('ngram_range', (1, 2)))
        )
        instance.vectorizer = vectorizer_obj
        instance.fitted = True
        return instance


class UserProfileVectorizer:
    """Vectorizador de perfiles de usuario con expansión de categorías"""
    
    def __init__(self, news_vectorizer: NewsVectorizer):
        """
        Inicializa el vectorizador de perfiles
        
        Args:
            news_vectorizer: Vectorizador de noticias ya ajustado
        """
        self.news_vectorizer = news_vectorizer
    
    def vectorize_profile(self, profile_text: str, categories: Optional[List[str]] = None) -> np.ndarray:
        """
        Vectoriza un perfil de usuario con expansión de categorías
        
        Las categorías detectadas se añaden al texto para reforzar
        los términos relevantes en el vector TF-IDF.
        
        Args:
            profile_text: Texto del perfil del usuario
            categories: Categorías de interés del usuario
            
        Returns:
            Vector del perfil
        """
        # Expandir el texto con las categorías (repetidas para darles peso)
        expanded_text = profile_text
        
        if categories:
            # Añadir categorías como términos adicionales (2 veces para refuerzo)
            category_text = ' '.join(categories) * 2
            expanded_text = f"{profile_text} {category_text}"
        
        # Usar el mismo vectorizador que las noticias
        vector = self.news_vectorizer.vectorize_article(expanded_text)
        
        return vector



