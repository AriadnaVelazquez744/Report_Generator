"""
Vectorización de noticias y perfiles de usuario
"""
import logging
from typing import List, Dict, Optional, Tuple
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
    
    def get_idf_values(self) -> np.ndarray:
        """Obtiene los valores IDF para cada término del vocabulario"""
        if not self.fitted:
            return np.array([])
        return self.vectorizer.idf_
    
    def get_term_idf(self, term: str) -> float:
        """Obtiene el IDF de un término específico"""
        if not self.fitted:
            return 0.0
        vocab = self.vectorizer.vocabulary_
        if term.lower() in vocab:
            idx = vocab[term.lower()]
            return float(self.vectorizer.idf_[idx])
        return 0.0
    
    def get_document_frequencies(self) -> Dict[str, int]:
        """
        Calcula la frecuencia de documento para cada término
        DF = N / exp(IDF - 1) aproximadamente
        """
        if not self.fitted:
            return {}
        
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_
        n_docs = len(idf)  # Aproximación
        
        # Invertir IDF para obtener DF: df = n_docs / exp(idf)
        # Con sublinear_tf y smoothing, la fórmula es: idf = log((n+1)/(df+1)) + 1
        # Entonces: df = (n+1) / exp(idf - 1) - 1
        df_dict = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for term, idx in vocab.items():
            # Aproximar DF desde IDF
            idf_val = idf[idx]
            # Invertir la fórmula de sklearn con smooth_idf=True
            df_approx = max(1, int(n_docs / np.exp(idf_val - 1)))
            df_dict[term] = df_approx
        
        return df_dict
    
    def compute_bm25_score(self, query_vector: np.ndarray, doc_vector: np.ndarray,
                           doc_length: int, avg_doc_length: float,
                           k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calcula BM25 score entre query y documento
        
        BM25 es mejor que cosine similarity para recuperación de información
        porque considera la longitud del documento y satura los TF altos.
        
        Args:
            query_vector: Vector TF-IDF del query
            doc_vector: Vector TF-IDF del documento  
            doc_length: Longitud del documento en palabras
            avg_doc_length: Longitud promedio de documentos
            k1: Parámetro de saturación de TF (típico: 1.2-2.0)
            b: Parámetro de normalización por longitud (típico: 0.75)
        """
        if not self.fitted:
            return 0.0
        
        idf = self.vectorizer.idf_
        
        # Normalización de longitud
        len_norm = 1 - b + b * (doc_length / max(1, avg_doc_length))
        
        score = 0.0
        # Solo procesar términos no-cero del query
        query_nonzero = np.nonzero(query_vector)[0]
        
        for idx in query_nonzero:
            if idx < len(doc_vector) and idx < len(idf):
                tf_doc = doc_vector[idx]
                tf_query = query_vector[idx]
                idf_val = idf[idx]
                
                # BM25 scoring
                numerator = tf_doc * (k1 + 1)
                denominator = tf_doc + k1 * len_norm
                
                if denominator > 0:
                    score += idf_val * (numerator / denominator) * tf_query
        
        return score
    
    def find_similar_terms(self, term: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Encuentra términos similares en el vocabulario basándose en
        co-ocurrencia implícita (términos con IDF similar en contextos similares)
        """
        if not self.fitted or term.lower() not in self.vectorizer.vocabulary_:
            return []
        
        vocab = self.vectorizer.vocabulary_
        feature_names = self.vectorizer.get_feature_names_out()
        target_idx = vocab[term.lower()]
        target_idf = self.vectorizer.idf_[target_idx]
        
        # Encontrar términos con IDF similar (heurística simple)
        similarities = []
        for t, idx in vocab.items():
            if t == term.lower():
                continue
            idf_diff = abs(self.vectorizer.idf_[idx] - target_idf)
            # Similitud inversa a la diferencia de IDF
            sim = 1.0 / (1.0 + idf_diff)
            similarities.append((t, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_top_terms_for_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Obtiene los términos más importantes de un vector TF-IDF
        """
        if not self.fitted:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Obtener índices de los valores más altos
        top_indices = np.argsort(vector)[-top_k:][::-1]
        
        result = []
        for idx in top_indices:
            if vector[idx] > 0:
                result.append((feature_names[idx], float(vector[idx])))
        
        return result
    
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
    """Vectorizador de perfiles de usuario con expansión de categorías y query expansion"""
    
    def __init__(self, news_vectorizer: NewsVectorizer):
        """
        Inicializa el vectorizador de perfiles
        
        Args:
            news_vectorizer: Vectorizador de noticias ya ajustado
        """
        self.news_vectorizer = news_vectorizer
    
    def expand_query_terms(self, terms: List[str], expansion_weight: float = 0.3) -> str:
        """
        Expande términos del query con términos relacionados del vocabulario
        
        Args:
            terms: Lista de términos originales
            expansion_weight: Peso relativo de términos expandidos
        """
        expanded_terms = list(terms)
        
        for term in terms:
            similar = self.news_vectorizer.find_similar_terms(term, top_k=2)
            for sim_term, score in similar:
                if score > 0.5:  # Solo términos muy similares
                    # Agregar con menor frecuencia para dar menos peso
                    expanded_terms.append(sim_term)
        
        return ' '.join(expanded_terms)
    
    def vectorize_profile(self, profile_text: str, categories: Optional[List[str]] = None,
                         use_expansion: bool = True) -> np.ndarray:
        """
        Vectoriza un perfil de usuario con expansión de categorías
        
        Las categorías detectadas se añaden al texto para reforzar
        los términos relevantes en el vector TF-IDF.
        
        Args:
            profile_text: Texto del perfil del usuario
            categories: Categorías de interés del usuario
            use_expansion: Si usar query expansion
            
        Returns:
            Vector del perfil
        """
        expanded_text = profile_text
        
        if categories:
            # Expandir categorías si está habilitado
            if use_expansion:
                category_text = self.expand_query_terms(categories)
            else:
                category_text = ' '.join(categories)
            
            # Añadir categorías con peso (repetidas)
            expanded_text = f"{profile_text} {category_text} {category_text}"
        
        # Usar el mismo vectorizador que las noticias
        vector = self.news_vectorizer.vectorize_article(expanded_text)
        
        return vector
    
    def get_profile_key_terms(self, profile_vector: np.ndarray, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Obtiene los términos clave del perfil del usuario
        Útil para debugging y explicabilidad
        """
        return self.news_vectorizer.get_top_terms_for_vector(profile_vector, top_k)



