"""
Gestión de perfiles de usuario
"""
import logging
from typing import Dict, List, Optional
from src.nlp.regex_annotator import RegexAnnotator
from src.nlp.preprocessing import TextPreprocessor
from src.recommendation.vectorizer import UserProfileVectorizer

logger = logging.getLogger(__name__)


class UserProfileManager:
    """Gestor de perfiles de usuario"""
    
    def __init__(self, vectorizer: UserProfileVectorizer):
        """
        Inicializa el gestor de perfiles
        
        Args:
            vectorizer: Vectorizador de perfiles
        """
        self.vectorizer = vectorizer
        self.regex_annotator = RegexAnnotator()
        self.preprocessor = TextPreprocessor()
    
    def create_profile(self, profile_text: str, nlp=None) -> Dict:
        """
        Crea un perfil de usuario desde texto
        
        Args:
            profile_text: Texto descriptivo del perfil
            nlp: Modelo de spaCy para extracción de entidades (opcional)
            
        Returns:
            Diccionario con información del perfil
        """
        # Preprocesar texto
        preprocessed = self.preprocessor.preprocess(profile_text, return_tokens=False)
        
        # Anotar con regex para extraer intereses
        annotations = self.regex_annotator.annotate(profile_text)
        categories = annotations.get('categories', [])
        
        # Extraer entidades de interés del usuario si hay nlp disponible
        user_entities = []
        if nlp:
            doc = nlp(profile_text)
            user_entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        
        # Vectorizar
        vector = self.vectorizer.vectorize_profile(profile_text, categories)
        
        return {
            'profile_text': profile_text,
            'preprocessed_text': preprocessed,
            'categories': categories,
            'entities': user_entities,
            'vector': vector.tolist(),
            'annotations': annotations
        }
    
    def update_profile(self, existing_profile: Dict, new_text: str) -> Dict:
        """
        Actualiza un perfil existente
        
        Args:
            existing_profile: Perfil existente
            new_text: Nuevo texto del perfil
            
        Returns:
            Perfil actualizado
        """
        return self.create_profile(new_text)



