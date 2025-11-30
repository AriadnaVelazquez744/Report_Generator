"""
Motor de matching simplificado v5
- Sin factor de tiempo
- Entidades más estrictas (solo PER, ORG, LOC, GPE con longitud mínima)
- Pesos simples y claros
"""
import math
from typing import List, Dict, Tuple, Optional, Set, TYPE_CHECKING
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from src.recommendation.vectorizer import NewsVectorizer


# Tipos de entidades útiles de spaCy para matching
# PER: Personas (políticos, líderes)
# ORG: Organizaciones (partidos, gobiernos, empresas)
# LOC: Ubicaciones (países, ciudades, regiones)
# GPE: Entidades geopolíticas
ENTITY_TYPES_PRIORITY = {'PER', 'ORG', 'GPE', 'LOC'}


class SimpleMatcher:
    """Matcher simplificado y efectivo"""
    
    def __init__(self, vectorizer: Optional['NewsVectorizer'] = None):
        self.vectorizer = vectorizer
        self._category_idf = {}
        self._avg_doc_length = 200
    
    @classmethod
    def from_articles(cls, articles: List[Dict], vectorizer: Optional['NewsVectorizer'] = None):
        """Crea matcher desde artículos"""
        instance = cls(vectorizer=vectorizer)
        
        # Calcular IDF de categorías
        category_counts = Counter()
        total_docs = len(articles)
        lengths = []
        
        for article in articles:
            for cat in article.get('categories', []):
                category_counts[cat] += 1
            text = article.get('clean_text', '') or article.get('text', '')
            lengths.append(len(text.split()))
        
        # IDF normalizado
        if category_counts and total_docs > 0:
            for cat, count in category_counts.items():
                idf = math.log(total_docs / (count + 1)) + 1
                instance._category_idf[cat] = idf
            
            # Normalizar a [0, 1]
            if instance._category_idf:
                max_idf = max(instance._category_idf.values())
                min_idf = min(instance._category_idf.values())
                range_idf = max_idf - min_idf if max_idf > min_idf else 1
                for cat in instance._category_idf:
                    instance._category_idf[cat] = (instance._category_idf[cat] - min_idf) / range_idf
        
        if lengths:
            instance._avg_doc_length = sum(lengths) / len(lengths)
        
        return instance
    
    def _extract_quality_entities(self, entities: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extrae entidades de calidad agrupadas por tipo.
        Solo mantiene entidades significativas para reducir falsos positivos.
        
        Returns:
            Dict con sets de entidades por tipo: {'PER': {...}, 'ORG': {...}, ...}
        """
        result = {t: set() for t in ENTITY_TYPES_PRIORITY}
        
        for ent in entities:
            label = ent.get('label', '')
            if label not in ENTITY_TYPES_PRIORITY:
                continue
            
            text = ent.get('text', '').strip()
            
            # Filtros estrictos para reducir falsos positivos
            
            # 1. Longitud mínima por tipo
            min_lengths = {'PER': 4, 'ORG': 3, 'GPE': 3, 'LOC': 3}
            if len(text) < min_lengths.get(label, 4):
                continue
            
            # 2. Descartar si es solo números o muy corto sin espacios
            clean = text.replace(' ', '').replace('.', '')
            if clean.isdigit():
                continue
            
            # 3. Para PER: debe tener al menos 2 palabras o ser > 6 chars
            if label == 'PER':
                words = text.split()
                if len(words) < 2 and len(text) < 7:
                    continue
            
            # 4. Normalizar a minúsculas para comparación
            normalized = text.lower().strip()
            
            # 5. Quitar artículos al inicio
            for prefix in ['el ', 'la ', 'los ', 'las ', 'un ', 'una ']:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            
            if len(normalized) >= 3:
                result[label].add(normalized)
        
        return result
    
    def _match_entities(
        self, 
        user_entities: Dict[str, Set[str]], 
        article_entities: Dict[str, Set[str]]
    ) -> Tuple[float, List[str]]:
        """
        Compara entidades por tipo con matching estricto.
        
        Returns:
            (score, lista de matches)
        """
        all_matches = []
        type_scores = []
        
        for ent_type in ENTITY_TYPES_PRIORITY:
            user_set = user_entities.get(ent_type, set())
            article_set = article_entities.get(ent_type, set())
            
            if not user_set or not article_set:
                continue
            
            # Match exacto
            exact = user_set & article_set
            
            # Match por substring (solo para entidades largas)
            substring_matches = set()
            for u in user_set:
                if u in exact:
                    continue
                if len(u) < 6:
                    continue
                for a in article_set:
                    if len(a) < 6:
                        continue
                    # Solo si uno contiene al otro completamente
                    if u in a or a in u:
                        substring_matches.add(u)
                        break
            
            matches = exact | substring_matches
            
            if matches:
                all_matches.extend([f"{ent_type}:{m}" for m in matches])
                # Ponderar por tipo (ORG y GPE son más específicos)
                weight = {'ORG': 1.2, 'GPE': 1.1, 'PER': 1.0, 'LOC': 0.9}.get(ent_type, 1.0)
                type_scores.append(len(matches) * weight)
        
        if not all_matches:
            return 0.0, []
        
        # Score basado en cantidad de matches ponderados
        total_score = sum(type_scores)
        # Normalizar: 1 match = 0.15, 2 = 0.25, 3+ = 0.35 max
        normalized = min(0.35, 0.15 * (1 + math.log(total_score + 1)))
        
        return normalized, all_matches
    
    def _calculate_category_score(
        self, 
        user_cats: List[str], 
        article_cats: List[str]
    ) -> Tuple[float, List[str]]:
        """Score de categorías ponderado por IDF"""
        if not user_cats or not article_cats:
            return 0.0, []
        
        common = set(user_cats) & set(article_cats)
        if not common:
            return 0.0, []
        
        # Score ponderado por IDF
        score = 0.0
        max_score = 0.0
        
        for cat in user_cats:
            idf = self._category_idf.get(cat, 0.5)
            max_score += idf
            if cat in common:
                score += idf
        
        if max_score == 0:
            return 0.0, list(common)
        
        base = score / max_score
        
        # Bonus por múltiples matches
        bonus = min(1.3, 1.0 + 0.1 * len(common))
        
        return min(1.0, base * bonus), list(common)
    
    def _calculate_semantic_score(
        self,
        user_vector: np.ndarray,
        article_vector: np.ndarray
    ) -> float:
        """Similitud coseno simple"""
        if len(user_vector) != len(article_vector):
            min_dim = min(len(user_vector), len(article_vector))
            user_vector = user_vector[:min_dim]
            article_vector = article_vector[:min_dim]
        
        if np.linalg.norm(user_vector) == 0 or np.linalg.norm(article_vector) == 0:
            return 0.0
        
        sim = cosine_similarity([user_vector], [article_vector])[0][0]
        
        # Transformación sigmoidea suave para expandir rango útil
        # Valores < 0.1 → ~0, valores > 0.3 → ~1
        return 1 / (1 + math.exp(-10 * (sim - 0.15)))
    
    def calculate_score(
        self,
        user_profile: Dict,
        article: Dict
    ) -> Tuple[float, Dict]:
        """
        Calcula score de relevancia.
        
        Pesos:
        - Semántico (TF-IDF): 50%
        - Categorías: 35%
        - Entidades: 15%
        """
        user_vector = np.array(user_profile.get('vector', []))
        article_vector = np.array(article.get('vector', []))
        
        if len(article_vector) == 0:
            return 0.0, {}
        
        # 1. Score semántico
        semantic = self._calculate_semantic_score(user_vector, article_vector)
        
        # 2. Score de categorías
        category, matching_cats = self._calculate_category_score(
            user_profile.get('categories', []),
            article.get('categories', [])
        )
        
        # 3. Score de entidades (solo como señal adicional, no determinante)
        user_ents = self._extract_quality_entities(user_profile.get('entities', []))
        article_ents = self._extract_quality_entities(article.get('entidades', []))
        entity, matching_ents = self._match_entities(user_ents, article_ents)
        
        # Pesos fijos - entidades tienen peso bajo para evitar falsos positivos
        # Las entidades solo dan un pequeño bonus, no determinan relevancia
        w_semantic = 0.55
        w_category = 0.40
        w_entity = 0.05  # Muy bajo - solo señal complementaria
        
        final = w_semantic * semantic + w_category * category + w_entity * entity
        
        details = {
            'semantic_score': round(semantic, 4),
            'category_score': round(category, 4),
            'entity_score': round(entity, 4),
            'matching_categories': matching_cats,
            'matching_entities': matching_ents,
            'final_score': round(final, 4)
        }
        
        return final, details
    
    def match_articles(
        self,
        user_profile: Dict,
        articles: List[Dict],
        top_k: int = 10,
        min_score: float = 0.15
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Encuentra artículos relevantes.
        
        Args:
            user_profile: Perfil del usuario
            articles: Lista de artículos
            top_k: Número máximo de resultados
            min_score: Score mínimo para incluir
            
        Returns:
            Lista de (artículo, score, detalles)
        """
        results = []
        
        for article in articles:
            score, details = self.calculate_score(user_profile, article)
            
            if score >= min_score:
                results.append((article, score, details))
        
        # Ordenar por score descendente
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Aplicar diversidad simple: evitar artículos muy similares
        if len(results) > top_k:
            diverse_results = [results[0]]
            
            for article, score, details in results[1:]:
                if len(diverse_results) >= top_k:
                    break
                
                # Verificar que no sea muy similar a los ya seleccionados
                article_vec = np.array(article.get('vector', []))
                is_diverse = True
                
                for selected, _, _ in diverse_results:
                    selected_vec = np.array(selected.get('vector', []))
                    if len(article_vec) == len(selected_vec):
                        sim = cosine_similarity([article_vec], [selected_vec])[0][0]
                        if sim > 0.8:  # Muy similar
                            is_diverse = False
                            break
                
                if is_diverse:
                    diverse_results.append((article, score, details))
            
            return diverse_results
        
        return results[:top_k]


# Alias para compatibilidad
NewsMatcher = SimpleMatcher
