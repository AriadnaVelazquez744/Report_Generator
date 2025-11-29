"""
Motor de matching y cálculo de relevancia - Sistema mejorado v3
"""
import logging
import math
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NewsMatcher:
    """Motor de matching de noticias con perfiles de usuario - v3 con calidad y diversidad"""
    
    # Umbral mínimo de similitud coseno para considerar un artículo relevante
    MIN_COSINE_THRESHOLD = 0.10
    
    # Longitud mínima de entidad para considerarla significativa
    MIN_ENTITY_LENGTH = 3
    
    # Longitud mínima de artículo (palabras) para considerarlo de calidad
    MIN_ARTICLE_LENGTH = 50
    
    # Entidades comunes a ignorar (stopwords de entidades)
    ENTITY_STOPWORDS = {
        'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'al',
        'que', 'con', 'por', 'para', 'como', 'más', 'sin', 'sobre',
        'país', 'estado', 'gobierno', 'año', 'años', 'día', 'días',
        'presidente', 'ministro', 'director', 'señor', 'señora',
        'hoy', 'ayer', 'mañana', 'semana', 'mes', 'tiempo',
        'parte', 'caso', 'forma', 'manera', 'vez', 'veces',
        'cosa', 'cosas', 'persona', 'personas', 'gente',
        'número', 'total', 'millones', 'miles', 'cientos',
        'nacional', 'internacional', 'mundial', 'local',
    }
    
    def __init__(self, category_frequencies: Optional[Dict[str, int]] = None):
        """
        Inicializa el matcher
        
        Args:
            category_frequencies: Dict con frecuencia de cada categoría en el corpus
        """
        self.category_frequencies = category_frequencies or {}
        self._total_articles = sum(self.category_frequencies.values()) if self.category_frequencies else 0
        
        # Pre-calcular IDF normalizado para cada categoría (con smoothing)
        self._category_idf = {}
        self._max_idf = 1.0
        if self.category_frequencies and self._total_articles > 0:
            for cat, freq in self.category_frequencies.items():
                # log1p para suavizado, evita log(0)
                idf = math.log1p(self._total_articles / (freq + 1))
                self._category_idf[cat] = idf
            # Normalizar IDF al rango [0.1, 1] para evitar extremos
            if self._category_idf:
                self._max_idf = max(self._category_idf.values())
                min_idf = min(self._category_idf.values())
                for cat in self._category_idf:
                    normalized = (self._category_idf[cat] - min_idf) / (self._max_idf - min_idf + 0.01)
                    self._category_idf[cat] = 0.1 + 0.9 * normalized  # Rango [0.1, 1]
    
    def _calculate_article_quality(self, article: Dict) -> float:
        """
        Calcula un factor de calidad del artículo basado en:
        - Longitud del texto
        - Densidad de categorías (keywords relevantes)
        - Presencia de entidades específicas
        
        Returns:
            Factor de calidad entre 0.5 y 1.2
        """
        text = article.get('text', '')
        clean_text = article.get('clean_text', '')
        categories = article.get('categories', [])
        entities = article.get('entidades', [])
        
        # Factor de longitud
        word_count = len(clean_text.split()) if clean_text else len(text.split())
        if word_count < self.MIN_ARTICLE_LENGTH:
            length_factor = 0.6  # Penalizar artículos muy cortos
        elif word_count < 100:
            length_factor = 0.8
        elif word_count < 300:
            length_factor = 1.0
        else:
            length_factor = 1.1  # Bonus para artículos sustanciosos
        
        # Factor de densidad de categorías (keywords por cada 100 palabras)
        category_density = (len(categories) / max(1, word_count)) * 100
        if category_density < 0.5:
            density_factor = 0.8  # Pocas keywords = menos relevante
        elif category_density > 3:
            density_factor = 0.9  # Demasiadas keywords = posible spam
        else:
            density_factor = 1.0
        
        # Factor de entidades específicas
        specific_entities = [e for e in entities 
                            if e.get('label') in {'PER', 'ORG', 'LOC', 'GPE'}
                            and len(e.get('text', '')) >= 4]
        if len(specific_entities) >= 3:
            entity_factor = 1.1  # Artículo con buenas entidades
        elif len(specific_entities) >= 1:
            entity_factor = 1.0
        else:
            entity_factor = 0.95
        
        # Combinar factores
        quality = length_factor * density_factor * entity_factor
        return max(0.5, min(1.2, quality))
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normaliza texto de entidad para comparación"""
        # Limpiar y normalizar
        text = text.strip().lower()
        # Quitar artículos y preposiciones comunes al inicio
        prefixes = ['el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'de ', 'del ']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text
    
    def _filter_entities(self, entities: List[Dict]) -> Set[str]:
        """
        Filtra entidades para quedarse solo con las significativas
        
        - Excluye entidades muy cortas
        - Excluye stopwords de entidades
        - Mantiene el TEXTO específico de la entidad
        """
        filtered = set()
        for ent in entities:
            text = self._normalize_entity_text(ent.get('text', ''))
            label = ent.get('label', 'MISC')
            
            # Longitud mínima (más estricta para MISC)
            min_len = self.MIN_ENTITY_LENGTH if label != 'MISC' else 6
            if len(text) < min_len:
                continue
            
            # Excluir stopwords
            if text in self.ENTITY_STOPWORDS:
                continue
            
            # Excluir si es solo números
            if text.replace(' ', '').isdigit():
                continue
            
            # Excluir entidades de una sola palabra muy comunes
            if ' ' not in text and len(text) < 5:
                continue
            
            filtered.add(text)
        
        return filtered
    
    def _fuzzy_entity_match(self, user_entities: Set[str], article_entities: Set[str]) -> Set[str]:
        """
        Encuentra matches de entidades con coincidencia parcial
        
        Permite matchear:
        - "Universidad de La Habana" con "Universidad Habana"
        - Substrings significativos
        """
        matches = set()
        
        # Match exacto primero
        exact_matches = user_entities & article_entities
        matches.update(exact_matches)
        
        # Match por substring (para entidades multi-palabra)
        for user_ent in user_entities:
            if user_ent in matches:
                continue
            
            # Solo buscar substrings si la entidad tiene al menos 5 chars
            if len(user_ent) < 5:
                continue
                
            for article_ent in article_entities:
                if article_ent in matches:
                    continue
                
                # Si una contiene a la otra (substring significativo)
                if len(user_ent) >= 5 and len(article_ent) >= 5:
                    if user_ent in article_ent or article_ent in user_ent:
                        matches.add(user_ent)
                        break
                
                # Match por palabras compartidas (para entidades multi-palabra)
                user_words = set(user_ent.split())
                article_words = set(article_ent.split())
                
                # Si comparten al menos 2 palabras significativas (>3 chars)
                significant_user = {w for w in user_words if len(w) > 3}
                significant_article = {w for w in article_words if len(w) > 3}
                
                common_words = significant_user & significant_article
                if len(common_words) >= 2:
                    matches.add(user_ent)
                    break
                
                # Si comparten 1 palabra muy específica (>6 chars)
                if len(common_words) == 1:
                    word = list(common_words)[0]
                    if len(word) >= 7:  # Palabra muy específica
                        matches.add(user_ent)
                        break
        
        return matches
    
    def _calculate_category_score(
        self, 
        user_categories: List[str], 
        article_categories: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calcula score de categorías con IDF weighting mejorado
        
        Returns:
            Tuple de (score, categorías coincidentes)
        """
        if not user_categories or not article_categories:
            return 0.0, []
        
        common = set(user_categories) & set(article_categories)
        if not common:
            return 0.0, []
        
        # Calcular score ponderado por IDF (categorías raras valen más)
        weighted_score = 0.0
        max_possible_score = 0.0
        
        for cat in user_categories:
            # IDF normalizado: categorías raras tienen IDF cercano a 1
            # categorías frecuentes tienen IDF cercano a 0.1
            idf = self._category_idf.get(cat, 0.5)  # Default 0.5 para categorías desconocidas
            max_possible_score += idf
            if cat in common:
                weighted_score += idf
        
        if max_possible_score == 0:
            return 0.0, list(common)
        
        # Score base normalizado
        base_score = weighted_score / max_possible_score
        
        # Bonus por cantidad de categorías coincidentes
        # Más categorías coincidentes = más confianza en la relevancia
        num_common = len(common)
        if num_common >= 4:
            quantity_bonus = 1.3
        elif num_common >= 3:
            quantity_bonus = 1.2
        elif num_common >= 2:
            quantity_bonus = 1.1
        else:
            quantity_bonus = 1.0
        
        score = min(1.0, base_score * quantity_bonus)
        
        return score, list(common)
    
    def _calculate_entity_score(
        self,
        user_entities: List[Dict],
        article_entities: List[Dict]
    ) -> Tuple[float, List[str]]:
        """
        Calcula score de entidades con filtrado estricto
        
        Returns:
            Tuple de (score, entidades coincidentes)
        """
        if not user_entities or not article_entities:
            return 0.0, []
        
        # Filtrar entidades
        user_filtered = self._filter_entities(user_entities)
        article_filtered = self._filter_entities(article_entities)
        
        if not user_filtered or not article_filtered:
            return 0.0, []
        
        # Usar fuzzy matching para encontrar coincidencias
        common = self._fuzzy_entity_match(user_filtered, article_filtered)
        
        if not common:
            return 0.0, []
        
        # Score basado en cantidad y especificidad de matches
        num_common = len(common)
        
        # Calcular "especificidad" promedio de las entidades matcheadas
        avg_specificity = sum(len(e) for e in common) / num_common
        
        if num_common >= 3:
            # 3+ entidades: match fuerte
            score = 0.30
        elif num_common == 2:
            # 2 entidades: match moderado
            score = 0.20
        elif num_common == 1:
            # 1 entidad: depende de especificidad
            entity = list(common)[0]
            if len(entity) >= 10:  # Entidad muy específica 
                score = 0.15
            elif len(entity) >= 6:
                score = 0.08
            else:
                score = 0.03
        else:
            score = 0.0
        
        # Bonus por especificidad promedio
        if avg_specificity > 12:
            score *= 1.2  # Entidades muy específicas
        
        return min(0.35, score), list(common)
    
    def _calculate_semantic_score(
        self,
        user_vector: np.ndarray,
        article_vector: np.ndarray
    ) -> float:
        """
        Calcula similitud semántica con umbral
        """
        if user_vector.shape != article_vector.shape:
            min_dim = min(len(user_vector), len(article_vector))
            user_vector = user_vector[:min_dim]
            article_vector = article_vector[:min_dim]
        
        # Verificar que los vectores no sean cero
        if np.linalg.norm(user_vector) == 0 or np.linalg.norm(article_vector) == 0:
            return 0.0
        
        cosine_sim = cosine_similarity([user_vector], [article_vector])[0][0]
        
        # Aplicar umbral: similitudes muy bajas = 0
        if cosine_sim < self.MIN_COSINE_THRESHOLD:
            return 0.0
        
        # Re-escalar desde el umbral hasta 1
        # Esto hace que el rango útil sea [threshold, 1] -> [0, 1]
        scaled = (cosine_sim - self.MIN_COSINE_THRESHOLD) / (1 - self.MIN_COSINE_THRESHOLD)
        
        return max(0.0, min(1.0, scaled))
    
    def calculate_relevance(
        self,
        user_vector: np.ndarray,
        article_vector: np.ndarray,
        article_categories: List[str],
        user_categories: List[str],
        article_sentiment: Optional[Dict] = None,
        user_sentiment_preference: Optional[str] = None,
        article_section: Optional[str] = None,
        user_section_preference: Optional[List[str]] = None,
        article_date: Optional[str] = None,
        article_entities: Optional[List[Dict]] = None,
        user_entities: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict]:
        """
        Calcula la relevancia de un artículo para un usuario
        
        Returns:
            Tuple de (score final, detalles de scoring)
        """
        # 1. Score semántico (TF-IDF cosine similarity)
        semantic_score = self._calculate_semantic_score(user_vector, article_vector)
        
        # 2. Score de categorías con IDF
        category_score, matching_cats = self._calculate_category_score(
            user_categories, article_categories
        )
        
        # 3. Score de entidades (filtrado estricto)
        entity_score, matching_ents = self._calculate_entity_score(
            user_entities or [], article_entities or []
        )
        
        # 4. Factor de recencia
        recency_factor = 1.0
        days_old = None
        if article_date:
            try:
                if isinstance(article_date, str):
                    article_dt = datetime.fromisoformat(article_date.replace('Z', '+00:00'))
                else:
                    article_dt = article_date
                
                now = datetime.now(timezone.utc)
                days_old = (now - article_dt).days
                
                # Decay más agresivo para artículos viejos
                decay_rate = 0.003
                recency_factor = math.exp(-decay_rate * max(0, days_old))
                recency_factor = max(0.6, recency_factor)
            except Exception:
                recency_factor = 1.0
        
        # SCORE FINAL con lógica mejorada:
        # La combinación depende de qué señales tenemos
        
        # Si hay match de categorías, darle más peso a eso
        if matching_cats:
            # Categorías coinciden: combinación balanceada
            raw_score = (
                semantic_score * 0.40 +
                category_score * 0.40 +
                entity_score * 0.20
            )
        elif matching_ents:
            # Sin categorías pero con entidades: más peso a semántico y entidades
            raw_score = (
                semantic_score * 0.50 +
                category_score * 0.20 +
                entity_score * 0.30
            )
        else:
            # Solo semántico: requiere score alto
            if semantic_score < 0.4:
                raw_score = semantic_score * 0.3  # Penalizar fuertemente
            else:
                raw_score = semantic_score * 0.70
        
        # Aplicar recencia como multiplicador
        final_score = raw_score * recency_factor
        
        # Asegurar rango válido
        final_score = max(0.0, min(1.0, final_score))
        
        # Detalles para justificación
        details = {
            'semantic_score': round(semantic_score, 4),
            'category_score': round(category_score, 4),
            'entity_score': round(entity_score, 4),
            'recency_factor': round(recency_factor, 4),
            'days_old': days_old,
            'matching_categories': matching_cats,
            'matching_entities': matching_ents,
            'final_score': round(final_score, 4)
        }
        
        return final_score, details
    
    def match_articles(
        self,
        user_profile: Dict,
        articles: List[Dict],
        top_k: int = 10,
        min_score: float = 0.12,
        diversity_weight: float = 0.3
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Encuentra los artículos más relevantes para un usuario
        Usa MMR (Maximal Marginal Relevance) para diversidad
        
        Args:
            user_profile: Perfil del usuario
            articles: Lista de artículos
            top_k: Número de artículos a retornar
            min_score: Score mínimo para incluir un artículo
            diversity_weight: Peso para diversidad (0=solo relevancia, 1=solo diversidad)
            
        Returns:
            Lista de tuplas (artículo, score, justificación)
        """
        user_vector = np.array(user_profile['vector'])
        user_categories = user_profile.get('categories', [])
        user_entities = user_profile.get('entities', [])
        
        # Primera pasada: calcular scores de relevancia para todos
        candidates = []
        
        for article in articles:
            article_vector = np.array(article.get('vector', []))
            
            if len(article_vector) == 0:
                continue
            
            article_categories = article.get('categories', [])
            article_entities = article.get('entidades', [])
            article_sentiment = article.get('sentiment')
            article_section = article.get('section')
            article_date = article.get('source_metadata', {}).get('date')
            
            # Calcular factor de calidad del artículo
            quality_factor = self._calculate_article_quality(article)
            
            score, details = self.calculate_relevance(
                user_vector=user_vector,
                article_vector=article_vector,
                article_categories=article_categories,
                user_categories=user_categories,
                article_sentiment=article_sentiment,
                article_section=article_section,
                article_date=article_date,
                article_entities=article_entities,
                user_entities=user_entities
            )
            
            # Aplicar factor de calidad
            adjusted_score = score * quality_factor
            
            # Filtrar por score mínimo
            if adjusted_score < min_score:
                continue
            
            details['quality_factor'] = round(quality_factor, 4)
            details['adjusted_score'] = round(adjusted_score, 4)
            
            justification = {
                'score': adjusted_score,
                'matching_categories': details['matching_categories'],
                'matching_entities': details['matching_entities'],
                'article_categories': article_categories,
                'sentiment': article_sentiment.get('label') if article_sentiment else None,
                'details': details
            }
            
            candidates.append((article, adjusted_score, justification, article_vector))
        
        if not candidates:
            return []
        
        # Segunda pasada: aplicar MMR para diversidad
        selected = []
        remaining = candidates.copy()
        
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -1
            
            for i, (article, score, just, vec) in enumerate(remaining):
                # Relevancia
                relevance = score
                
                # Diversidad: máxima similitud con artículos ya seleccionados
                if selected:
                    similarities = []
                    for _, _, _, sel_vec in selected:
                        if len(vec) == len(sel_vec):
                            sim = cosine_similarity([vec], [sel_vec])[0][0]
                            similarities.append(sim)
                    max_similarity = max(similarities) if similarities else 0
                else:
                    max_similarity = 0
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance - diversity_weight * max_similarity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        # Retornar sin el vector (no necesario en output)
        return [(art, score, just) for art, score, just, _ in selected]

