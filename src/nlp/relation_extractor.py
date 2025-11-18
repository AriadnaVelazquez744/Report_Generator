"""
Extracción de relaciones usando spaCy y análisis sintáctico
Basado en el análisis de LCA (Lowest Common Ancestor)
"""
import logging
from typing import List, Dict, Tuple, Optional
import itertools
import re
import spacy

logger = logging.getLogger(__name__)

# Cargar modelo de spaCy
try:
    nlp = spacy.load("es_core_news_lg")
except OSError:
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        logger.error("No se encontró modelo de spaCy. Por favor instálalo con: python -m spacy download es_core_news_sm")
        nlp = None

LABEL_MAP = {"PER": "Persona", "ORG": "Organizacion", "LOC": "Lugar"}


class RelationExtractor:
    """Extractor de relaciones entre entidades usando análisis sintáctico"""
    
    def __init__(self, nlp_model=None):
        """
        Inicializa el extractor de relaciones
        
        Args:
            nlp_model: Modelo de spaCy (opcional)
        """
        self.nlp = nlp_model or nlp
        if not self.nlp:
            raise ValueError("No se pudo cargar el modelo de spaCy")
    
    def get_path_to_root(self, token) -> List:
        """Obtiene el camino desde un token hasta la raíz"""
        path = []
        current = token
        while current != current.head:
            path.append(current)
            current = current.head
        path.append(current)  # Add root
        return path
    
    def find_lowest_common_ancestor(self, token1, token2) -> Optional:
        """Encuentra el ancestro común más bajo entre dos tokens"""
        path1 = set(self.get_path_to_root(token1))
        path2 = self.get_path_to_root(token2)
        
        for token in path2:
            if token in path1:
                return token
        return None
    
    def extract_relation_phrase(self, token1, token2, lca) -> str:
        """Extrae la frase que representa la relación entre dos entidades"""
        path1 = self.get_path_to_root(token1)
        path2 = self.get_path_to_root(token2)
        
        idx1 = path1.index(lca)
        idx2 = path2.index(lca)
        
        full_path = path1[:idx1] + [lca] + path2[idx2+1:]
        
        relation_tokens = []
        for token in full_path:
            if token.pos_ in ("VERB", "AUX", "ADP", "ADV", "SCONJ"):
                negation = [child for child in token.children if child.dep_ == "neg"]
                if negation:
                    relation_tokens.append("NO")
                relation_tokens.append(token.lemma_.upper())
        
        return "_".join(relation_tokens) if relation_tokens else "RELACIONADO_CON"
    
    def identify_subject_object(self, ent1, ent2, lca) -> Tuple:
        """Identifica cuál entidad es sujeto y cuál es objeto"""
        root1, root2 = ent1.root, ent2.root
        
        if root1.dep_ in ("nsubj", "nsubj:pass"):
            return ent1, ent2
        elif root2.dep_ in ("nsubj", "nsubj:pass"):
            return ent2, ent1
        
        lca_children = list(lca.children)
        if root1 in lca_children and root1.dep_ in ("nsubj", "nsubj:pass"):
            return ent1, ent2
        elif root2 in lca_children and root2.dep_ in ("nsubj", "nsubj:pass"):
            return ent2, ent1
        
        if root1.i < root2.i:
            return ent1, ent2
        else:
            return ent2, ent1
    
    def handle_apposition(self, ent1, ent2) -> Optional[Dict]:
        """Maneja relaciones de aposición"""
        root1, root2 = ent1.root, ent2.root
        
        if (root1.head == root2 and root1.dep_ == "appos") or \
           (root2.head == root1 and root2.dep_ == "appos"):
            return {
                "sujeto": ent1.text.strip(),
                "relacion": "ES",
                "objeto": ent2.text.strip()
            }
        return None
    
    def handle_prepositional_relation(self, ent1, ent2) -> Optional[Dict]:
        """Maneja relaciones preposicionales directas"""
        root1, root2 = ent1.root, ent2.root
        
        for token in [root1, root2]:
            if token.dep_ == "prep":
                prep_objects = [child for child in token.children if child.dep_ == "pobj"]
                if prep_objects:
                    obj_ent = prep_objects[0]
                    if obj_ent in [root1, root2]:
                        subject = root2 if obj_ent == root1 else root1
                        return {
                            "sujeto": subject.text.strip(),
                            "relacion": f"EN_{token.lemma_.upper()}",
                            "objeto": obj_ent.text.strip()
                        }
        return None
    
    def extract_relations_from_sentence(self, sent) -> List[Dict]:
        """Extrae relaciones de una oración individual"""
        relations = []
        entities = [ent for ent in sent.ents if ent.label_ in LABEL_MAP]
        
        if len(entities) < 2:
            return relations
        
        for ent1, ent2 in itertools.combinations(entities, 2):
            if ent1.text == ent2.text:
                continue
            
            apposition_rel = self.handle_apposition(ent1, ent2)
            if apposition_rel:
                relations.append(apposition_rel)
                continue
            
            prep_rel = self.handle_prepositional_relation(ent1, ent2)
            if prep_rel:
                relations.append(prep_rel)
                continue
            
            lca = self.find_lowest_common_ancestor(ent1.root, ent2.root)
            if not lca:
                continue
            
            if lca.pos_ not in ("VERB", "AUX", "ADP"):
                continue
            
            try:
                sujeto, objeto = self.identify_subject_object(ent1, ent2, lca)
                relation_phrase = self.extract_relation_phrase(ent1.root, ent2.root, lca)
                
                relations.append({
                    "sujeto": sujeto.text.strip(),
                    "relacion": self.sanitize_relation_type(relation_phrase),
                    "objeto": objeto.text.strip()
                })
            except Exception as e:
                logger.debug(f"Error procesando relación entre {ent1.text} y {ent2.text}: {e}")
                continue
        
        return relations
    
    def sanitize_relation_type(self, rel_type: str) -> str:
        """Sanitiza el tipo de relación para que sea válido"""
        if not rel_type:
            return "RELACIONADO_CON"
        
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type)
        if sanitized and sanitized[0].isdigit():
            sanitized = "REL_" + sanitized
        
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        
        if not sanitized:
            return "RELACIONADO_CON"
        
        return sanitized.upper()
    
    def extract(self, text: str) -> Dict:
        """
        Extrae entidades y relaciones de un texto
        
        Args:
            text: Texto a procesar
            
        Returns:
            Diccionario con entidades y relaciones
        """
        if not text:
            return {
                'entidades': [],
                'relaciones': []
            }
        
        doc = self.nlp(text)
        
        entidades = []
        relaciones = []
        
        for ent in doc.ents:
            if ent.label_ in LABEL_MAP:
                entidades.append({
                    "nombre": ent.text.strip(),
                    "tipo": LABEL_MAP[ent.label_]
                })
        
        for sent in doc.sents:
            sent_relations = self.extract_relations_from_sentence(sent)
            relaciones.extend(sent_relations)
        
        # Eliminar duplicados
        entidades_unicas = [dict(t) for t in {tuple(d.items()) for d in entidades}]
        relaciones_unicas = []
        seen_relations = set()
        
        for rel in relaciones:
            rel_key = (rel["sujeto"], rel["relacion"], rel["objeto"])
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                relaciones_unicas.append(rel)
        
        return {
            "entidades": entidades_unicas,
            "relaciones": relaciones_unicas
        }



