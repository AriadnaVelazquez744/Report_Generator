"""
Cargador de artículos JSON desde Data/Data_articles/
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class ArticleLoader:
    """Carga y valida artículos desde archivos JSON"""
    
    def __init__(self, data_dir: str = "Data/Data_articles"):
        """
        Inicializa el cargador de artículos
        
        Args:
            data_dir: Directorio base donde están los artículos
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"El directorio {data_dir} no existe")
    
    def load_article(self, file_path: Path) -> Optional[Dict | List[Dict]]:
        """
        Carga un artículo desde un archivo JSON
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            Diccionario con el artículo o None si hay error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)

            # Si es una lista (como all_articles_with_metadata.json), normalizar cada elemento
            if isinstance(article, list):
                normalized_list: List[Dict] = []
                for idx, item in enumerate(article):
                    if not isinstance(item, dict):
                        logger.warning(f"Elemento {idx} en {file_path} no es un diccionario; se omite")
                        continue
                    normalized = self._normalize_article(item)
                    normalized['_file_path'] = str(file_path)
                    normalized['_file_name'] = file_path.name
                    normalized['_list_index'] = idx
                    normalized_list.append(normalized)
                return normalized_list if normalized_list else None

            # Validar estructura básica
            if not isinstance(article, dict):
                logger.warning(f"El archivo {file_path} no contiene un diccionario")
                return None

            # Normalizar campos
            normalized = self._normalize_article(article)
            normalized['_file_path'] = str(file_path)
            normalized['_file_name'] = file_path.name

            return normalized
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando JSON en {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
            return None
    
    def _normalize_article(self, article: Dict) -> Dict:
        """
        Normaliza los campos de un artículo
        
        Args:
            article: Diccionario con el artículo
            
        Returns:
            Diccionario normalizado
        """
        normalized = {
            'url': article.get('url', ''),
            'title': article.get('title', ''),
            'section': article.get('section', ''),
            'tags': article.get('tags', []),
            'text': article.get('text', ''),
            'source_metadata': article.get('source_metadata', {}),
            'preprocessing': article.get('preprocessing', {}),
            'regex_annotations': article.get('regex_annotations', {})
        }
        
        # Asegurar que tags es una lista
        if not isinstance(normalized['tags'], list):
            normalized['tags'] = []
        
        # Asegurar que text es string
        if not isinstance(normalized['text'], str):
            normalized['text'] = str(normalized['text']) if normalized['text'] else ''
        
        # Asegurar que source_metadata es dict
        if not isinstance(normalized['source_metadata'], dict):
            normalized['source_metadata'] = {}
        
        return normalized
    
    def load_all_articles(self, recursive: bool = True) -> List[Dict]:
        """
        Carga todos los artículos desde el directorio
        
        Args:
            recursive: Si True, busca recursivamente en subdirectorios
            
        Returns:
            Lista de artículos cargados
        """
        articles = []
        
        if recursive:
            json_files = list(self.data_dir.rglob("*.json"))
        else:
            json_files = list(self.data_dir.glob("*.json"))
        
        logger.info(f"Encontrados {len(json_files)} archivos JSON")
        
        for json_file in json_files:
            article = self.load_article(json_file)
            if not article:
                continue

            # Si load_article devolvió una lista, extender; si es dict, agregar
            if isinstance(article, list):
                articles.extend(article)
            else:
                articles.append(article)
        
        logger.info(f"Cargados {len(articles)} artículos exitosamente")
        return articles
    
    def load_articles_iterator(self, recursive: bool = True) -> Iterator[Dict]:
        """
        Carga artículos de forma iterativa (para grandes volúmenes)
        
        Args:
            recursive: Si True, busca recursivamente en subdirectorios
            
        Yields:
            Artículos uno por uno
        """
        if recursive:
            json_files = self.data_dir.rglob("*.json")
        else:
            json_files = self.data_dir.glob("*.json")
        
        for json_file in json_files:
            article = self.load_article(json_file)
            if article:
                yield article
    
    def get_article_count(self) -> int:
        """
        Obtiene el número total de archivos JSON disponibles
        
        Returns:
            Número de archivos JSON
        """
        return self.count_articles(recursive=True)
    
    def count_articles(self, recursive: bool = True) -> int:
        """
        Cuenta el número de archivos JSON disponibles sin cargarlos
        
        Args:
            recursive: Si True, busca recursivamente en subdirectorios
            
        Returns:
            Número de archivos JSON
        """
        if recursive:
            json_files = list(self.data_dir.rglob("*.json"))
        else:
            json_files = list(self.data_dir.glob("*.json"))
        return len(json_files)


def load_articles_from_dir(data_dir: str = "Data/Data_articles", recursive: bool = True) -> List[Dict]:
    """
    Función de conveniencia para cargar artículos
    
    Args:
        data_dir: Directorio base donde están los artículos
        recursive: Si True, busca recursivamente en subdirectorios
        
    Returns:
        Lista de artículos cargados
    """
    loader = ArticleLoader(data_dir)
    return loader.load_all_articles(recursive=recursive)



