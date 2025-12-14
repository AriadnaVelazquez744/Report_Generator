"""
Intermediario entre news_recomendation.py y report_generator.py
Transforma datos y genera reportes en texto plano
"""
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from src.recommendation.report_generator import ReportGenerator
from src.summarization.summarizer import PersonalizedSummarizer
from src.process.news_recomendation import generate_report_recommendations

logger = logging.getLogger(__name__)


def generate_plain_text_report(
    recommendations_result: Dict[str, Any],
    summarizer: Optional[PersonalizedSummarizer] = None,
    max_articles: int = 10,
    user_query: Optional[str] = None,
    user_name: Optional[str] = None
) -> str:
    """
    Genera un reporte en texto plano a partir del resultado de generate_report_recommendations.
    
    Args:
        recommendations_result: Resultado de generate_report_recommendations() con:
            - matches: List[Tuple(article, score, details)]
            - user_profile: Dict
            - search_stats: Dict
        summarizer: Instancia de PersonalizedSummarizer (opcional, no se usa por ahora)
        max_articles: Número máximo de artículos en el reporte
        user_query: Query/pregunta del usuario (opcional, se muestra en lugar del perfil)
        user_name: Nombre del usuario (opcional)
        
    Returns:
        Texto plano formateado del reporte
    """
    if 'error' in recommendations_result:
        return f"Error: {recommendations_result['error']}"
    
    # Extraer datos
    matches = recommendations_result.get('matches', [])
    user_profile = recommendations_result.get('user_profile', {})
    search_stats = recommendations_result.get('search_stats', {})
    
    # Transformar matches al formato esperado por ReportGenerator
    transformed_matches = _transform_matches_for_report(matches)
    
    # Generar reporte (sin summarizer por ahora)
    report_generator = ReportGenerator(summarizer=None)
    report = report_generator.generate_report(
        matched_articles=transformed_matches,
        user_profile=user_profile,
        max_articles=max_articles
    )
    
    # Agregar estadísticas de búsqueda al reporte
    report['search_stats'] = search_stats
    
    # Formatear como texto plano personalizado
    return _format_report_text_custom(
        report,
        user_query=user_query,
        user_name=user_name,
        search_stats=search_stats
    )


def _format_report_text_custom(
    report: Dict[str, Any],
    user_query: Optional[str] = None,
    user_name: Optional[str] = None,
    search_stats: Optional[Dict] = None
) -> str:
    """
    Formatea el reporte como texto plano similar al PDF generado.
    
    Args:
        report: Diccionario del reporte
        user_query: Query/pregunta del usuario (se muestra en lugar del perfil)
        user_name: Nombre del usuario
        search_stats: Estadísticas de búsqueda
        
    Returns:
        Texto formateado del reporte
    """
    from datetime import datetime
    
    lines = []
    
    # Encabezado principal
    lines.append("=" * 80)
    lines.append("REPORTE PERSONALIZADO DE NOTICIAS")
    lines.append("=" * 80)
    
    # Información del usuario
    if user_name:
        lines.append(f"\nUsuario: {user_name}")
    
    # Fecha de generación
    try:
        generated_date = datetime.fromisoformat(report['generated_at']).strftime('%d/%m/%Y %H:%M')
        lines.append(f"Generado: {generated_date}")
    except:
        lines.append(f"Generado: {report['generated_at']}")
    
    lines.append("")
    
    # Mostrar la query en lugar del perfil
    if user_query:
        lines.append("📝 BÚSQUEDA REALIZADA")
        lines.append("-" * 80)
        lines.append(f"{user_query}")
        lines.append("")
    
    # Mostrar categorías de interés si existen
    user_profile = report.get('user_profile', {})
    if user_profile and user_profile.get('categories'):
        lines.append("🏷️  CATEGORÍAS DE INTERÉS")
        lines.append("-" * 80)
        categories = user_profile.get('categories', [])[:15]
        lines.append(", ".join(categories))
        if len(user_profile.get('categories', [])) > 15:
            lines.append(f" (+{len(user_profile.get('categories', [])) - 15} más)")
        lines.append("")
    
    # Mostrar entidades de interés si existen
    if user_profile and user_profile.get('entities'):
        lines.append("👤 ENTIDADES MENCIONADAS EN EL PERFIL")
        lines.append("-" * 80)
        entities = user_profile.get('entities', [])[:10]
        entity_texts = [f"{e['text']} ({e['label']})" for e in entities]
        lines.append(", ".join(entity_texts))
        if len(user_profile.get('entities', [])) > 10:
            lines.append(f" (+{len(user_profile.get('entities', [])) - 10} más)")
        lines.append("")
    
    # Sección de artículos recomendados
    lines.append("📰 ARTÍCULOS RECOMENDADOS")
    lines.append("=" * 80)
    lines.append("")
    
    # Artículos
    articles = report.get('articles', [])
    for i, article in enumerate(articles, 1):
        # Título del artículo
        lines.append(f"{i}. {article['title']}")
        
        # Metadata
        lines.append(f"   Sección: {article['section']}")
        
        # Fecha si existe
        if article.get('date'):
            try:
                article_date = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
                lines.append(f"   Fecha: {article_date.strftime('%d/%m/%Y')}")
            except:
                pass
        
        lines.append("")
        
        # Resumen/Texto
        lines.append("   Resumen:")
        summary = article.get('summary', '')
        if summary:
            # Limitar a primeras 500 caracteres para texto plano
            if len(summary) > 500:
                summary = summary[:500] + "..."
            # Indentar el resumen
            for line in summary.split('\n'):
                lines.append(f"   {line}")
        
        lines.append("")
        
        # Entidades mencionadas en el artículo
        entities = article.get('entities', [])
        if entities:
            entity_texts = [e['text'] for e in entities[:8]]
            entities_str = ", ".join(entity_texts)
            if len(entities) > 8:
                entities_str += f" (+{len(entities) - 8} más)"
            lines.append(f"   🏷️  Entidades mencionadas: {entities_str}")
        
        # Categorías coincidentes
        justification = article.get('justification', {})
        if justification.get('matching_categories'):
            categories_text = ", ".join(justification['matching_categories'])
            lines.append(f"   ✓ Categorías coincidentes: {categories_text}")
        
        # Entidades coincidentes (relevante para el usuario)
        if justification.get('matching_entities'):
            matching_entities_text = ", ".join(justification['matching_entities'])
            lines.append(f"   ⭐ Entidades de tu interés: {matching_entities_text}")
        
        # URL
        lines.append(f"   URL: {article['url']}")
        
        # Separador entre artículos
        if i < len(articles):
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
    
    # Pie de página
    lines.append("=" * 80)
    lines.append(f"Total de artículos en este reporte: {len(articles)}")
    lines.append(f"Total de artículos relevantes encontrados: {report.get('total_articles', 0)}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def _transform_matches_for_report(
    matches: List[Tuple[Dict, float, Dict]]
) -> List[Tuple[Dict, float, Dict]]:
    """
    Transforma los matches del formato de news_recomendation al formato esperado por ReportGenerator.
    
    Convierte 'details' (del matcher) en 'justification' (para el reporte).
    
    Args:
        matches: Lista de tuplas (article, score, details) del matcher
        
    Returns:
        Lista de tuplas (article, score, justification) para ReportGenerator
    """
    transformed = []
    
    for article, score, details in matches:
        # Transformar details a justification
        justification = {
            'matching_categories': details.get('matching_categories', []),
            'matching_entities': details.get('matching_entities', []),
            'article_categories': article.get('categories', []),
            'sentiment': article.get('sentiment', {}).get('label') if isinstance(article.get('sentiment'), dict) else None,
            'score': score,
            'semantic_score': details.get('semantic_score', 0),
            'category_score': details.get('category_score', 0),
            'time_score': details.get('time_score', 0),
            'entity_score': details.get('entity_score', 0),
            'final_score': details.get('final_score', score)
        }
        
        transformed.append((article, score, justification))
    
    return transformed


def generate_report_from_user_query(
    user_id: str,
    user_query: str,
    text_processor,
    annotator,
    profile_vectorizer,
    matcher,
    summarizer: Optional[PersonalizedSummarizer] = None,
    nlp=None,
    max_articles: int = 10,
    users_file_path: str = "Data/Data_users/users.json"
) -> str:
    """
    Función de conveniencia que orquesta todo el proceso:
    1. Genera recomendaciones (news_recomendation.generate_report_recommendations)
    2. Transforma datos
    3. Genera reporte en texto plano
    
    Args:
        user_id: ID del usuario
        user_query: Query/pregunta del usuario
        text_processor: TextPreprocessor
        annotator: RegexAnnotator
        profile_vectorizer: UserProfileVectorizer
        matcher: NewsMatcher
        summarizer: PersonalizedSummarizer (opcional)
        nlp: Modelo spaCy (opcional)
        max_articles: Número máximo de artículos
        users_file_path: Ruta al archivo de usuarios
        
    Returns:
        Texto plano del reporte formateado
    """
    from src.process.news_recomendation import find_user_profile_by_id
    
    # Obtener nombre del usuario
    user_profile = find_user_profile_by_id(user_id, users_file_path)
    user_name = user_profile.get('name', 'Usuario') if user_profile else 'Usuario'
    
    # Generar recomendaciones
    recommendations = generate_report_recommendations(
        user_input=user_query,
        user_id=user_id,
        text_processor=text_processor,
        annotator=annotator,
        profile_vectorizer=profile_vectorizer,
        matcher=matcher,
        nlp=nlp,
        top_k=max_articles,
        users_file_path=users_file_path
    )
    
    # Generar reporte en texto plano
    return generate_plain_text_report(
        recommendations,
        summarizer=summarizer,
        max_articles=max_articles,
        user_query=user_query,
        user_name=user_name
    )