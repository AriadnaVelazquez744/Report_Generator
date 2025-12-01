"""
Generador de reportes personalizados
"""
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from ..summarization.summarizer import PersonalizedSummarizer

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _remove_noise_from_text(text: str) -> str:
    """
    Elimina patrones de ruido como "LEA TAMBIÉN" del texto.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto sin ruido
    """
    if not isinstance(text, str):
        return ""
    
    # Patrones robustos para capturar todas las variantes de "LEA TAMBIÉN"
    noise_patterns = [
        # Variante: LEA TAMBIÉN: seguido de texto en la misma línea
        r'LEA\s+TAMBI[EÉ]N\s*:\s*[^\n]+',
        # Variante: LEA TAMBIÉN seguido de salto de línea y texto hasta el siguiente párrafo
        r'LEA\s+TAMBI[EÉ]N\s*\n+[^\n]+(?:\n(?![A-Z]))*',
        # Variante: LEA TAMBIÉN con texto hasta punto final
        r'LEA\s+TAMBI[EÉ]N\s*:?\s*[^.!?\n]*[.!?]?',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Limpiar saltos de línea múltiples resultantes
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalizar espacios múltiples
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


class ReportGenerator:
    """Generador de reportes personalizados de noticias"""
    
    def __init__(self, summarizer: PersonalizedSummarizer):
        """
        Inicializa el generador de reportes
        
        Args:
            summarizer: Resumidor personalizado
        """
        self.summarizer = summarizer
    
    def generate_report(
        self,
        matched_articles: List[tuple],
        user_profile: Dict,
        max_articles: int = 10
    ) -> Dict:
        """
        Genera un reporte personalizado
        
        Args:
            matched_articles: Lista de tuplas (artículo, score, justificación)
            user_profile: Perfil del usuario
            max_articles: Número máximo de artículos en el reporte
            
        Returns:
            Diccionario con el reporte completo
        """
        user_categories = user_profile.get('categories', [])
        
        report_items = []
        
        for article, score, justification in matched_articles[:max_articles]:
            # Generar resumen personalizado - primero limpiar el texto de ruido
            article_text = article.get('text', '')
            article_text = _remove_noise_from_text(article_text)
            summary = self.summarizer.summarize_for_profile(
                article_text,
                user_categories,
                num_sentences=3
            )
            
            report_item = {
                'article_id': article.get('id'),
                'title': article.get('title'),
                'url': article.get('url'),
                'section': article.get('section'),
                'summary': summary,
                'score': score,
                'justification': {
                    'matching_categories': justification.get('matching_categories', []),
                    'matching_entities': justification.get('matching_entities', []),
                    'article_categories': justification.get('article_categories', []),
                    'sentiment': justification.get('sentiment'),
                    'score_breakdown': justification.get('score', 0)
                },
                'date': article.get('source_metadata', {}).get('date'),
                'tags': article.get('tags', []),
                'entities': article.get('entidades', [])
            }
            
            report_items.append(report_item)
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'user_profile': {
                'categories': user_categories,
                'entities': user_profile.get('entities', []),
                'profile_text': user_profile.get('profile_text', '')
            },
            'total_articles': len(matched_articles),
            'articles_in_report': len(report_items),
            'articles': report_items
        }
        
        return report
    
    def format_report_text(self, report: Dict) -> str:
        """
        Formatea el reporte como texto legible
        
        Args:
            report: Diccionario del reporte
            
        Returns:
            Texto formateado del reporte
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REPORTE PERSONALIZADO DE NOTICIAS")
        lines.append("=" * 80)
        lines.append(f"\nGenerado: {report['generated_at']}")
        lines.append(f"Total de artículos relevantes: {report['total_articles']}")
        lines.append(f"Artículos en este reporte: {report['articles_in_report']}")
        lines.append("\n" + "-" * 80)
        
        for i, article in enumerate(report['articles'], 1):
            lines.append(f"\n{i}. {article['title']}")
            lines.append(f"   Sección: {article['section']}")
            lines.append(f"   Score de relevancia: {article['score']:.3f}")
            lines.append(f"\n   Resumen:")
            lines.append(f"   {article['summary']}")
            
            if article['justification']['matching_categories']:
                lines.append(f"\n   Categorías coincidentes: {', '.join(article['justification']['matching_categories'])}")
            
            if article['justification']['sentiment']:
                lines.append(f"   Sentimiento: {article['justification']['sentiment']}")
            
            lines.append(f"\n   URL: {article['url']}")
            lines.append("-" * 80)
        
        return "\n".join(lines)
    
    def generate_pdf(self, report: Dict, output_path: str, user_name: Optional[str] = None) -> bool:
        """
        Genera un reporte en formato PDF
        
        Args:
            report: Diccionario del reporte
            output_path: Ruta donde guardar el PDF
            user_name: Nombre del usuario (opcional)
            
        Returns:
            True si se generó exitosamente, False en caso contrario
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("reportlab no está instalado. Instálalo con: pip install reportlab")
            return False
        
        try:
            # Crear directorio si no existe
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Crear documento
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )
            
            # Contenedor de elementos
            story = []
            
            # Estilos
            styles = getSampleStyleSheet()
            
            # Estilo personalizado para título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            # Estilo para subtítulos
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#555555'),
                spaceAfter=12,
                alignment=TA_CENTER,
            )
            
            # Estilo para títulos de artículos
            article_title_style = ParagraphStyle(
                'ArticleTitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=6,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            )
            
            # Estilo para metadata
            meta_style = ParagraphStyle(
                'Meta',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#7f8c8d'),
                spaceAfter=6,
            )
            
            # Estilo para resumen
            summary_style = ParagraphStyle(
                'Summary',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leading=14,
            )
            
            # Estilo para perfil de usuario
            profile_style = ParagraphStyle(
                'Profile',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#34495e'),
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                leading=13,
                leftIndent=20,
                rightIndent=20,
            )
            
            # Estilo para categorías
            category_style = ParagraphStyle(
                'Category',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#16a085'),
                spaceAfter=6,
            )
            
            # Encabezado
            story.append(Paragraph("REPORTE PERSONALIZADO DE NOTICIAS", title_style))
            
            # Información del usuario
            if user_name:
                story.append(Paragraph(f"Usuario: {user_name}", subtitle_style))
            
            # Fecha de generación
            generated_date = datetime.fromisoformat(report['generated_at']).strftime('%d/%m/%Y %H:%M')
            story.append(Paragraph(f"Generado: {generated_date}", meta_style))
            story.append(Spacer(1, 12))
            
            # Perfil del usuario - Gustos e Intereses
            user_profile = report.get('user_profile', {})
            if user_profile:
                story.append(Paragraph("<b>📋 Perfil de Intereses del Usuario</b>", article_title_style))
                
                # Texto del perfil
                profile_text = user_profile.get('profile_text', '')
                if profile_text:
                    story.append(Paragraph(f"<i>{profile_text}</i>", profile_style))
                    story.append(Spacer(1, 12))
                
                # Categorías de interés
                categories = user_profile.get('categories', [])
                if categories:
                    story.append(Paragraph("<b>🏷️  Categorías de Interés:</b>", category_style))
                    # Mostrar las primeras 15 categorías más relevantes
                    categories_display = categories[:15]
                    categories_text = ", ".join(categories_display)
                    if len(categories) > 15:
                        categories_text += f" <i>(+{len(categories) - 15} más)</i>"
                    story.append(Paragraph(categories_text, category_style))
                    story.append(Spacer(1, 8))
                
                # Entidades de interés
                entities = user_profile.get('entities', [])
                if entities:
                    story.append(Paragraph("<b>👤 Entidades Mencionadas en el Perfil:</b>", category_style))
                    entity_texts = [f"{e['text']} ({e['label']})" for e in entities[:10]]
                    entities_display = ", ".join(entity_texts)
                    if len(entities) > 10:
                        entities_display += f" <i>(+{len(entities) - 10} más)</i>"
                    story.append(Paragraph(entities_display, category_style))
                    story.append(Spacer(1, 12))
            
            
            story.append(Spacer(1, 20))
            
            # Título de la sección de artículos
            story.append(Paragraph("<b>📰 Artículos Recomendados</b>", article_title_style))
            story.append(Spacer(1, 10))
            
            # Artículos
            for i, article in enumerate(report['articles'], 1):
                # Título del artículo
                title_text = f"{i}. {article['title'].replace("-"," ").replace("teleSUR",'')}"
                story.append(Paragraph(title_text, article_title_style))
                
                # Metadata
                meta_info = []
                meta_info.append(f"<b>Sección:</b> {article['section']}")
                meta_info.append(f"<b>Score de relevancia:</b> {article['score']:.3f}")
                
                if article.get('date'):
                    try:
                        article_date = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
                        meta_info.append(f"<b>Fecha:</b> {article_date.strftime('%d/%m/%Y')}")
                    except:
                        pass
                
                story.append(Paragraph(" | ".join(meta_info), meta_style))
                story.append(Spacer(1, 8))
                
                # Resumen
                story.append(Paragraph("<b>Resumen:</b>", summary_style))
                story.append(Paragraph(article['summary'], summary_style))
                
                # Entidades mencionadas en el artículo
                entities = article.get('entities', [])
                if entities:
                    entity_texts = [f"{e['text']}" for e in entities[:8]]
                    entities_str = ", ".join(entity_texts)
                    if len(entities) > 8:
                        entities_str += f" <i>(+{len(entities) - 8} más)</i>"
                    story.append(Paragraph(f"<b>🏷️  Entidades mencionadas:</b> {entities_str}", meta_style))
                
                # Categorías coincidentes
                if article['justification']['matching_categories']:
                    categories_text = ", ".join(article['justification']['matching_categories'])
                    story.append(Paragraph(f"<b>✓ Categorías coincidentes:</b> {categories_text}", category_style))
                
                # Entidades coincidentes (relevante para el usuario)
                if article['justification'].get('matching_entities'):
                    matching_entities_text = ", ".join(article['justification']['matching_entities'])
                    story.append(Paragraph(f"<b>⭐ Entidades de tu interés:</b> {matching_entities_text}", category_style))
                
                # URL
                url_text = f"<b>URL:</b> <link href='{article['url']}'>{article['url']}</link>"
                story.append(Paragraph(url_text, meta_style))
                
                # Separador entre artículos
                if i < len(report['articles']):
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("─" * 60, meta_style))
                    story.append(Spacer(1, 10))
            
            # Generar PDF
            doc.build(story)
            logger.info(f"PDF generado exitosamente en: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generando PDF: {e}")
            return False



