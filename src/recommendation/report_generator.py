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
    
    # Patrones de ruido agresivos: quitar desde el disparador hasta el siguiente punto final (.)
    noise_triggers = r'(LEA\s+TAMBI[EÉ]N|LE\s+PUEDE\s+INTERESAR|MIRA\s+TAMBI[EÉ]N|M[AÁ]S\s+EN\s+ESTA\s+SECCI[OÓ]N|VEA\s+ADEM[AÁ]S|TE\s+PUEDE\s+INTERESAR|TAMBI[EÉ]N\s+PUEDES\s+VER|SIGUE\s+LEYENDO)'
    text = re.sub(rf'(?i){noise_triggers}.*?(\.|$)', ' ', text, flags=re.DOTALL)
    
    # Limpiar saltos de línea múltiples resultantes
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalizar espacios múltiples
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


class ReportGenerator:
    """Generador de reportes personalizados de noticias"""
    
    def __init__(self):
        """
        Inicializa el generador de reportes
        """
        pass
    
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
            # Usar el texto original del artículo (sin resumen por ahora)
            article_text = article.get('text', '')
            article_text = _remove_noise_from_text(article_text)


            # Eliminar '(+ Imágenes)' y sufijos de medios como ' - teleSUR' del título
            import re
            raw_title = article.get('title', '')
            # Quitar '(+ Imágenes)' y variantes
            clean_title = re.sub(r"\s*\(\+ Imágenes\)", "", raw_title)
            # Quitar sufijos de medios como ' - teleSUR', ' – teleSUR', ' — teleSUR' (con espacios y guiones)
            clean_title = re.sub(r"\s*[-–—]\s*teleSUR\s*$", "", clean_title, flags=re.IGNORECASE)
            clean_title = clean_title.strip()

            report_item = {
                'article_id': article.get('id'),
                'title': clean_title,
                'url': article.get('url'),
                'section': article.get('section'),
                # Usar el summary generado por el modelo si existe, sino usar texto original
                'summary': article.get('summary') or article_text,
                'generated_by_model': article.get('generated_by_model', False),
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
    
       
    def generate_pdf(
        self,
        report: Dict,
        output_path: str,
        user_name: Optional[str] = None,
        user_query: Optional[str] = None
    ) -> bool:
        """
        Genera un reporte en formato PDF con estructura mejorada basada en _format_report_text_custom
        
        Args:
            report: Diccionario del reporte
            output_path: Ruta donde guardar el PDF
            user_name: Nombre del usuario (opcional)
            user_query: Query del usuario (opcional)
            
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
            
            # Estilo para títulos de sección
            section_title_style = ParagraphStyle(
                'SectionTitle',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold'
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
            
            # Estilo para separadores
            separator_style = ParagraphStyle(
                'Separator',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#bdc3c7'),
                spaceAfter=10,
                spaceBefore=10,
                alignment=TA_CENTER,
            )
            
            # Encabezado principal - Línea superior
            story.append(Paragraph("=" * 80, separator_style))
            
            # Título principal
            story.append(Paragraph("REPORTE PERSONALIZADO DE NOTICIAS", title_style))
            
            # Línea inferior del encabezado
            story.append(Paragraph("=" * 80, separator_style))
            story.append(Spacer(1, 12))
            
            # Información del usuario
            if user_name:
                story.append(Paragraph(f"<b>Usuario:</b> {user_name}", subtitle_style))
            
            # Fecha de generación
            try:
                generated_date = datetime.fromisoformat(report['generated_at']).strftime('%d/%m/%Y %H:%M')
                story.append(Paragraph(f"<b>Generado:</b> {generated_date}", meta_style))
            except:
                story.append(Paragraph(f"<b>Generado:</b> {report['generated_at']}", meta_style))
            
            story.append(Spacer(1, 20))
            
            # Mostrar la query en lugar del perfil
            if user_query:
                story.append(Paragraph("📝 BÚSQUEDA REALIZADA", section_title_style))
                story.append(Paragraph("-" * 80, separator_style))
                story.append(Paragraph(user_query, profile_style))
                story.append(Spacer(1, 20))
            
            # Mostrar categorías de interés si existen
            user_profile = report.get('user_profile', {})
            if user_profile and user_profile.get('categories'):
                story.append(Paragraph("🏷️ CATEGORÍAS DE INTERÉS", section_title_style))
                story.append(Paragraph("-" * 80, separator_style))
                categories = user_profile.get('categories', [])[:15]
                categories_text = ", ".join(categories)
                if len(user_profile.get('categories', [])) > 15:
                    categories_text += f" <i>(+{len(user_profile.get('categories', [])) - 15} más)</i>"
                story.append(Paragraph(categories_text, category_style))
                story.append(Spacer(1, 20))
            
            # Mostrar entidades de interés si existen
            if user_profile and user_profile.get('entities'):
                story.append(Paragraph("👤 ENTIDADES MENCIONADAS EN EL PERFIL", section_title_style))
                story.append(Paragraph("-" * 80, separator_style))
                entities = user_profile.get('entities', [])[:10]
                entity_texts = [f"{e['text']} ({e['label']})" for e in entities]
                entities_display = ", ".join(entity_texts)
                if len(user_profile.get('entities', [])) > 10:
                    entities_display += f" <i>(+{len(user_profile.get('entities', [])) - 10} más)</i>"
                story.append(Paragraph(entities_display, category_style))
                story.append(Spacer(1, 20))
            
            # Sección de artículos recomendados
            story.append(Paragraph("📰 ARTÍCULOS RECOMENDADOS", section_title_style))
            story.append(Paragraph("=" * 80, separator_style))
            story.append(Spacer(1, 15))
            
            # Artículos
            articles = report.get('articles', [])
            for i, article in enumerate(articles, 1):
                # Título del artículo
                title_text = f"{i}. {article.get('title', 'Sin título')}"
                story.append(Paragraph(title_text, article_title_style))
                
                # Metadata
                meta_info = []
                meta_info.append(f"<b>Sección:</b> {article.get('section', 'Sin sección')}")
                
                # Fecha si existe
                if article.get('date'):
                    try:
                        article_date = datetime.fromisoformat(article.get('date'))
                        meta_info.append(f"<b>Fecha:</b> {article_date.strftime('%d/%m/%Y')}")
                    except:
                        meta_info.append(f"<b>Fecha:</b> {article.get('date')}")
                
                story.append(Paragraph(" | ".join(meta_info), meta_style))
                story.append(Spacer(1, 8))
                
                # Resumen/Texto
                story.append(Paragraph("<b>Resumen:</b>", summary_style))
                summary = article.get('summary', '')
                if summary:
                    # Limitar longitud para mejor visualización en PDF
                    if len(summary) > 800:
                        summary = summary[:800] + "..."
                    story.append(Paragraph(summary, summary_style))
                
                story.append(Spacer(1, 8))
                
                # Entidades mencionadas en el artículo
                entities = article.get('entities', [])
                if entities:
                    entity_texts = [e['text'] for e in entities[:8]]
                    entities_str = ", ".join(entity_texts)
                    if len(entities) > 8:
                        entities_str += f" <i>(+{len(entities) - 8} más)</i>"
                    story.append(Paragraph(f"<b>🏷️ Entidades mencionadas:</b> {entities_str}", meta_style))
                
                # Categorías coincidentes
                justification = article.get('justification', {})
                if justification.get('matching_categories'):
                    categories_text = ", ".join(justification['matching_categories'])
                    story.append(Paragraph(f"<b>✓ Categorías coincidentes:</b> {categories_text}", category_style))
                
                # Entidades coincidentes (relevante para el usuario)
                if justification.get('matching_entities'):
                    matching_entities_text = ", ".join(justification['matching_entities'])
                    story.append(Paragraph(f"<b>⭐ Entidades de tu interés:</b> {matching_entities_text}", category_style))
                
                # URL
                url_text = f"<b>URL:</b> <link href='{article.get('url', '')}'>{article.get('url', '')}</link>"
                story.append(Paragraph(url_text, meta_style))
                
                # Separador entre artículos
                if i < len(articles):
                    story.append(Spacer(1, 15))
                    story.append(Paragraph("-" * 80, separator_style))
                    story.append(Spacer(1, 10))
            
            # Pie de página
            story.append(Spacer(1, 20))
            story.append(Paragraph("=" * 80, separator_style))
            story.append(Paragraph(f"<b>Total de artículos en este reporte:</b> {len(articles)}", meta_style))
            story.append(Paragraph(f"<b>Total de artículos relevantes encontrados:</b> {report.get('total_articles', 0)}", meta_style))
            story.append(Paragraph("=" * 80, separator_style))
            
            # Generar PDF
            doc.build(story)
            logger.info(f"PDF mejorado generado exitosamente en: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generando PDF mejorado: {e}")
            return False
