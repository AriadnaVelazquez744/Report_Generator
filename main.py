from telegram.extract_data_tg import ScraperT
from src.summarization.summarizer import PersonalizedSummarizer, TextRankSummarizer
from src.recommendation.vectorizer import NewsVectorizer, UserProfileVectorizer
from src.recommendation.matcher import NewsMatcher
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.report_generator import ReportGenerator
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
import os 
import json
import numpy as np
import spacy

nlp  = spacy.load('es_core_news_lg')

path = 'Data_articles'
data_dirs = [x for x in os.listdir(path) if not x.startswith(".")]

def load_raw_data(limit=None):
    """Carga datos crudos de artículos"""
    all_data = []
    count = 0
    for data_dir in data_dirs:
        dir_path = os.path.join(path, data_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(dir_path, filename)) as f:
                        article = json.load(f)
                        all_data.append(article)
                        count += 1
                        if limit and count >= limit:
                            return all_data
                except:
                    continue
    return all_data


def process_single_article(args):
    """Procesa un solo artículo (para paralelización con threading)"""
    idx, article_data, nlp, text_processor, annotator = args
    
    try:
        text = article_data.get('text', '')
        if not text:
            return None
        
        # Procesar con spaCy
        doc = nlp(text)
        
        # Extraer entidades
        current_ents = [{'text': e.text, 'label': e.label_} for e in doc.ents]
        
        # Anotar con regex
        annotations = annotator.annotate(text)
        
        # Preprocesar texto
        clean_tokens = text_processor.preprocess(text)
        clean_text = ' '.join(clean_tokens)
        
        return {
            'id': idx,
            'title': article_data.get('title', 'Sin título'),
            'text': text,
            'clean_text': clean_text,
            'categories': annotations['categories'],
            'entidades': current_ents,
            'section': article_data.get('section', 'General'),
            'tags': article_data.get('tags', []),
            'url': article_data.get('url', ''),
            'source_metadata': article_data.get('source_metadata', {}),
        }
    except Exception as e:
        print(f"⚠️ Error procesando artículo {idx}: {e}")
        return None


def prepare_articles(raw_data, text_processor, annotator, news_vectorizer, nlp):
    """
    Prepara artículos: extrae texto, categoriza con regex, limpia y vectoriza
    Usa ThreadPoolExecutor para paralelización real
    
    Returns:
        Lista de artículos procesados con vectores y metadatos
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tqdm
    import multiprocessing
    
    print(f"\nProcesando {len(raw_data)} artículos...")
    
    # Preparar argumentos para cada artículo
    tasks = [(i, article_data, nlp, text_processor, annotator) 
             for i, article_data in enumerate(raw_data)]
    
    articles = []
    clean_texts = []
    
    
    num_workers = multiprocessing.cpu_count()
    print(f"Procesando en paralelo con {num_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        futures = {executor.submit(process_single_article, task): task[0] 
                   for task in tasks}
        
        # Recopilar resultados con barra de progreso
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                articles.append(result)
                clean_texts.append(result['clean_text'])
    
    # Ordenar por ID original
    articles.sort(key=lambda x: x['id'])
    
    print(f"✅ {len(articles)} artículos procesados exitosamente")
    
    # Vectorizar todos los textos limpios
    print(f"\nVectorizando artículos con TF-IDF...")
    article_matrix = news_vectorizer.fit_transform0(clean_texts)
    print(f"✅ Matriz de artículos: {article_matrix.shape}")
    
    # Agregar vectores a los artículos
    for i, article in enumerate(articles):
        article['vector'] = article_matrix[i].tolist()
    
    return articles


def save_processed_articles(articles, filepath='processed_articles.json', vectorizer=None):
    """Guarda los artículos procesados y el vectorizador en un único archivo JSON"""
    print(f"\n💾 Guardando artículos procesados en {filepath}...")
    
    data = {
         'vectorizer': vectorizer.to_dict() if vectorizer else {},
        'articles': articles
       
    }

    def make_json_serializable(obj):
        """Recursively convert numpy types and other non-JSON types to native Python types."""
        # Numpy scalar
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        # Basic types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # Datetime
        try:
            from datetime import datetime
            if isinstance(obj, datetime):
                return obj.isoformat()
        except Exception:
            pass

        # Dict
        if isinstance(obj, dict):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}

        # Iterable (list/tuple)
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]

        # Fallback: try to cast to string
        try:
            return str(obj)
        except Exception:
            return None

    serializable_data = make_json_serializable(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Artículos guardados exitosamente")


def load_processed_articles(filepath='processed_articles.json'):
    """Carga los artículos procesados y vectorizador desde un archivo JSON"""
    if os.path.exists(filepath):
        print(f"\n📂 Cargando artículos procesados desde {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Compatibilidad con formato antiguo (solo lista de artículos)
        if isinstance(data, list):
            print(f"✅ {len(data)} artículos cargados desde cache (formato antiguo)")
            return {'articles': data, 'vectorizer_data': None}
        
        articles = data.get('articles', [])
        vectorizer_data = data.get('vectorizer', {})
        print(f"✅ {len(articles)} artículos cargados desde cache")
        
        return {'articles': articles, 'vectorizer_data': vectorizer_data}
    return None


def create_simulated_users():
    """Crea perfiles de usuarios simulados con diferentes intereses basados en categorías regex"""
    users = [
        {
            'name': 'Sofía - Crítica de Arte',
            'profile_text': (
                'Soy una apasionada del arte contemporáneo, las exposiciones y las galerías. '
                'Me interesan las obras de artistas emergentes, el muralismo, la escultura y '
                'la fotografía artística. Sigo festivales culturales, bienales de arte, '
                'inauguraciones de museos y eventos de patrimonio cultural. Me fascina el '
                'teatro, la danza, el cine de autor y las manifestaciones artísticas urbanas. '
                'Disfruto la música clásica, jazz, y expresiones folclóricas tradicionales.'
            )
        },
        {
            'name': 'Diego - Ambientalista',
            'profile_text': (
                'Me dedico a la conservación ambiental y protección de ecosistemas. '
                'Sigo temas de biodiversidad, especies en peligro de extinción, reservas naturales '
                'y parques nacionales. Me preocupan los desastres naturales como terremotos, '
                'inundaciones y huracanes. Denuncio la deforestación, contaminación de ríos, '
                'derrames de petróleo y el cambio climático. Apoyo energías renovables, '
                'reciclaje y desarrollo sostenible. Me interesan proyectos de reforestación '
                'y la protección de océanos y recursos hídricos.'
            )
        },
        {
            'name': 'Laura - Educadora Cultural',
            'profile_text': (
                'Me apasiona la educación, la literatura y la promoción cultural. '
                'Sigo lanzamientos de libros, ferias literarias, conciertos y recitales de poesía. '
                'Me interesan programas educativos, becas, talleres artísticos y actividades '
                'para niños y jóvenes. Apoyo bibliotecas comunitarias, centros culturales '
                'y espacios de creación artística. Me gusta el teatro comunitario, '
                'la música folclórica y las tradiciones ancestrales. Valoro la preservación '
                'del patrimonio inmaterial y las lenguas indígenas.'
            )
        },
        {
            'name': 'Martín - Fotógrafo de Naturaleza',
            'profile_text': (
                'Soy fotógrafo especializado en naturaleza, paisajes y vida silvestre. '
                'Me apasionan los parques naturales, santuarios de fauna, volcanes y montañas. '
                'Documento especies animales, aves migratorias, flora endémica y ecosistemas únicos. '
                'Me interesan expediciones científicas, descubrimientos de nuevas especies '
                'y proyectos de conservación de hábitats. Sigo fenómenos naturales, auroras, '
                'eclipses y eventos astronómicos. Apoyo el turismo ecológico y responsable.'
            )
        },
        {
            'name': 'Carmen - Historiadora del Arte',
            'profile_text': (
                'Investigo historia del arte latinoamericano, arquitectura colonial y '
                'patrimonio histórico. Me fascinan las restauraciones de monumentos, '
                'excavaciones arqueológicas y descubrimientos de sitios históricos. '
                'Estudio arte prehispánico, culturas indígenas y tradiciones artesanales. '
                'Me interesan museos, archivos históricos, documentales culturales '
                'y la preservación de arte sacro. Valoro el arte popular, textiles tradicionales '
                'y técnicas ancestrales de pintura y cerámica.'
            )
        },
    ]
    return users


def main(nlp):

    print("=" * 80)
    print("SISTEMA DE RECOMENDACIÓN DE NOTICIAS PERSONALIZADO")
    print("=" * 80)
    
    # Inicializar componentes
    text_processor = TextPreprocessor(use_spacy=False)
    annotator = RegexAnnotator()
    
    # Inicializar vectorizador de noticias (necesario siempre para perfiles de usuario)
    news_vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
    
    # Intentar cargar artículos procesados desde cache
    processed_cache_file = 'processed_articles.json'
    cache_data = load_processed_articles(processed_cache_file)
    
    if cache_data is None:
        # No existe cache, procesar artículos desde cero
        print("\n📂 Cargando artículos crudos...")
        raw_data = load_raw_data()  # Cambia el limit o quítalo para cargar todos
        print(f"✅ {len(raw_data)} artículos crudos cargados")
        
        # Preparar artículos: categorizar, limpiar y vectorizar
        articles = prepare_articles(raw_data, text_processor, annotator, news_vectorizer, nlp)
        
        # Guardar en cache para futuras ejecuciones
        save_processed_articles(articles, processed_cache_file, vectorizer=news_vectorizer)
    else:
        # Cargar artículos desde cache
        articles = cache_data['articles']
        vectorizer_data = cache_data['vectorizer_data']
        
        if vectorizer_data:
            # Cargar vectorizador desde datos en JSON
            print("\n🔧 Restaurando vectorizador desde cache...")
            news_vectorizer = NewsVectorizer.from_dict(vectorizer_data)
            if news_vectorizer:
                print("✅ Vectorizador restaurado")
            else:
                # Fallback si falla la deserialización
                print("⚠️  Error restaurando vectorizador, reajustando...")
                clean_texts = [article['clean_text'] for article in articles]
                news_vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
                news_vectorizer.fit0(clean_texts)
                print("✅ Vectorizador ajustado")
        else:
            # Formato antiguo sin vectorizador, necesitamos ajustar
            print("\n🔧 Ajustando vectorizador con artículos del cache...")
            clean_texts = [article['clean_text'] for article in articles]
            news_vectorizer.fit0(clean_texts)
            print("✅ Vectorizador ajustado")

    # Crear perfiles de usuarios simulados
    print("\n👥 Creando usuarios simulados...")
    simulated_users = create_simulated_users()
    
    # Inicializar componentes de recomendación
    profile_vectorizer = UserProfileVectorizer(news_vectorizer)
    profile_manager = UserProfileManager(profile_vectorizer)
    matcher = NewsMatcher()
    
    # Inicializar resumidores
    base_summarizer = TextRankSummarizer(language="spanish")
    personalized_summarizer = PersonalizedSummarizer(base_summarizer)
    
    # Inicializar generador de reportes
    report_generator = ReportGenerator(personalized_summarizer)
    
    # Procesar cada usuario
    print("\n" + "=" * 80)
    print("GENERANDO RECOMENDACIONES PERSONALIZADAS")
    print("=" * 80)
    
    all_reports = []
    
    # Crear directorio para PDFs
    pdf_output_dir = "reportes_pdf"
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    for user in simulated_users:
        print(f"\n{'='*80}")
        print(f"👤 Usuario: {user['name']}")
        print(f"{'='*80}")
        print(f"📝 Perfil: {user['profile_text'][:100]}...")
        
        # Crear perfil del usuario con extracción de entidades
        user_profile = profile_manager.create_profile(user['profile_text'], nlp=nlp)
        
        print(f"\n🏷️  Categorías de interés detectadas: {user_profile['categories'][:8]}")
        print(f"👤 Entidades de interés: {[e['text'] for e in user_profile.get('entities', [])][:10]}")
        print(f"📊 Dimensión del vector de perfil: {len(user_profile['vector'])}")
        
        # Encontrar artículos relevantes
        matches = matcher.match_articles(user_profile, articles, top_k=10)
        
        # Generar reporte personalizado
        report = report_generator.generate_report(matches, user_profile, max_articles=5)
        all_reports.append({
            'user_name': user['name'],
            'report': report
        })
        
        import time
        # Generar PDF
        # Crear nombre de archivo seguro
        safe_name = user['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
        pdf_filename = f"{pdf_output_dir}/reporte_{safe_name}_{int(time.time())}.pdf"
        
        print(f"\n📄 Generando PDF...")
        if report_generator.generate_pdf(report, pdf_filename, user['name']):
            print(f"✅ PDF guardado en: {pdf_filename}")
        else:
            print(f"⚠️  No se pudo generar el PDF (instala reportlab: pip install reportlab)")
        
        print(f"\n{'='*80}\n")
    
    # Estadísticas generales
    print("\n" + "=" * 80)
    print("📊 ESTADÍSTICAS GENERALES")
    print("=" * 80)
    
    # Categorías más comunes en artículos
    category_counts = {}
    for article in articles:
        for cat in article['categories']:
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n🏆 Top 10 categorías más frecuentes en artículos:")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat, count in sorted_cats:
        print(f"   {cat}: {count} artículos")
    
    print(f"\n📁 Reportes PDF guardados en: {pdf_output_dir}/")
    print("\n✅ Sistema completado exitosamente!")


if __name__ == "__main__":
    main(nlp)
