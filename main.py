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


def clean_article_noise(text: str) -> str:
    """Elimina patrones de ruido como 'LEA TAMBIÉN:."""
    import re
    if not text:
        return ""
    
    # Patrones de referencias a otros artículos
    patterns = [
        r'LEA\s+TAMBIÉN\s*[:.].*?(?=\.\s|$)',
      
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Limpiar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_single_article(args):
    """Procesa un solo artículo (para paralelización con threading)"""
    idx, article_data, nlp, text_processor, annotator = args
    
    try:
        text = article_data.get('text', '')
        if not text:
            return None
        
        # Limpiar ruido antes de procesar
        text = clean_article_noise(text)
        
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
    """Crea perfiles de usuarios simulados enfocados en política latinoamericana"""
    users = [
        {
            'name': 'María - Analista de Política Latinoamericana',
            'profile_text': (
                'Sigo de cerca los procesos políticos en América Latina, especialmente en Venezuela, '
                'Cuba, Nicaragua, Bolivia y México. Me interesan los gobiernos progresistas, '
                'el socialismo del siglo XXI y las políticas de izquierda. Analizo elecciones, '
                'reformas constitucionales, asambleas nacionales y decisiones del poder ejecutivo. '
                'Sigo a líderes como Maduro, Díaz-Canel, Petro, AMLO y Lula. Me preocupan '
                'los golpes de estado, la injerencia extranjera y las sanciones de Estados Unidos. '
                'Apoyo la soberanía nacional, la integración regional y organismos como CELAC y ALBA.'
            )
        },
        {
            'name': 'Carlos - Corresponsal de Conflictos Internacionales',
            'profile_text': (
                'Cubro conflictos armados, guerras y crisis geopolíticas a nivel mundial. '
                'Me especializo en el conflicto Israel-Palestina, la guerra en Ucrania, '
                'tensiones en Medio Oriente y conflictos en África. Denuncio crímenes de guerra, '
                'bombardeos a civiles, uso de armas prohibidas y violaciones del derecho internacional. '
                'Sigo las acciones de la ONU, el Consejo de Seguridad, la Corte Penal Internacional '
                'y organizaciones humanitarias. Me interesan los refugiados, desplazados, '
                'crisis humanitarias y operaciones de paz. Analizo el papel de potencias como '
                'Estados Unidos, Rusia, China e Irán en los conflictos globales.'
            )
        },
        {
            'name': 'Rosa - Defensora de Derechos Humanos',
            'profile_text': (
                'Me dedico a documentar violaciones de derechos humanos en América Latina. '
                'Sigo casos de represión política, presos políticos, persecución a opositores '
                'y asesinatos de líderes sociales. Me preocupan los pueblos indígenas, '
                'comunidades afrodescendientes, campesinos y trabajadores. Denuncio '
                'la violencia policial, paramilitares, narcotráfico y crimen organizado. '
                'Apoyo movimientos sociales, sindicatos, organizaciones de mujeres y colectivos LGBTQ+. '
                'Sigo informes de Amnistía Internacional, Human Rights Watch y la CIDH. '
                'Valoro la justicia social, la memoria histórica y la verdad sobre dictaduras pasadas.'
            )
        },
        {
            'name': 'Jorge - Economista Político',
            'profile_text': (
                'Analizo la economía política de América Latina y el impacto de las sanciones. '
                'Me interesan las políticas económicas de Venezuela, Cuba y Nicaragua bajo bloqueo. '
                'Sigo el precio del petróleo, la inflación, el tipo de cambio y la deuda externa. '
                'Estudio el papel del FMI, Banco Mundial y las políticas de austeridad. '
                'Me preocupan la pobreza, la desigualdad, el desempleo y la crisis alimentaria. '
                'Apoyo la nacionalización de recursos, la reforma agraria y la soberanía económica. '
                'Analizo tratados comerciales, inversiones chinas y rusas en la región, '
                'y alternativas al dólar como moneda de intercambio.'
            )
        },
        {
            'name': 'Lucía - Periodista de Política Electoral',
            'profile_text': (
                'Cubro procesos electorales, campañas políticas y resultados de votaciones '
                'en toda América Latina. Me interesan las elecciones presidenciales, legislativas '
                'y referéndums en Venezuela, Colombia, Brasil, Argentina, México y Chile. '
                'Analizo encuestas, debates presidenciales, fraudes electorales y observación internacional. '
                'Sigo partidos políticos de izquierda y derecha, coaliciones y alianzas. '
                'Me preocupa la participación ciudadana, el voto electrónico y la transparencia electoral. '
                'Documento victorias progresistas, derrotas de la derecha y cambios de gobierno. '
                'Valoro la democracia, las instituciones electorales y el respeto al voto popular.'
            )
        },
        {
            'name': 'Fernando - Analista Antiimperialista',
            'profile_text': (
                'Estudio las relaciones de poder entre Estados Unidos y América Latina. '
                'Denuncio el imperialismo, las intervenciones militares, golpes de estado '
                'y operaciones de cambio de régimen patrocinadas por la CIA. Me interesan '
                'las sanciones económicas contra Venezuela, Cuba, Nicaragua e Irán. '
                'Sigo las bases militares estadounidenses, el Comando Sur y la OTAN. '
                'Apoyo la multipolaridad, el BRICS, la cooperación Sur-Sur y la desdolarización. '
                'Analizo el papel de China y Rusia como contrapeso a la hegemonía estadounidense. '
                'Me preocupan los medios de comunicación occidentales y la guerra de información.'
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
    print("\n📊 Inicializando matcher ...")
    profile_vectorizer = UserProfileVectorizer(news_vectorizer)
    profile_manager = UserProfileManager(profile_vectorizer)
    
    # Crear matcher desde artículos
    matcher = NewsMatcher.from_articles(articles, vectorizer=news_vectorizer)
    print(f"✅ Matcher inicializado con {len(articles)} artículos")
    
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
        
        print(f"\n🏷️  Categorías detectadas: {user_profile['categories'][:10]}")
        
        # Mostrar entidades por tipo
        entities = user_profile.get('entities', [])
        ent_by_type = {}
        for e in entities:
            label = e.get('label', 'MISC')
            if label in {'PER', 'ORG', 'GPE', 'LOC'}:
                if label not in ent_by_type:
                    ent_by_type[label] = []
                ent_by_type[label].append(e['text'])
        
        if ent_by_type:
            print("🔍 Entidades extraídas:")
            for label, texts in ent_by_type.items():
                label_name = {'PER': 'Personas', 'ORG': 'Organizaciones', 'GPE': 'Países/Ciudades', 'LOC': 'Lugares'}.get(label, label)
                print(f"   {label_name}: {texts[:5]}")
        
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
    
    
    
    print(f"\n📁 Reportes PDF guardados en: {pdf_output_dir}/")
    print("\n✅ Sistema completado exitosamente!")


if __name__ == "__main__":
    main(nlp)
