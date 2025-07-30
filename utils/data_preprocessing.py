# utils/data_preprocessing.py
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import os
import requests
import zipfile
import io

# --- 1. Configuración de Lenguaje y Tokenizadores ---
# El primer paso en cualquier proyecto de NLP es definir los idiomas con los que trabajaremos.
# Esto nos ayuda a cargar las herramientas correctas (como los tokenizadores) para cada uno.
SRC_LANGUAGE = 'es' # Idioma fuente: Español
TGT_LANGUAGE = 'en' # Idioma objetivo: Inglés

# Para que el modelo entienda el texto, primero debemos dividirlo en unidades más pequeñas llamadas "tokens".
# Usamos 'spacy', una librería de NLP muy potente, que tiene modelos pre-entrenados para tokenizar
# de forma inteligente en diferentes idiomas, respetando la gramática y estructura de cada uno.
try:
    spacy_es = spacy.load('es_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    # Si los modelos de spacy no están instalados, los descargamos automáticamente.
    print("Descargando modelos de lenguaje para spacy (es/en)...")
    from spacy.cli import download
    download('es_core_news_sm')
    download('en_core_web_sm')
    spacy_es = spacy.load('es_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

def tokenize_es(text):
    """
    Tokenizador para español.
    Convierte una cadena de texto (oración) en una lista de tokens (palabras).
    Ejemplo: "hola mundo" -> ["hola", "mundo"]
    """
    return [tok.text for tok in spacy_es.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizador para inglés.
    Hace lo mismo que el anterior, pero usando las reglas del idioma inglés.
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# 'torchtext.data.Field' es una herramienta increíblemente útil. Define un "plano" de cómo se deben procesar los datos de texto.
# Le decimos qué tokenizador usar, si añadir tokens especiales, y si convertir todo a minúsculas.
# Tokens especiales:
#   - <sos>: "Start of Sentence" (Inicio de oración). Marca el comienzo de cada secuencia.
#   - <eos>: "End of Sentence" (Fin de oración). Marca el final.
# Estos tokens ayudan al modelo a saber cuándo empezar y terminar de generar una traducción.
# 'batch_first=True' es crucial para los Transformers, ya que organiza los datos en tensores con la
# dimensión del batch primero ([batch_size, seq_len]), que es como la arquitectura espera la entrada.
SRC = Field(tokenize=tokenize_es,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TGT = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

# --- 2. Descarga y Preparación del Dataset ---
def download_and_prepare_data():
    """
    Esta función se encarga de obtener los datos de entrenamiento.
    Los modelos de machine learning necesitan muchos ejemplos para aprender.
    Aquí, descargamos un corpus paralelo (oraciones en español con sus traducciones al inglés)
    del sitio de Tatoeba y lo formateamos a un archivo TSV (valores separados por tabuladores)
    que es fácil de leer para torchtext.
    """
    data_path = 'data'
    tsv_path = os.path.join(data_path, 'spa-eng.tsv')

    # Si ya hemos descargado y procesado los datos, no lo hacemos de nuevo para ahorrar tiempo.
    if os.path.exists(tsv_path):
        print(f"Dataset ya encontrado en {tsv_path}")
        return tsv_path

    print("Dataset no encontrado. Descargando y preparando...")
    os.makedirs(data_path, exist_ok=True)
    
    url = "http://www.manythings.org/anki/spa-eng.zip"
    
    # A veces, los servidores bloquean las peticiones automáticas.
    # Añadimos una cabecera 'User-Agent' para simular que la petición viene de un navegador web,
    # lo que suele evitar problemas de descarga (Error 403 Forbidden).
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Lanza un error si la descarga falló.

        z = zipfile.ZipFile(io.BytesIO(response.content))

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")
        return None 
    except zipfile.BadZipFile:
        print("Error: El archivo descargado no es un ZIP válido o está corrupto.")
        return None

    # Extraemos el contenido del archivo .zip en nuestra carpeta de datos.
    z.extractall(data_path)

    txt_path = os.path.join(data_path, 'spa.txt')

    # El archivo original tiene el formato "Inglés \t Español". Lo convertimos a un formato TSV
    # con cabeceras ("es_sentence", "en_sentence"), que es el que espera `TabularDataset` de torchtext.
    with open(txt_path, 'r', encoding='utf-8') as f_in, \
         open(tsv_path, 'w', encoding='utf-8') as f_out:
        f_out.write('es_sentence\ten_sentence\n') # Escribimos la cabecera del TSV.
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) >= 2: # Nos aseguramos que la línea tiene al menos inglés y español.
                en = parts[0]
                es = parts[1]
                f_out.write(f'{es}\t{en}\n') # Escribimos la oración en español, luego en inglés.

    print(f"Dataset preparado y guardado en {tsv_path}")
    return tsv_path

# --- 3. Creación de DataLoaders ---
def get_dataloaders(device, batch_size):
    """
    Esta es la función principal que orquesta todo el preprocesamiento.
    Convierte los datos crudos en lotes (batches) de tensores numéricos listos para ser
    alimentados al modelo Transformer.
    """
    dataset_path = download_and_prepare_data()

    # Mapeamos las columnas de nuestro archivo TSV ('es_sentence', 'en_sentence') a los 'Fields'
    # que definimos antes (SRC, TGT). Así, torchtext sabe cómo procesar cada columna.
    fields = [('es_sentence', SRC), ('en_sentence', TGT)]

    # `TabularDataset` carga los datos desde el archivo TSV y los procesa usando los 'Fields' especificados.
    dataset = TabularDataset(
        path=dataset_path,
        format='tsv',
        fields=fields,
        skip_header=True # Importante para que no lea la cabecera como un ejemplo de datos.
    )

    # Dividimos el dataset en tres conjuntos:
    # 1. Entrenamiento (train): La mayor parte de los datos, usados para que el modelo aprenda los patrones.
    # 2. Validación (valid): Usados para evaluar el modelo durante el entrenamiento y ajustar hiperparámetros.
    # 3. Prueba (test): Usados una sola vez al final para dar una evaluación final e imparcial del rendimiento.
    train_data, valid_data, test_data = dataset.split(
        split_ratio=[0.8, 0.1, 0.1] # 80% para entrenar, 10% para validar, 10% para probar.
    )

    print(f"Número de ejemplos de entrenamiento: {len(train_data.examples)}")
    print(f"Número de ejemplos de validación: {len(valid_data.examples)}")
    print(f"Número de ejemplos de prueba: {len(test_data.examples)}")

    # Construimos el "vocabulario": un mapeo de cada palabra única a un número entero.
    # Los modelos neuronales solo entienden números, no texto.
    # `min_freq=2` significa que solo se incluirán en el vocabulario las palabras que aparezcan
    # al menos 2 veces. Esto ayuda a ignorar errores tipográficos y palabras muy raras,
    # haciendo el vocabulario más pequeño y el modelo más robusto.
    SRC.build_vocab(train_data, min_freq=2)
    TGT.build_vocab(train_data, min_freq=2)

    print(f"Tamaño del vocabulario fuente ({SRC_LANGUAGE}): {len(SRC.vocab)}")
    print(f"Tamaño del vocabulario objetivo ({TGT_LANGUAGE}): {len(TGT.vocab)}")

    # `BucketIterator` es un iterador inteligente. Agrupa oraciones de longitudes similares en los mismos lotes.
    # Esto hace que el "padding" (añadir tokens extra para que todas las oraciones en un lote tengan la misma
    # longitud) sea mucho más eficiente, acelerando el entrenamiento.
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.es_sentence), # Ordena los ejemplos por longitud para agruparlos eficientemente.
        sort_within_batch=False,
        device=device # Mueve los tensores directamente a la GPU si está disponible.
    )
    
    # Finalmente, devolvemos los iteradores (que generan los lotes de datos) y los objetos Field
    # (que contienen los vocabularios y la configuración de preprocesamiento).
    return train_iterator, valid_iterator, test_iterator, SRC, TGT