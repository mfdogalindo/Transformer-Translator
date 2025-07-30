# utils/data_preprocessing.py
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import os
import requests
import zipfile
import io

# --- 1. Configuración de Lenguaje y Tokenizadores ---
SRC_LANGUAGE = 'es'
TGT_LANGUAGE = 'en'

# Cargar modelos de spacy (con los nombres completos y correctos)
try:
    spacy_es = spacy.load('es_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("Descargando modelos de lenguaje para spacy (es/en)...")
    from spacy.cli import download
    download('es_core_news_sm')
    download('en_core_web_sm')
    spacy_es = spacy.load('es_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

def tokenize_es(text):
    """Tokenizador para español"""
    return [tok.text for tok in spacy_es.tokenizer(text)]

def tokenize_en(text):
    """Tokenizador para inglés"""
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Definir los Fields para el preprocesamiento
SRC = Field(tokenize=tokenize_es,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True) # batch_first=True para [batch_size, seq_len]

TGT = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

# --- 2. Descarga y Preparación del Dataset ---
def download_and_prepare_data():
    """
    Descarga el dataset de Anki/Tatoeba, lo procesa y guarda como TSV.
    Devuelve la ruta al archivo TSV.
    """
    data_path = 'data'
    tsv_path = os.path.join(data_path, 'spa-eng.tsv')

    if os.path.exists(tsv_path):
        print(f"Dataset ya encontrado en {tsv_path}")
        return tsv_path

    print("Dataset no encontrado. Descargando y preparando...")
    os.makedirs(data_path, exist_ok=True)
    
    url = "http://www.manythings.org/anki/spa-eng.zip"
    
    # --- INICIO DE LA SECCIÓN MODIFICADA ---
    # Añadimos una cabecera para simular ser un navegador
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Pasamos la cabecera a la petición
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() 

        z = zipfile.ZipFile(io.BytesIO(response.content))

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")
        return None 
    except zipfile.BadZipFile:
        print("Error: El archivo descargado no es un ZIP válido o está corrupto.")
        return None
    # --- FIN DE LA SECCIÓN MODIFICADA ---

    z.extractall(data_path)

    txt_path = os.path.join(data_path, 'spa.txt')

    with open(txt_path, 'r', encoding='utf-8') as f_in, \
         open(tsv_path, 'w', encoding='utf-8') as f_out:
        f_out.write('es_sentence\ten_sentence\n') # Escribir cabecera
        for line in f_in:
            parts = line.strip().split('\t')
            # Si la línea tiene la estructura esperada (al menos 3 partes)
            if len(parts) >= 3:
                # La estructura es: Ingles, Español, Atribución
                en = parts[0]
                es = parts[1]
                f_out.write(f'{es}\t{en}\n')

    print(f"Dataset preparado y guardado en {tsv_path}")
    return tsv_path

# --- 3. Creación de DataLoaders ---
def get_dataloaders(device, batch_size):
    """
    Crea vocabularios e iteradores para el dataset ES-EN usando TabularDataset.
    """
    dataset_path = download_and_prepare_data()

    # Definir los campos que coinciden con la cabecera del TSV
    fields = [('es_sentence', SRC), ('en_sentence', TGT)]

    # Cargar los datos usando TabularDataset
    dataset = TabularDataset(
        path=dataset_path,
        format='tsv',
        fields=fields,
        skip_header=True # Saltar la cabecera que escribimos
    )

    # Dividir en conjuntos de entrenamiento, validación y prueba (ej. 80/10/10)
    train_data, valid_data, test_data = dataset.split(
        split_ratio=[0.8, 0.1, 0.1]
    )

    print(f"Número de ejemplos de entrenamiento: {len(train_data.examples)}")
    print(f"Número de ejemplos de validación: {len(valid_data.examples)}")
    print(f"Número de ejemplos de prueba: {len(test_data.examples)}")

    # Construir el vocabulario
    SRC.build_vocab(train_data, min_freq=2)
    TGT.build_vocab(train_data, min_freq=2)

    print(f"Tamaño del vocabulario fuente ({SRC_LANGUAGE}): {len(SRC.vocab)}")
    print(f"Tamaño del vocabulario objetivo ({TGT_LANGUAGE}): {len(TGT.vocab)}")

    # Crear los iteradores
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.es_sentence), # Ordenar para padding eficiente
        sort_within_batch=False,
        device=device
    )
    
    return train_iterator, valid_iterator, test_iterator, SRC, TGT