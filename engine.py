# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
import spacy

# --- ENTRENAMIENTO ---
# Esta función encapsula la lógica para entrenar el modelo durante una "época" (un pase completo sobre el dataset).
def train_epoch(model, iterator, optimizer, criterion, device):
    """
    Realiza una época de entrenamiento completa para el modelo Transformer.
    """
    # Ponemos el modelo en modo "entrenamiento". Esto activa capas como el Dropout.
    model.train()
    # Inicializamos la pérdida total de esta época en cero.
    total_loss = 0
    
    # Iteramos sobre cada batch de datos que nos proporciona el iterador.
    # tqdm es una librería que nos muestra una barra de progreso, ¡muy útil!
    for batch in tqdm(iterator, desc="Training"):
        # Movemos las oraciones de origen (español) y destino (inglés) al dispositivo (GPU o CPU).
        src = batch.es_sentence.to(device)
        tgt = batch.en_sentence.to(device)
        
        # Reiniciamos los gradientes del optimizador. Es importante hacerlo en cada iteración.
        optimizer.zero_grad()
        
        # --- PASO HACIA ADELANTE (FORWARD PASS) ---
        # Le pasamos la oración de origen y la de destino (sin el último token <eos>) al modelo.
        # El modelo intentará predecir la oración de destino basándose en la de origen.
        output = model(src, tgt[:, :-1])
        
        # --- CÁLCULO DE LA PÉRDIDA (LOSS) ---
        # Para calcular la pérdida, necesitamos redimensionar la salida del modelo y el objetivo.
        output_dim = output.shape[-1] # Tamaño del vocabulario de destino.
        # Aplanamos la salida del modelo a 2D: [num_tokens, vocab_size]
        output = output.reshape(-1, output_dim)
        # Aplanamos el target a 1D: [num_tokens]. Usamos la oración de destino sin el primer token <sos>.
        tgt_for_loss = tgt[:, 1:].reshape(-1)
        
        # La función de pérdida (CrossEntropyLoss) compara la predicción del modelo con el objetivo real.
        loss = criterion(output, tgt_for_loss)
        
        # --- PASO HACIA ATRÁS (BACKWARD PASS) ---
        # Calculamos los gradientes de la pérdida con respecto a los parámetros del modelo.
        loss.backward()
        
        # "Clip grad norm" ayuda a prevenir el problema del "gradiente explosivo", estabilizando el entrenamiento.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Actualizamos los pesos del modelo usando el optimizador (Adam).
        optimizer.step()
        
        # Acumulamos la pérdida de este batch. .item() extrae el valor numérico.
        total_loss += loss.item()
        
    # Devolvemos el promedio de la pérdida durante esta época.
    return total_loss / len(iterator)

# --- EVALUACIÓN ---
# Esta función evalúa qué tan bien funciona el modelo en un conjunto de datos que no ha visto durante el entrenamiento (validación).
def evaluate(model, iterator, criterion, device):
    """
    Evalúa el rendimiento del modelo en el conjunto de datos de validación.
    """
    # Ponemos el modelo en modo "evaluación". Esto desactiva capas como el Dropout.
    model.eval()
    # Inicializamos la pérdida total en cero.
    total_loss = 0
    
    # "with torch.no_grad()" le dice a PyTorch que no necesitamos calcular gradientes aquí.
    # Esto ahorra memoria y acelera el proceso, ya que no vamos a entrenar.
    with torch.no_grad():
        # Iteramos sobre el conjunto de datos de validación.
        for batch in tqdm(iterator, desc="Evaluating"):
            # Movemos los datos al dispositivo correspondiente.
            src = batch.es_sentence.to(device)
            tgt = batch.en_sentence.to(device)
            
            # Obtenemos la predicción del modelo.
            output = model(src, tgt[:, :-1])
            
            # Redimensionamos la salida y el objetivo para calcular la pérdida, igual que en el entrenamiento.
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_for_loss = tgt[:, 1:].reshape(-1)
            
            # Calculamos la pérdida.
            loss = criterion(output, tgt_for_loss)
            total_loss += loss.item()
            
    # Devolvemos el promedio de la pérdida en el conjunto de validación.
    return total_loss / len(iterator)

# --- TRADUCCIÓN (INFERENCIA) ---
# Esta función utiliza el modelo ya entrenado para traducir una nueva oración.
def translate_sentence(sentence, src_field, tgt_field, model, device, max_len=50):
    """
    Traduce una oración de español a inglés usando el modelo entrenado.
    """
    # Nos aseguramos de que el modelo esté en modo evaluación.
    model.eval()
    
    # --- PREPROCESAMIENTO DE LA ORACIÓN DE ENTRADA ---
    # Tokenizamos la oración de entrada (la separamos en palabras/tokens).
    if isinstance(sentence, str):
        spacy_es = spacy.load('es_core_news_sm')
        tokens = [tok.text.lower() for tok in spacy_es.tokenizer(sentence)]
    else:
        tokens = [tok.lower() for tok in sentence]

    # Añadimos los tokens especiales de inicio (<sos>) y fin (<eos>).
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # Convertimos los tokens a sus correspondientes índices numéricos del vocabulario.
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    
    # Convertimos la lista de índices en un tensor de PyTorch y lo enviamos al dispositivo.
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # --- GENERACIÓN AUTO-REGRESIVA DE LA TRADUCCIÓN ---
    # Empezamos la traducción con el token de inicio <sos>.
    tgt_indices = [tgt_field.vocab.stoi[tgt_field.init_token]]
    
    # Generamos la traducción token por token hasta un máximo de 'max_len'.
    for _ in range(max_len):
        # Convertimos los tokens de destino generados hasta ahora en un tensor.
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        
        # No calculamos gradientes durante la inferencia.
        with torch.no_grad():
            # El modelo predice el siguiente token basándose en la oración de origen y lo que ha traducido hasta ahora.
            output = model(src_tensor, tgt_tensor)
        
        # Escogemos el token con la probabilidad más alta (argmax) de la última posición de la secuencia.
        pred_token_idx = output.argmax(2)[:, -1].item()
        # Añadimos el token predicho a nuestra lista de traducción.
        tgt_indices.append(pred_token_idx)
        
        # Si el modelo predice el token de fin (<eos>), detenemos la traducción.
        if pred_token_idx == tgt_field.vocab.stoi[tgt_field.eos_token]:
            break
            
    # Convertimos los índices de la traducción de vuelta a tokens (palabras).
    tgt_tokens = [tgt_field.vocab.itos[i] for i in tgt_indices]
    
    # Unimos los tokens para formar la oración final, eliminando los tokens <sos> y <eos>.
    return " ".join(tgt_tokens[1:-1])