# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pickle # Importamos pickle para guardar/cargar objetos de Python
from model.transformer import Transformer
# Importamos SRC_LANGUAGE y TGT_LANGUAGE para usarlos en el print de inferencia
from utils.data_preprocessing import get_dataloaders, SRC_LANGUAGE, TGT_LANGUAGE
from engine import train_epoch, evaluate, translate_sentence

def main(args):
    """
    Función principal que orquesta el entrenamiento o la inferencia del modelo.
    """
    
    # --- 1. CONFIGURACIÓN INICIAL ---
    # Concepto: Seleccionar el hardware adecuado (GPU si está disponible) es crucial
    # para acelerar el entrenamiento de modelos grandes como el Transformer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Concepto: El vocabulario es un diccionario que mapea cada palabra a un número único.
    # Es esencial guardarlo después de entrenar para poder usar el modelo correctamente en el futuro,
    # ya que el modelo solo entiende estos números, no las palabras en sí.
    vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")

    # --- 2. MODO DE ENTRENAMIENTO ---
    # Concepto: Esta sección se encarga de enseñar al modelo a traducir.
    # Implica preparar los datos, inicializar el modelo, y luego iterar sobre los datos
    # para ajustar los "pesos" o parámetros del modelo, minimizando el error en sus traducciones.
    if args.mode == 'train':
        print("Cargando y preprocesando datos (ES-EN)...")
        # Concepto: Los 'dataloaders' son iteradores eficientes que alimentan al modelo
        # con datos en pequeños lotes (batches) para optimizar el uso de memoria y la velocidad de entrenamiento.
        # También se crean los objetos de vocabulario (SRC, TGT).
        train_iterator, valid_iterator, test_iterator, SRC, TGT = get_dataloaders(device, args.batch_size)
        
        # Concepto: El tamaño del vocabulario y el índice del token de 'padding' (relleno)
        # son hiperparámetros fundamentales para construir la arquitectura del modelo correctamente.
        src_vocab_size = len(SRC.vocab)
        tgt_vocab_size = len(TGT.vocab)
        pad_idx = SRC.vocab.stoi[SRC.pad_token]
        
        # --- GUARDAR VOCABULARIOS ---
        # Se guardan los vocabularios para que el modo 'infer' pueda cargarlos
        # y procesar nuevas oraciones exactamente de la misma manera que se hizo en el entrenamiento.
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(vocab_path, 'wb') as f:
            pickle.dump((SRC, TGT), f)
        print(f"Vocabularios guardados en {vocab_path}")
        
        print("Inicializando modelo...")
        # Concepto: Se crea una instancia de la arquitectura Transformer con las dimensiones
        # y configuraciones especificadas (ej. número de capas, cabezas de atención, etc.).
        model = Transformer(
            src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
            args.num_layers, args.d_ff, args.max_len, args.dropout, pad_idx
        ).to(device)
        
        print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Concepto: El optimizador (Adam es una elección popular y robusta) es el algoritmo
        # que actualiza los pesos del modelo basándose en el error (loss) calculado.
        # La función de pérdida (CrossEntropyLoss) mide qué tan "equivocadas" están las predicciones del modelo.
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        best_val_loss = float('inf')
        
        print("Iniciando entrenamiento...")
        # Concepto: El bucle de entrenamiento itera varias "épocas". En cada época, el modelo
        # ve todo el conjunto de datos de entrenamiento y validación para aprender y ser evaluado.
        for epoch in range(args.epochs):
            # Se entrena el modelo y se evalúa su rendimiento en datos que no ha visto (validación).
            train_loss = train_epoch(model, train_iterator, optimizer, criterion, device)
            val_loss = evaluate(model, valid_iterator, criterion, device)
            
            print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

            # Concepto: Guardar el modelo solo cuando mejora (menor pérdida en validación)
            # es una práctica común para evitar el "sobreajuste" (overfitting) y quedarse con la mejor versión del modelo.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.model_path)
                print(f"Modelo guardado en {args.model_path}")

    # --- 3. MODO DE INFERENCIA ---
    # Concepto: Esta sección se usa para traducir texto nuevo utilizando un modelo ya entrenado.
    # No hay aprendizaje aquí, solo se aplican los conocimientos que el modelo ya adquirió.
    elif args.mode == 'infer':
        # --- COMPROBACIÓN DE ARCHIVOS ---
        if not os.path.exists(args.model_path) or not os.path.exists(vocab_path):
            print(f"Error: No se encontró el modelo '{args.model_path}' o el vocabulario '{vocab_path}'.")
            print("Por favor, entrena el modelo primero usando --mode train.")
            return

        # --- CARGAR VOCABULARIOS ---
        # Es CRÍTICO cargar los mismos vocabularios usados en el entrenamiento
        # para asegurar que las nuevas palabras se conviertan a los números correctos.
        with open(vocab_path, 'rb') as f:
            SRC, TGT = pickle.load(f)
        print(f"Vocabularios cargados desde {vocab_path}")
        
        src_vocab_size = len(SRC.vocab)
        tgt_vocab_size = len(TGT.vocab)
        pad_idx = SRC.vocab.stoi[SRC.pad_token]
        
        # --- INICIALIZAR MODELO ---
        # Se reconstruye la arquitectura del modelo con exactamente los mismos hiperparámetros que en el entrenamiento.
        print("Inicializando modelo...")
        model = Transformer(
            src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
            args.num_layers, args.d_ff, args.max_len, args.dropout, pad_idx
        ).to(device)

        # --- CARGAR PESOS Y TRADUCIR ---
        # Se cargan los pesos (el "conocimiento") que el modelo aprendió durante el entrenamiento.
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Modelo cargado desde {args.model_path}")
        
        # Se llama a la función que tokeniza la oración de entrada, la procesa a través del modelo
        # y genera la traducción de forma auto-regresiva (palabra por palabra).
        translation = translate_sentence(args.sentence, SRC, TGT, model, device)
        print(f"\nOración original ({SRC_LANGUAGE}): {args.sentence}")
        print(f"Traducción ({TGT_LANGUAGE}): {translation}")


# --- 4. PUNTO DE ENTRADA DEL SCRIPT ---
# Concepto: `argparse` es una herramienta estándar en Python para crear programas de línea de comandos robustos.
# Permite al usuario configurar fácilmente el comportamiento del script (ej. entrenar vs. inferir, número de épocas, etc.).
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar o ejecutar inferencia en un modelo Transformer.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], 
                       help='Modo de ejecución: "train" para entrenar, "infer" para traducir.')
    parser.add_argument('--sentence', type=str, 
                       help='Oración en español para traducir en modo inferencia.')
    
    # Argumentos que definen los hiperparámetros del modelo y el entrenamiento
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005) # Tasa de aprendizaje
    parser.add_argument('--d_model', type=int, default=512) # Dimensión de los embeddings
    parser.add_argument('--num_heads', type=int, default=8) # Número de cabezas de atención
    parser.add_argument('--num_layers', type=int, default=3) # Número de capas en Encoder/Decoder
    parser.add_argument('--d_ff', type=int, default=2048) # Dimensión de la capa Feed-Forward
    parser.add_argument('--max_len', type=int, default=100) # Longitud máxima de secuencia
    parser.add_argument('--dropout', type=float, default=0.1) # Regularización para evitar overfitting
    parser.add_argument('--model_path', type=str, default='saved_models/transformer_es_en.pt')

    args = parser.parse_args()

    # Validación para asegurar que el usuario provea una oración cuando quiera traducir.
    if args.mode == 'infer' and not args.sentence:
        parser.error("--sentence es requerido en modo 'infer'.")

    main(args)