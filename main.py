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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Definimos la ruta donde se guardará/cargará el vocabulario
    vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.pkl")

    if args.mode == 'train':
        print("Cargando y preprocesando datos (ES-EN)...")
        train_iterator, valid_iterator, test_iterator, SRC, TGT = get_dataloaders(device, args.batch_size)
        
        src_vocab_size = len(SRC.vocab)
        tgt_vocab_size = len(TGT.vocab)
        pad_idx = SRC.vocab.stoi[SRC.pad_token]
        
        # --- GUARDAR VOCABULARIOS ---
        # Guardamos los objetos SRC y TGT para usarlos después en la inferencia
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(vocab_path, 'wb') as f:
            pickle.dump((SRC, TGT), f)
        print(f"Vocabularios guardados en {vocab_path}")
        
        print("Inicializando modelo...")
        model = Transformer(
            src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
            args.num_layers, args.d_ff, args.max_len, args.dropout, pad_idx
        ).to(device)
        
        print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        best_val_loss = float('inf')
        
        print("Iniciando entrenamiento...")
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_iterator, optimizer, criterion, device)
            val_loss = evaluate(model, valid_iterator, criterion, device)
            
            print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.model_path)
                print(f"Modelo guardado en {args.model_path}")

    elif args.mode == 'infer':
        # --- COMPROBACIÓN DE ARCHIVOS ---
        if not os.path.exists(args.model_path) or not os.path.exists(vocab_path):
            print(f"Error: No se encontró el modelo '{args.model_path}' o el vocabulario '{vocab_path}'.")
            print("Por favor, entrena el modelo primero usando --mode train.")
            return

        # --- CARGAR VOCABULARIOS ---
        # Cargamos los mismos vocabularios que se usaron en el entrenamiento
        with open(vocab_path, 'rb') as f:
            SRC, TGT = pickle.load(f)
        print(f"Vocabularios cargados desde {vocab_path}")
        
        src_vocab_size = len(SRC.vocab)
        tgt_vocab_size = len(TGT.vocab)
        pad_idx = SRC.vocab.stoi[SRC.pad_token]
        
        # --- INICIALIZAR MODELO ---
        # Ahora el modelo se crea con las dimensiones correctas
        print("Inicializando modelo...")
        model = Transformer(
            src_vocab_size, tgt_vocab_size, args.d_model, args.num_heads,
            args.num_layers, args.d_ff, args.max_len, args.dropout, pad_idx
        ).to(device)

        # --- CARGAR PESOS Y TRADUCIR ---
        # Añadimos weights_only=True por seguridad y para eliminar el warning
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Modelo cargado desde {args.model_path}")
        
        translation = translate_sentence(args.sentence, SRC, TGT, model, device)
        print(f"\nOración original ({SRC_LANGUAGE}): {args.sentence}")
        print(f"Traducción ({TGT_LANGUAGE}): {translation}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar o ejecutar inferencia en un modelo Transformer.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], 
                       help='Modo de ejecución: "train" para entrenar, "infer" para traducir.')
    parser.add_argument('--sentence', type=str, 
                       help='Oración en español para traducir en modo inferencia.')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Cambié el nombre del modelo por defecto para que sea más genérico
    parser.add_argument('--model_path', type=str, default='saved_models/transformer_es_en.pt')

    args = parser.parse_args()

    if args.mode == 'infer' and not args.sentence:
        parser.error("--sentence es requerido en modo 'infer'.")

    main(args)