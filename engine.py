# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
import spacy

def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(iterator, desc="Training"):
        src = batch.es_sentence.to(device)  # [batch_size, src_len]
        tgt = batch.en_sentence.to(device)  # [batch_size, tgt_len]
        
        optimizer.zero_grad()
        
        # El modelo espera [batch_size, seq_len]
        # El target para el decoder es hasta el penúltimo token
        output = model(src, tgt[:, :-1])
        
        # Preparar la salida y el objetivo para la pérdida
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        # El target para la loss es desde el segundo token
        tgt_for_loss = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt_for_loss)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch.es_sentence.to(device)
            tgt = batch.en_sentence.to(device)
            
            output = model(src, tgt[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_for_loss = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt_for_loss)
            total_loss += loss.item()
            
    return total_loss / len(iterator)

def translate_sentence(sentence, src_field, tgt_field, model, device, max_len=50):
    model.eval()
    
    # Tokenizar con el tokenizador de spacy correcto
    if isinstance(sentence, str):
        spacy_es = spacy.load('es_core_news_sm')
        tokens = [tok.text.lower() for tok in spacy_es.tokenizer(sentence)]
    else:
        tokens = [tok.lower() for tok in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device) # [1, src_len]

    # Generación auto-regresiva
    tgt_indices = [tgt_field.vocab.stoi[tgt_field.init_token]]
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device) # [1, tgt_len]
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        pred_token_idx = output.argmax(2)[:, -1].item()
        tgt_indices.append(pred_token_idx)
        
        if pred_token_idx == tgt_field.vocab.stoi[tgt_field.eos_token]:
            break
            
    tgt_tokens = [tgt_field.vocab.itos[i] for i in tgt_indices]
    
    return " ".join(tgt_tokens[1:-1])