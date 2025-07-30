# model/transformer.py
import torch
import torch.nn as nn
from model.transformer_components import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding

# --- Capa del Codificador (Encoder) ---
# El codificador se compone de una pila de estas capas. Cada capa procesa la
# secuencia de entrada para generar una representación más abstracta y contextual.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # 1. Mecanismo de Auto-Atención (Self-Attention):
        # Permite que cada palabra en la secuencia de entrada "mire" a todas las
        # demás palabras para capturar dependencias y relaciones entre ellas.
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 2. Red Feed-Forward:
        # Una red neuronal simple que se aplica a cada posición de forma independiente
        # para procesar la salida de la capa de atención y añadir complejidad.
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # 3. Normalización de Capa (Layer Normalization):
        # Estabiliza el entrenamiento al normalizar las salidas de las sub-capas.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 4. Dropout:
        # Técnica de regularización para prevenir el sobreajuste.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # La entrada 'x' pasa primero por la atención y luego por una conexión residual y normalización.
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # Conexión residual + Normalización

        # La salida pasa por la red feed-forward, seguida de otra conexión residual y normalización.
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # Conexión residual + Normalización
        return x

# --- Capa del Decodificador (Decoder) ---
# El decodificador también se compone de una pila de estas capas. Su función es
# generar la secuencia de salida (la traducción) palabra por palabra.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # 1. Auto-Atención Enmascarada (Masked Self-Attention):
        # Similar a la del codificador, pero "enmascarada" para que, al predecir
        # una palabra, solo pueda atender a las palabras anteriores en la secuencia de salida.
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # 2. Atención Cruzada (Cross-Attention):
        # ¡Aquí ocurre la magia! Esta capa permite al decodificador prestar atención
        # a la secuencia de entrada (la salida del codificador) para decidir qué
        # palabra traducir a continuación.
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) # Una normalización extra para la atención cruzada.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # La entrada 'x' (la secuencia de salida hasta ahora) pasa por la auto-atención enmascarada.
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # La atención cruzada relaciona la salida del decodificador con la salida del codificador.
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Finalmente, pasa por la red feed-forward.
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# --- Arquitectura Completa del Transformer ---
# Esta clase une todas las piezas: embeddings, codificación posicional,
# el codificador, el decodificador y la capa de salida final.
class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout, pad_idx):
        super(Transformer, self).__init__()
        
        # Guardamos el índice del token de padding para usarlo en las máscaras.
        self.pad_idx = pad_idx
        
        # --- Componentes del Modelo ---
        # 1. Embeddings: Convierten los tokens de entrada (números) en vectores densos.
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 2. Codificación Posicional: Añade información sobre la posición de cada token en la secuencia.
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 3. Pila de Codificadores y Decodificadores: El corazón del Transformer.
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 4. Capa de Salida: Proyecta la salida del decodificador al tamaño del vocabulario objetivo
        # para obtener las probabilidades de la siguiente palabra.
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Crea las máscaras necesarias para el proceso de atención.
        # Máscara de padding de la fuente (src_mask): para ignorar los tokens de padding en la entrada.
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Máscara de padding del objetivo (tgt_pad_mask): para ignorar los tokens de padding en la salida.
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Máscara de no-pico (nopeak_mask): impide que el decodificador "vea" tokens futuros en la secuencia de salida.
        seq_length = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).bool().to(tgt.device)
        nopeak_mask = ~nopeak_mask
        
        # La máscara final del objetivo combina la de padding y la de no-pico.
        tgt_mask = tgt_pad_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # --- Flujo de Datos a través del Modelo ---

        # 1. Entradas del Codificador: Se suman los embeddings y la codificación posicional.
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        # 2. Entradas del Decodificador: Lo mismo para la secuencia objetivo.
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # 3. Generación de máscaras.
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 4. Proceso del Codificador: La entrada pasa a través de todas las capas del codificador.
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # 5. Proceso del Decodificador: La entrada del decodificador y la salida del codificador
        # pasan a través de todas las capas del decodificador.
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # 6. Salida Final: Se aplica la capa lineal para obtener los logits.
        output = self.fc_out(dec_output)
        return output