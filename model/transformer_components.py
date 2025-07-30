# model/transformer_components.py
import torch
import torch.nn as nn
import math

# --- 1. Atención Multi-Cabeza (Multi-Head Attention) ---
# La atención es el corazón del Transformer. Permite al modelo sopesar la importancia
# de diferentes palabras en la secuencia de entrada al procesar una palabra específica.
# "Multi-cabeza" significa que este proceso se realiza varias veces en paralelo,
# y cada "cabeza" puede aprender a enfocarse en diferentes tipos de relaciones
# (por ejemplo, una cabeza puede enfocarse en relaciones sintácticas y otra en semánticas).

class MultiHeadAttention(nn.Module):
    """
    Implementación de la Atención Multi-Cabeza. [1]
    Permite al modelo atender a información de diferentes subespacios de representación.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # d_model es la dimensionalidad de los embeddings de las palabras.
        # num_heads es el número de "cabezas de atención" en las que dividiremos el proceso.
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        # d_k es la dimensión de cada cabeza de atención.
        self.d_k = d_model // num_heads

        # Creamos capas lineales para transformar las entradas Q (Query), K (Key) y V (Value).
        # Estas transformaciones proyectan los embeddings en un espacio donde pueden ser comparados.
        self.W_q = nn.Linear(d_model, d_model) # Matriz de pesos para Query
        self.W_k = nn.Linear(d_model, d_model) # Matriz de pesos para Key
        self.W_v = nn.Linear(d_model, d_model) # Matriz de pesos para Value
        self.W_o = nn.Linear(d_model, d_model) # Matriz de pesos para la salida final

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calcula la atención de producto escalar escalonado. [1, 2]
        Esta es la fórmula de atención principal: softmax((Q * K^T) / sqrt(d_k)) * V
        """
        # 1. Calcular los "puntajes de atención": ¿Qué tan relevante es cada palabra (K) para la palabra actual (Q)?
        # Se multiplican las matrices de Query y Key.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. (Opcional) Aplicar una máscara.
        # La máscara se usa en el decodificador para evitar que el modelo "vea" palabras futuras
        # y en el codificador/decodificador para ignorar el padding.
        if mask is not None:
            # Rellenamos con un valor muy negativo donde la máscara es 0 para que el softmax los anule.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. Convertir los puntajes en probabilidades (pesos de atención) usando softmax.
        # Esto nos da una distribución que suma 1, indicando cuánta "atención" dar a cada palabra.
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. Ponderar los valores (V) con las probabilidades de atención.
        # Las palabras más relevantes (con mayor attn_probs) tendrán un mayor impacto en el resultado.
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        Divide la última dimensión en (num_heads, d_k). [1]
        Transforma la forma de [batch, seq_len, d_model] a [batch, num_heads, seq_len, d_k].
        Esto permite que cada cabeza realice el cálculo de atención de forma independiente.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combina las cabezas de atención de vuelta a la dimensión d_model. [1]
        Es la operación inversa a split_heads.
        """
        batch_size, _, seq_length, d_k = x.size()
        # .contiguous() es necesario para asegurar que el tensor esté en un bloque de memoria contiguo.
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Paso hacia adelante para la Atención Multi-Cabeza. [3, 1]
        """
        # 1. Proyectar Q, K, V usando las capas lineales.
        # 2. Dividir las proyecciones en múltiples cabezas.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 3. Calcular la atención para cada cabeza en paralelo.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Combinar los resultados de todas las cabezas y proyectarlos a la salida final.
        output = self.W_o(self.combine_heads(attn_output))
        return output

# --- 2. Red Feed-Forward Posicional ---
# Después de que la capa de atención ha recopilado información contextual de otras palabras,
# esta red neuronal simple procesa la representación de cada palabra de forma aislada.
# Ayuda a transformar y refinar la información, añadiendo capacidad no lineal al modelo.
class PositionwiseFeedForward(nn.Module):
    """
    Implementación de la red Feed-Forward posicional. [1, 4]
    Aplica dos transformaciones lineales con una activación ReLU en medio.
    """
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        # d_ff es la dimensionalidad de la capa oculta interna (suele ser 4 * d_model).
        self.fc1 = nn.Linear(d_model, d_ff) # Expande la dimensión
        self.fc2 = nn.Linear(d_ff, d_model) # Contrae la dimensión de vuelta a la original
        self.relu = nn.ReLU()

    def forward(self, x):
        # La fórmula es: FC2(ReLU(FC1(x)))
        return self.fc2(self.relu(self.fc1(x)))

# --- 3. Codificación Posicional (Positional Encoding) ---
# El Transformer, por diseño, no tiene idea del orden de las palabras.
# La codificación posicional inyecta información sobre la posición de cada token
# en la secuencia directamente en sus embeddings. Utiliza una combinación de funciones
# seno y coseno para crear un vector único para cada posición.
class PositionalEncoding(nn.Module):
    """
    Inyecta información sobre la posición relativa o absoluta de los tokens. [1, 5, 6]
    Utiliza funciones seno y coseno de diferentes frecuencias.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Creamos una matriz 'pe' (positional encoding) de tamaño [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # Creamos un vector de posiciones de 0 a max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 'div_term' calcula las diferentes frecuencias para las funciones seno y coseno.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Aplicamos seno a las dimensiones pares.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Aplicamos coseno a las dimensiones impares.
        pe[:, 1::2] = torch.cos(position * div_term)

        # 'register_buffer' guarda 'pe' como un estado persistente del modelo,
        # pero no como un parámetro a ser entrenado.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Sumamos la codificación posicional al embedding de entrada.
        # x tiene forma [batch_size, seq_len, d_model]
        # self.pe[:, :x.size(1)] selecciona las codificaciones para la longitud de la secuencia actual.
        x = x + self.pe[:, :x.size(1)]
        return x