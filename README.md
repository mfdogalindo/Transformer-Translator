# 🤖 Traductor Transformer: Español a Inglés 📖

## ✨ ¡Bienvenido a este proyecto educativo!

Este repositorio contiene el código para un modelo de **Traductor Automático Neuronal** que utiliza la arquitectura **Transformer**. El objetivo principal es desglosar y explicar, de una manera clara y concisa, cómo funciona esta tecnología que ha revolucionado el campo del Procesamiento del Lenguaje Natural (NLP).

El modelo ha sido entrenado para traducir oraciones del **español al inglés**. 

---

## 🏛️ La Arquitectura Transformer: Una Mirada Detallada

El modelo Transformer, introducido en el famoso paper "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", se basa en un mecanismo llamado **atención** para entender las relaciones entre palabras en una oración. A diferencia de modelos anteriores (como las RNNs), no procesa las palabras secuencialmente, lo que le permite ser mucho más rápido y eficiente.

Nuestro modelo se compone de dos partes principales: el **Codificador (Encoder)** y el **Decodificador (Decoder)**.

### 🧠 El Codificador (Encoder)

La misión del codificador es "entender" la oración en el idioma original (español). Para ello, la procesa a través de una pila de capas idénticas. Cada capa tiene dos sub-componentes clave:

1.  **Atención Multi-Cabeza (Multi-Head Attention)** 🎯
    * **¿Qué hace?** Permite que el modelo, al procesar una palabra, preste atención a todas las demás palabras de la oración.
    * **¿Por qué "Multi-Cabeza"?** En lugar de hacerlo una sola vez, realiza este cálculo de atención varias veces en paralelo (8 en nuestro caso). Cada "cabeza" se especializa en detectar diferentes tipos de relaciones (sintácticas, semánticas, etc.). Es como tener varios expertos analizando la oración desde distintos ángulos a la vez.

2.  **Red Feed-Forward (Position-wise Feed-Forward Network)** 💡
    * **¿Qué hace?** Una vez que la capa de atención ha generado una representación de la oración rica en contexto, esta red neuronal procesa cada posición de forma independiente. Ayuda a transformar y refinar la información obtenida por la atención.

### ✍️ El Decodificador (Decoder)

El decodificador tiene la tarea de generar la traducción palabra por palabra en el idioma de destino (inglés), utilizando la información que le proporciona el codificador. También tiene una pila de capas, pero cada una de ellas tiene **tres** sub-componentes:

1.  **Atención Multi-Cabeza Enmascarada (Masked Multi-Head Attention)** 🎭
    * **¿Qué hace?** Similar a la del codificador, pero con una "máscara". Al predecir la siguiente palabra de la traducción, el modelo solo puede prestar atención a las palabras que ya ha generado, no a las futuras. ¡Esto evita que haga trampa mirando la respuesta!

2.  **Atención Codificador-Decodificador (Encoder-Decoder Attention)** 🤝
    * **¿Qué hace?** ¡Aquí ocurre la magia de la traducción! Esta capa permite que el decodificador preste atención a las palabras más relevantes de la oración original en español para generar la siguiente palabra en inglés. Por ejemplo, al traducir "gato", prestará mucha atención a la palabra "cat" en la entrada.

3.  **Red Feed-Forward (Position-wise Feed-Forward Network)** 💡
    * **¿Qué hace?** Al igual que en el codificador, procesa y refina la salida de las capas de atención para prepararla para la siguiente capa o para la salida final.

### 📍 Codificación Posicional (Positional Encoding)

Dado que el Transformer no procesa las palabras en orden, ¿cómo sabe la posición de cada una? La respuesta es la **Codificación Posicional**. Antes de que los datos entren al codificador o decodificador, inyectamos una pequeña pieza de información matemática (usando funciones seno y coseno) a cada palabra para indicar su posición en la secuencia.

---

## 🚀 ¿Cómo Usar el Proyecto?

Sigue estos pasos para entrenar tu propio modelo o usar el que ya está entrenado para traducir.

### 1. Prerrequisitos

Asegúrate de tener Python instalado. Luego, instala todas las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

Esto instalará PyTorch, Spacy, y otras librerías necesarias. También se descargarán los modelos de lenguaje de Spacy para español e inglés la primera vez que se ejecute el script.

### 2. Entrenar el Modelo

Para entrenar el modelo desde cero, ejecuta el siguiente comando en tu terminal:

```bash
python main.py --mode train --epochs 10 --batch_size 128
```

* `--mode train`: Indica que queremos entrenar el modelo.
* `--epochs 10`: Número de veces que el modelo verá el dataset completo.
* `--batch_size 128`: Número de oraciones que se procesan en cada paso.

El script descargará automáticamente el dataset de traducción (un conjunto de pares de oraciones español-inglés), lo preprocesará y comenzará el entrenamiento. Los pesos del mejor modelo se guardarán en la carpeta `saved_models/` y los vocabularios en `saved_models/vocab.pkl`.

### 3. Traducir una Oración (Inferencia)

Una vez que el modelo ha sido entrenado (o si ya tienes un modelo guardado), puedes usarlo para traducir.

```bash
python main.py --mode infer --sentence "una niña está comiendo una manzana"
```

* `--mode infer`: Indica que queremos usar el modelo para traducir.
* `--sentence "..."`: La oración en español que deseas traducir.

El programa cargará el modelo y los vocabularios guardados y te mostrará la traducción en la terminal.

---

## 📂 Estructura del Proyecto

```
/
|-- 📂 model/
|   |-- transformer.py               # Define la arquitectura principal del Transformer.
|   |-- transformer_components.py    # Contiene los bloques de construcción (Atención, FF, etc.).
|-- 📂 utils/
|   |-- data_preprocessing.py        # Descarga, procesa y prepara los datos.
|-- main.py                          # Script principal para entrenar o ejecutar inferencia.
|-- engine.py                        # Contiene las funciones de entrenamiento, evaluación y traducción.
|-- requirements.txt                 # Lista de dependencias del proyecto.
|-- LICENSE                          # Licencia del proyecto (MIT).
```

---

## ⚖️ Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
