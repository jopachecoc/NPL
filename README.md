# NPL

Mini-proyecto de procesamiento de texto utilizando técnicas clásicas

# Integrantes

Jonathan Pacheco

Joshua Triana

Julio Morales

# Proyecto Final crear tu propio ChatGPT usando un RAG

# 📰 Retrieval-Augmented Generation (RAG) con Ollama y Langchain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/jopachecoc/NPL/blob/main/PF_ChatGPT_RAG.ipynb) PF_ChatGPT_RAG.ipynb

---

# PF_ChatGPT_RAG: Retrieval-Augmented Generation con Ollama y Langchain

Este notebook implementa un sistema de RAG (Retrieval-Augmented Generation) usando modelos pre-entrenados, Ollama y LangChain, sobre un corpus de noticias en español. El objetivo es crear un chatbot capaz de responder preguntas utilizando información recuperada de los documentos.

## Ejecución en Google Colab

Puedes ejecutar el notebook directamente en Google Colab usando el siguiente enlace:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jopachecoc/NPL/blob/main/PF_ChatGPT_RAG.ipynb)

## Requisitos

- GPU recomendada (T4 o superior)
- El notebook instala automáticamente las dependencias necesarias (Ollama, LangChain, FAISS, Gradio, etc.)

## Pasos principales del notebook

1. Instalación de dependencias y configuración de entorno
2. Carga y exploración del dataset de noticias en español
3. Indexación de documentos con FAISS y generación de embeddings
4. Configuración de Ollama y LangChain para el modelo LLM
5. Implementación de la cadena de recuperación y generación de respuestas
6. Interfaz conversacional con Gradio

## Notas

- No se requiere entrenamiento de modelos, solo se usan modelos pre-entrenados.
- El corpus utilizado puede ser modificado, pero debe mantener una estructura similar.
- El historial de conversación se maneja tanto en LangChain como en la interfaz de usuario.

## Referencias

- [LangChain](https://www.langchain.com)
- [Ollama](https://ollama.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://gradio.app)

## Evidencias de ejecución

A continuación se muestran ejemplos de la ejecución del notebook `PF_ChatGPT_RAG.ipynb` en Google Colab, donde se observa el funcionamiento del chatbot con RAG y LangChain:

### Ejemplo 1: Inicio y carga del sistema

![Evidencia 1: Inicio y carga del sistema](IMG1ChB1.jpg)

---

### Ejemplo 2: Pregunta al chatbot y recuperación de contexto

![Evidencia 2: Pregunta al chatbot y recuperación de contexto](IMG1ChB2.jpg)

---

### Ejemplo 3: Respuesta generada y referencias a documentos

![Evidencia 3: Respuesta generada y referencias a documentos](IMG1ChB3.jpg)

---

## Estas imágenes muestran el flujo completo: desde la carga del sistema, la interacción con el chatbot, hasta la generación de respuestas con referencias a los documentos recuperados del corpus.

# ENTREGA_3: Clasificación de Texto en Español con BERT y Hugging Face

En esta entrega se implementa un clasificador de texto en español utilizando modelos pre-entrenados tipo BERT y la librería Hugging Face Transformers. El objetivo es especializar un modelo BERT en tareas de clasificación de sentimientos y noticias, aprovechando transfer learning y fine-tuning.

## Flujo de trabajo

1. **Carga y análisis exploratorio de datasets**

   - Uso de datasets de Hugging Face como `pysentimiento/spanish-tweets` y `tweet_eval`.
   - Estadísticas de longitud y distribución de clases.

2. **Preprocesamiento y tokenización**

   - Uso del tokenizador oficial del modelo BERT en español ([dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)).
   - Conversión de textos a secuencias de tokens y máscaras de atención.

3. **Preparación de datos**

   - División en conjuntos de entrenamiento, validación y prueba.
   - Conversión de etiquetas a IDs.

4. **Definición y ajuste del modelo**

   - Uso de `AutoModelForSequenceClassification` de Hugging Face.
   - Entrenamiento como featurizer (congelando BERT) y fine-tuning (entrenando todo el modelo).
   - Experimentación con diferentes cabezas de clasificación.

5. **Entrenamiento y evaluación**
   - Entrenamiento con la clase `Trainer` de Hugging Face.
   - Monitoreo con TensorBoard.
   - Evaluación de precisión y análisis de errores.

## Resultados

- Precisión superior al 90% en el conjunto de prueba usando fine-tuning.
- El uso de modelos pre-entrenados reduce el tiempo de entrenamiento y mejora la calidad respecto a modelos desde cero.

## Datasets utilizados

- pysentimiento/spanish-tweets'

## Ejecución

Puedes ejecutar el notebook [tercera_entrega.ipynb](tercera_entrega.ipynb) localmente o en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jopachecoc/NPL/blob/main/tercera_entrega.ipynb)

# ENTREGA 2: Clasificación de Noticias en Español con LSTM

Este proyecto implementa un clasificador de noticias en español utilizando una red neuronal LSTM (Long Short-Term Memory) con PyTorch Lightning. El objetivo es categorizar titulares de noticias en cinco categorías: Deportes, Salud, Tecnología, Colombia y Economía.

## Descripción

Se utiliza un dataset de noticias reales del portal RCN, disponible en [Hugging Face Hub](https://huggingface.co/datasets/Nicky0007/titulos_noticias_rcn_clasificadas). El flujo de trabajo incluye:

1. **Carga y análisis exploratorio del dataset**

   - Estadísticas de longitud de los textos.
   - Distribución de clases.

2. **Preprocesamiento y tokenización**

   - Tokenizador simple y construcción de vocabulario.
   - Conversión de textos a secuencias de tokens.

3. **Preparación de datos**

   - División en conjuntos de entrenamiento, validación y prueba.
   - Balanceo de clases.

4. **Definición del modelo**

   - Implementación de un bloque LSTM y una capa densa para clasificación.
   - Entrenamiento con PyTorch Lightning.

5. **Entrenamiento y evaluación**

   - Ajuste de hiperparámetros.
   - Monitoreo con TensorBoard.
   - Evaluación de precisión global y por clase.

6. **Resultados**
   - Precisión general superior al 90%.
   - Análisis de errores y recomendaciones para mejorar la clase "Colombia".

## Requisitos

- Python 3.10+
- PyTorch
- PyTorch Lightning
- datasets
- matplotlib
- scikit-learn
- numpy
- pandas

Instala las dependencias con:

```sh
pip install -r requerimientos.txt
```

## Ejecución

Puedes ejecutar el notebook [segundo_entrega.ipynb](segundo_entrega.ipynb) localmente o en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtar10/icesi-nlp/blob/main/Sesion2/2-nlp-with-lstm.ipynb)

## Notas

- El modelo utiliza una longitud máxima de 35 tokens por texto, optimizada según la distribución del dataset.
- Se emplea el estado oculto final de la LSTM como representación del texto.
- El código está preparado para facilitar la experimentación y el ajuste de hiperparámetros.

## Resultados

- Precisión global: ~90%
- Precisión por clase: superior al 91% en la mayoría de categorías, con margen de mejora en "Colombia".

## Aplicaciones

- Clasificación automática de noticias en portales de contenido.
- Organización y personalización de la experiencia del usuario.

---
