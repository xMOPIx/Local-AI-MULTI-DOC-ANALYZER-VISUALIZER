============================================================
           Local AI: MULTI-DOC ANALYZER & VISUALIZER
============================================================

1. DESCRIPCIÓN DEL PROYECTO
---------------------------
Es un asistente inteligente basado en Inteligencia Artificial 
Permite al usuario interactuar con múltiples archivos PDF técnicos a la vez.

La aplicación utiliza una arquitectura RAG (Retrieval-Augmented Generation) 
para extraer información precisa y es capaz de generar visualizaciones 
de datos automáticas mediante Python.

2. TECNOLOGIAS
--------------------
- Lenguaje: Python 3.12+
- Framework IA: LangChain
- Modelos Locales (vía Ollama): 
    * LLM: Llama 3.1
    * Embeddings: Nomic Embed Text
- Interfaz: Streamlit
- Procesamiento de Datos: NumPy, Matplotlib, PyPDF

3. CARACTERÍSTICAS TÉCNICAS
---------------------------
- Sistema RAG: Recuperación de información en tiempo real sin necesidad 
  de entrenamiento (Fine-Tuning).
- Base de Datos Vectorial: Uso de DocArray en memoria RAM para búsquedas 
  semánticas rápidas.
- Segmentación Inteligente: División de documentos en trozos (chunks) de 
  800 caracteres con 150 de solapamiento (overlap) para mantener el contexto.
- Visualización de Datos: Generación de gráficas proactiva cuando la IA 
  detecta información numérica.

4. INSTRUCCIONES DE INSTALACIÓN
-------------------------------
1. Clonar o descargar la carpeta del proyecto.
2. Crear un entorno virtual: python -m venv venv
3. Activar el entorno: .\venv\Scripts\activate
4. Instalar librerías: pip install -r requirements.txt
5. Asegurarse de tener instalado Ollama y los modelos:
   - ollama pull llama3.1
   - ollama pull nomic-embed-text

5. EJECUCIÓN
------------
streamlit run app.py

6. NOTAS SOBRE EL MODELO
------------------------
Utiliza una arquitectura RAG para que Llama 3.1 actúe como un "examinador 
a libro abierto", consultando la base de datos vectorial de tus archivos.

No emplea Fine-tuning porque el RAG permite procesar nuevos documentos al 
instante sin re-entrenar el modelo, garantiza precisión técnica al evitar 
alucinaciones y opera de forma ligera y privada en hardware local.
