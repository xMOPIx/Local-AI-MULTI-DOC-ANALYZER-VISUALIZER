============================================================
           TELECOBRAIN: MULTI-DOC ANALYZER & VISUALIZER
============================================================

1. DESCRIPCIÓN DEL PROYECTO
---------------------------
TelecoBrain es un asistente inteligente basado en Inteligencia Artificial 
diseñado específicamente para el ámbito de la ingeniería de telecomunicaciones. 
Permite al usuario interactuar con múltiples archivos PDF técnicos (guías 
docentes, apuntes, hojas de datos) de forma simultánea.

La aplicación utiliza una arquitectura RAG (Retrieval-Augmented Generation) 
para extraer información precisa y es capaz de generar visualizaciones 
de datos automáticas mediante Python.

2. STACK TECNOLÓGICO
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
  semánticas ultra rápidas.
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
Para arrancar la aplicación, ejecuta el siguiente comando en la terminal:
streamlit run app.py

6. NOTAS SOBRE EL MODELO
------------------------
Este proyecto NO utiliza Fine-Tuning. Utiliza una arquitectura RAG, lo que 
permite que el modelo Llama 3.1 actúe como un "examinador a libro abierto", 
consultando la base de datos vectorial generada a partir de tus archivos PDF 
antes de generar cada respuesta.

============================================================
Desarrollado como proyecto de Ingeniería de Telecomunicaciones