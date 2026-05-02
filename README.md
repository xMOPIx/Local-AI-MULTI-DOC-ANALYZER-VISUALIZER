# 🤖 Local AI Multi-Doc Analyzer & Visualizer

Sistema avanzado de **RAG (Retrieval-Augmented Generation)** diseñado para el análisis profundo y la visualización de múltiples documentos de forma 100% local. Este proyecto prioriza la soberanía de los datos y la privacidad total utilizando modelos de lenguaje de última generación.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-orange)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)

## 🚀 Descripción del Proyecto

Esta herramienta transforma tus carpetas de documentos (PDF, TXT, etc.) en una base de conocimiento interactiva. Gracias a la integración con **Ollama**, el procesamiento se realiza íntegramente en tu hardware, eliminando la necesidad de enviar datos a la nube o pagar por APIs externas.

### 🌟 Características Principales
* **Privacidad Total:** Inferencia local. Tus documentos nunca salen de tu red.
* **Análisis Multi-Doc:** Sube varios archivos simultáneamente y realiza consultas cruzadas entre ellos.
* **Búsqueda Semántica:** Utiliza embeddings vectoriales para encontrar respuestas precisas basadas en el contexto.
* **Interfaz Interactiva:** Dashboard en Streamlit que permite chatear con tus datos y visualizar insights.
* **Arquitectura Robusta:** Orquestación mediante Docker para un despliegue sin errores de dependencias.

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
| :--- | :--- |
| **LLM Inferencia** | [Ollama](https://ollama.com/) (Llama 3 / Mistral) |
| **Framework RAG** | LangChain / LlamaIndex |
| **Base de Datos Vectorial** | ChromaDB / FAISS (Local) |
| **Interfaz de Usuario** | Streamlit |
| **Contenerización** | Docker & Docker Compose |

---

## 📦 Guía de Instalación y Ejecución

Sigue estos pasos para poner en marcha el sistema:

### 1. Requisitos Previos
* Tener instalado [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/).
* Tener [Ollama](https://ollama.ai/) instalado y funcionando en tu equipo.
* Descargar los modelos necesarios en tu terminal:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```

### 2. Clonar el Repositorio
Copia y pega este comando exactamente (asegúrate de no incluir corchetes):
```bash
git clone https://github.com/xMOPIx/Local-AI-MULTI-DOC-ANALYZER-VISUALIZER.git

### 3. Entrar a la carpeta

cd Local-AI-MULTI-DOC-ANALYZER-VISUALIZER

### 4. Ejecución del sistema

docker-compose up --build