# Local AI Multi-Doc Analyzer & Visualizer 🤖📊

Este proyecto es un sistema avanzado de **RAG (Retrieval-Augmented Generation)** diseñado para el análisis y visualización de múltiples documentos de forma totalmente local, priorizando la privacidad de los datos y la eficiencia arquitectónica.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-orange)](https://ollama.ai/)

## 🚀 Descripción del Proyecto
La herramienta permite cargar diversos tipos de documentos (PDF, texto, etc.), procesarlos mediante técnicas de **NLP** y realizar consultas en lenguaje natural. A diferencia de las soluciones basadas en la nube, este analizador utiliza modelos locales a través de **Ollama**, garantizando que la información sensible nunca salga de la infraestructura del usuario.

### Características Principales:
*   **Procesamiento Multi-Documento:** Ingesta y segmentación inteligente de archivos.
*   **Arquitectura RAG:** Implementación de embeddings locales para una recuperación de información precisa.
*   **Interfaz de Usuario:** Dashboard interactivo desarrollado en **Streamlit** para visualización de datos y chat.
*   **Despliegue Contenerizado:** Configuración lista para producción mediante **Docker**, asegurando la portabilidad y consistencia del entorno.

## 🛠️ Stack Tecnológico
*   **Lenguaje:** Python 3.x
*   **IA/LLM:** Ollama (Llama 3 / Mistral) para inferencia local.
*   **Orquestación:** Docker & Docker Compose.
*   **Frontend:** Streamlit.
*   **Procesamiento de Datos:** LangChain / LlamaIndex (dependiendo de tu implementación específica).

## 📦 Instalación y Uso

### Requisitos Previos
*   Tener instalado [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/).
*   [Ollama](https://ollama.ai/) ejecutándose en el host o configurado como contenedor.

### Despliegue con Docker
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/xMOPIx/Local-AI-MULTI-DOC-ANALYZER-VISUALIZER.git](https://github.com/xMOPIx/Local-AI-MULTI-DOC-ANALYZER-VISUALIZER.git)
   cd Local-AI-MULTI-DOC-ANALYZER-VISUALIZER