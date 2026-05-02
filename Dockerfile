# Se utiliza Python 3.12, pero ten en cuenta que versiones muy nuevas pueden obligar 
# a compilar librerías desde 0 si no hay precompilados disponibles (wheels).
FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema operativo (necesarias para compilar C++)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# USANDO CACHÉ: Esto hace que si vuelves a hacer build, Docker no vuelva a descargar
# todas las librerías desde internet, usando su caché interna y acelerando enormemente el proceso.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puertos: 8000 (FastAPI) y 8501 (Streamlit)
EXPOSE 8000
EXPOSE 8501
