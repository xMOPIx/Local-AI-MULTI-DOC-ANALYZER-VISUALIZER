# Usamos Python 3.12 como dice tu README
FROM python:3.12-slim

WORKDIR /app

# Instalamos dependencias del sistema necesarias para Matplotlib/NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos el puerto que usa Streamlit por defecto
EXPOSE 8501

# Comando para arrancar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]