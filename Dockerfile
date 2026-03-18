# Dockerfile para ChileCompra GenAI - API de Catalogación
# Imagen base con Python 3.11
FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="ChileCompra GenAI - Sistema de Catalogación Automática"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Para procesamiento de PDFs
    poppler-utils \
    # Para EasyOCR / OpenCV
    libgl1 \
    libglib2.0-0 \
    # Para build de algunos paquetes Python
    gcc \
    g++ \
    # Limpieza
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelos de EasyOCR durante el build (evita descarga en runtime)
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False)" 2>&1 | tail -5

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p \
    attachments \
    processed \
    cache \
    cache/Computadores \
    cache/Medicamentos \
    cache/Adjuntos \
    cache/faiss_dict \
    chroma_db \
    logs

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando por defecto
# Para desarrollo local: python main.py (con DEV_RELOAD=true en .env)
# Para producción: gunicorn con uvicorn workers (máximo paralelismo sin mezcla de estado)
CMD ["python", "-m", "gunicorn", "main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "8", "--bind", "0.0.0.0:8000", "--timeout", "300", "--graceful-timeout", "30"]
