# Sistema de Catalogación Automática - ChileCompra GenAI

Sistema automatizado para extracción y catalogación de atributos de productos en licitaciones públicas usando LLM, RAG y LangGraph.

## 🎯 Características

- **API REST con FastAPI** - Endpoints listos para producción
- **Autenticación con API Key** - Seguridad mediante headers
- **Descarga automática** de adjuntos desde Mercado Público
- **Procesamiento de archivos** (PDF, DOCX, XLSX, imágenes) con OCR
- **Extracción de atributos** usando RAG con LLM
- **Normalización** con diccionarios técnicos
- **Completado inteligente** de especificaciones
- **Flujo orquestado** con LangGraph
- **Múltiples LLMs** (OpenAI, Google Gemini, DeepSeek)
- **Docker ready** - Fácil despliegue con Docker y Docker Compose

## 📦 Categorías Soportadas

### ✅ Computadores (Disponible)
- Desktops
- Notebooks
- All in One (AIO)

**Atributos extraídos:**
- Modalidad (Compra/Arriendo)
- Tipo Producto
- Marca
- Procesador
- Núcleos
- Hilos Procesador
- RAM (GB)
- Tipo RAM
- Sistema Operativo
- Tipo Almacenamiento
- Almacenamiento (GB)
- Pantalla (Pulgadas)

### 🔜 Próximamente
- Medicamentos
- Otros productos

## 🚀 Inicio Rápido

### Opción 1: Docker (Recomendado)

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 2. Construir imagen
docker build -t catalogacion-api .

# 3. Ejecutar contenedor
docker run -d -p 8000:8000 --env-file .env --name catalogacion-api catalogacion-api

# 4. Ver logs
docker logs -f catalogacion-api

# Detener: docker stop catalogacion-api
# Reiniciar: docker restart catalogacion-api
```

La API estará disponible en: **http://localhost:8000**
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Opción 2: Instalación Local

#### 1. Clonar repositorio

```bash
git clone <repo-url>
cd CatalogacionEndpoint
```

#### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```bash
# API Key
API_KEY=tu-api-key-secreta

# LLM Provider
DEFAULT_LLM_PROVIDER=gemini

# Google API
GOOGLE_API_KEY=tu_api_key_aqui

# Mercado Público
CU_USER=tu_usuario_clave_unica
CU_PASSWORD=tu_password_clave_unica
GECKO_DRIVER_PATH=C:/SeleniumDrivers/geckodriver.exe
```

#### 5. Instalar Geckodriver

1. Descargar: https://github.com/mozilla/geckodriver/releases
2. Extraer a la ruta especificada en `GECKO_DRIVER_PATH`

#### 6. Crear diccionarios

```bash
# Crear vector stores de diccionarios
python agents/create_vector_dic_comp.py create
python agents/create_vector_dic_med.py create
```

#### 7. Iniciar API

```bash
python main.py
```

## 🔌 Uso de la API

### Endpoints Principales

#### POST /catalogar
Cataloga un producto individual.

**Request:**
```bash
curl -X POST http://localhost:8000/catalogar \
  -H "X-API-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "ROWNUM": "1",
      "Categoria": "Computadores",
      "DescripcionProductoComprador": "NOTEBOOK 512GB SSD 8GB RAM",
      "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION",
      "productoname": "Computadores"
    },
    "codigo_cotizacion": "12345678",
    "rut_proveedor": "76123456-7",
    "use_diccionarios": true,
    "llm_provider": "gemini"
  }'
```

**Response:**
```json
{
  "success": true,
  "resultado": {
    "ROWNUM": "1",
    "Modalidad": "Compra",
    "Tipo Producto": "Notebook",
    "Marca": "HP",
    "Procesador": "Intel Core i5-1135G7",
    "Núcleos": "4",
    "Hilos Procesador": "8",
    "RAM (GB)": "8",
    "Tipo RAM": "DDR4",
    "Sistema Operativo": "Windows 11 Home",
    "Tipo Almacenamiento": "SSD",
    "Almacenamiento (GB)": "512",
    "Pantalla (Pulgadas)": "15.6"
  },
  "errores": [],
  "warnings": [],
  "metadata": { ... }
}
```

#### POST /catalogar/lote
Cataloga múltiples productos.

**Request:**
```bash
curl -X POST http://localhost:8000/catalogar/lote \
  -H "X-API-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "payloads": [
      {
        "ROWNUM": "1",
        "Categoria": "Computadores",
        ...
      },
      {
        "ROWNUM": "2",
        "Categoria": "Computadores",
        ...
      }
    ],
    "codigo_cotizacion": "12345678",
    "rut_proveedor": "76123456-7"
  }'
```

### Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Uso con Python

```python
import requests

headers = {"X-API-Key": "tu-api-key"}

response = requests.post(
    "http://localhost:8000/catalogar",
    headers=headers,
    json={
        "payload": {
            "ROWNUM": "1",
            "Categoria": "Computadores",
            "DescripcionProductoComprador": "NOTEBOOK HP 8GB",
            "DescripcionProductoProveedor": "HP Pavilion",
            "productoname": "Computadores"
        },
        "codigo_cotizacion": "12345678",
        "rut_proveedor": "76123456-7"
    }
)

resultado = response.json()
print(resultado['resultado'])
```

## 🐳 Docker

### Comandos Básicos

```bash
# Construir imagen
docker build -t catalogacion-api .

# Ejecutar (simple)
docker run -d -p 8000:8000 --env-file .env --name catalogacion-api catalogacion-api

# Ejecutar (con volúmenes para persistencia)
docker run -d -p 8000:8000 --env-file .env \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/diccionarios:/app/diccionarios:ro \
  -v $(pwd)/attachments:/app/attachments \
  --name catalogacion-api catalogacion-api

# Ver logs
docker logs -f catalogacion-api

# Detener
docker stop catalogacion-api

# Iniciar (si está detenido)
docker start catalogacion-api

# Reiniciar
docker restart catalogacion-api

# Eliminar contenedor
docker rm catalogacion-api

# Eliminar imagen
docker rmi catalogacion-api
```

### Actualizar la API

```bash
# 1. Detener y eliminar contenedor
docker stop catalogacion-api && docker rm catalogacion-api

# 2. Reconstruir imagen
docker build -t catalogacion-api .

# 3. Ejecutar nuevo contenedor
docker run -d -p 8000:8000 --env-file .env --name catalogacion-api catalogacion-api
```

## 📖 Uso con CLI

### Comando Básico

```bash
python extraer_atributos.py \
  --payload payload_ejemplo.json \
  --codigo 12345678 \
  --rut 76123456-7
```

### Opciones

```bash
# Ver ayuda
python extraer_atributos.py --help

# Procesar lote
python extraer_atributos.py \
  --payload payloads_lote_ejemplo.json \
  --codigo 12345678 \
  --rut 76123456-7 \
  --lote

# Guardar resultado
python extraer_atributos.py \
  --payload payload_ejemplo.json \
  --codigo 12345678 \
  --rut 76123456-7 \
  --output resultado.json

# Usar modelo específico
python extraer_atributos.py \
  --payload payload_ejemplo.json \
  --codigo 12345678 \
  --rut 76123456-7 \
  --llm openai
```

## ⚙️ Configuración

### Variables de Entorno (.env)

```bash
# API Configuration
API_KEY=tu-api-key-secreta
API_HOST=0.0.0.0
API_PORT=8000

# LLM Provider
DEFAULT_LLM_PROVIDER=gemini

# API Keys
GOOGLE_API_KEY=tu_google_api_key
OPENAI_API_KEY=tu_openai_api_key
DEEPSEEK_API_KEY=tu_deepseek_api_key

# Mercado Público
CU_USER=tu_usuario_clave_unica
CU_PASSWORD=tu_password_clave_unica
GECKO_DRIVER_PATH=/usr/local/bin/geckodriver

# Temperaturas LLM
TEMPERATURE_ADJUNTOS=0.7
TEMPERATURE_DICCIONARIOS=0.3

# Retrieval
SEARCH_K_ADJUNTOS=5
SEARCH_K_DICCIONARIOS=2

# Chunks
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────┐
│           FastAPI REST API (main.py)            │
│              (Autenticación con API Key)        │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │   extraer_atributos  │
         │   (Orchestrator)     │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │   LangGraph Workflow │
         │   (grafo_comp.py)    │
         └───────────┬──────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌────────┐    ┌──────────┐    ┌──────────┐
│Download│    │ Process  │    │   RAG    │
│Adjuntos│───▶│ Adjuntos │───▶│ Extract  │
└────────┘    └──────────┘    └─────┬────┘
                                    │
                              ┌─────▼─────┐
                              │    RAG    │
                              │ Normalize │
                              └─────┬─────┘
                                    │
                              ┌─────▼─────┐
                              │ Complete  │
                              │   Specs   │
                              └───────────┘
```

### Componentes

- **main.py** - API REST con FastAPI
- **extraer_atributos.py** - CLI y orquestador
- **agents/grafo_comp.py** - Workflow con LangGraph
- **agents/nodos_comp.py** - Nodos del grafo
- **agents/catalogacion_comp.py** - Lógica de catalogación
- **agents/retriever.py** - Retriever de adjuntos
- **agents/retriever_diccionario_comp.py** - Retriever de diccionarios
- **utils/** - Utilidades (descarga, procesamiento)

## 🧪 Testing

### Test de la API

```bash
# Ejecutar suite de tests
python test_api.py
```

### Test rápido

```bash
# Sin archivos JSON
python test_quick.py
```

### Test de componentes individuales

```bash
# Retriever de adjuntos
python agents/retriever.py 12345678 76123456-7 "procesador intel" 5

# Retriever de diccionarios
python agents/retriever_diccionario_comp.py "intel core i7" 2

# Catalogación completa
python agents/catalogacion_comp.py
```

## 🐛 Troubleshooting

### Docker

**Error: "No space left on device"**
```bash
# Limpiar imágenes no usadas
docker system prune -a
```

**Error: "Cannot connect to Docker daemon"**
```bash
# Iniciar Docker Desktop (Windows/Mac)
# O iniciar servicio (Linux)
sudo systemctl start docker
```

### API

**Error: "API Key inválida"**
- Verificar que el header `X-API-Key` coincide con `.env`

**Error de conexión**
- Verificar que el servidor esté corriendo: `docker-compose ps`

### Catalogación

**Error: "No se encontró el vector store"**
```bash
# Crear vector stores
python agents/create_vector_dic_comp.py create
```

**Error en descarga de adjuntos**
- Verificar credenciales de Clave Única en `.env`
- Verificar que Geckodriver esté instalado

## 📊 Monitoreo

### Logs con Docker

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Ver últimas 100 líneas
docker-compose logs --tail=100

# Logs de API específicamente
docker-compose logs -f catalogacion-api
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🔒 Seguridad

### Producción

1. **Cambiar API Key**: Usar clave fuerte y rotarla periódicamente
2. **HTTPS**: Usar certificados SSL/TLS
3. **CORS**: Especificar orígenes permitidos en `main.py`
4. **Rate Limiting**: Implementar límites de requests
5. **Secrets**: Usar Docker secrets o variables de entorno seguras

## 📝 Formato de Datos

### Payload de Entrada

```json
{
  "ROWNUM": "1",
  "Categoria": "Computadores",
  "DescripcionProductoComprador": "NOTEBOOK HP 8GB RAM",
  "DescripcionProductoProveedor": "Notebook HP Pavilion",
  "productoname": "Computadores"
}
```

### Resultado de Salida

```json
{
  "ROWNUM": "1",
  "Modalidad": "Compra",
  "Tipo Producto": "Notebook",
  "Marca": "HP",
  "Procesador": "Intel Core i5-1135G7",
  "Núcleos": "4",
  "Hilos Procesador": "8",
  "RAM (GB)": "8",
  "Tipo RAM": "DDR4",
  "Sistema Operativo": "Windows 11 Home",
  "Tipo Almacenamiento": "SSD",
  "Almacenamiento (GB)": "512",
  "Pantalla (Pulgadas)": "15.6"
}
```

## 🤝 Contribuciones

[Instrucciones de contribución]

## 📧 Contacto

[Tu información de contacto]

## 📄 Licencia

[Tu licencia aquí]
