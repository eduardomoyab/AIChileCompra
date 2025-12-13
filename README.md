# ChileCompra GenAI - Sistema de Catalogación Automática

Sistema de catalogación automática de productos en licitaciones públicas usando LLM, RAG (Retrieval Augmented Generation) y LangGraph.

## Características

- **Extracción automática de atributos** de productos usando LLM con contexto de documentos adjuntos
- **RAG (Retrieval Augmented Generation)** con FAISS para búsqueda semántica en adjuntos
- **Normalización con diccionarios** de valores estándar
- **Extracción paralela de campos personalizados** con agentes especializados
- **Workflow orquestado con LangGraph** para flujos complejos
- **Multi-LLM support**: OpenAI, Google Gemini, DeepSeek
- **API REST con FastAPI** para integración fácil
- **Sesión persistente** con Mercado Público para descargas optimizadas

## Arquitectura

### Workflow de Catalogación

```
START
  ↓
descargar_adjuntos (Descarga archivos desde Mercado Público)
  ↓
procesar_adjuntos (Extracción de texto de PDFs, DOCX, XLSX, etc.)
  ↓
rag_adjuntos (Extracción de atributos base con RAG)
  ↓
┌─────────────────────┐
│ ¿campos_manuales?   │ (Si el usuario especificó campos adicionales)
└─────────────────────┘
  ↓ Sí               ↓ No
campos_manuales      │
(N agentes en paralelo)
  ↓                  ↓
┌──────────────┐
│ ¿usar_dic?   │ (Si usar normalización con diccionarios)
└──────────────┘
  ↓ Sí    ↓ No
rag_diccionarios  consolidar_resultado
  ↓
consolidar_resultado (Merge de todos los resultados)
  ↓
END
```

### Componentes Principales

- **LangGraph**: Orquestación del workflow con estados y transiciones
- **FAISS**: Vector store en memoria para búsqueda semántica
- **LangChain**: Framework para integración con LLMs
- **FastAPI**: API REST con validación automática
- **Selenium**: Automatización de descarga de adjuntos

## Instalación

### Requisitos

- Python 3.10+
- Firefox + GeckoDriver (para descarga de adjuntos)

### 1. Clonar repositorio

```bash
git clone <repository-url>
cd AIChileCompra
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Copiar `.env.example` a `.env` y configurar:

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```bash
# Proveedor de LLM por defecto (openai, gemini, deepseek)
DEFAULT_LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_MODEL=gpt-4o-mini

# Google Gemini Configuration
GOOGLE_API_KEY=tu_google_api_key_aqui
GEMINI_MODEL=gemini-2.5-flash

# ChileCompra - Clave Única Credentials
CU_USER=tu_rut_aqui
CU_PASSWORD=tu_password_aqui
GECKO_DRIVER_PATH=C:/SeleniumDrivers/geckodriver.exe

# API Configuration
API_KEY=tu-api-key-segura
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Iniciar servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8000`

- **Documentación interactiva**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Uso

### Endpoint Principal: `/catalogar`

#### Request

**Headers:**
```
X-API-Key: tu-api-key-configurada
Content-Type: application/json
```

**Body:**
```json
{
  "payload": {
    "Categoria": "Computadores",
    "DescripcionProductoComprador": "NOTEBOOK RYZEN 7, 16GB DE RAM (INTEGRADA), ALMACENAMIENTO SSD M.2 DE 1TB, PANTALLA ANTIREFLECTANTE CON MICROBORDE DE 33.8CM (13.3)",
    "DescripcionProductoProveedor": "NOTEBOOK RYZEN 7, 16GB DE RAM (INTEGRADA), ALMACENAMIENTO SSD M.2 DE 1TB, PANTALLA ANTIREFLECTANTE CON MICROBORDE DE 33.8CM (13.3)",
    "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
  },
  "codigo_cotizacion": "377-164-COT24",
  "rut_proveedor": "76292976-7",
  "use_diccionarios": false,
  "campos_manuales": [
    "Pantalla (Pulgadas)",
    "Procesador",
    "Marca",
    "Tipo RAM",
    "RAM (GB)",
    "Tipo Almacenamiento",
    "Almacenamiento (GB)",
    "Hilos",
    "Nucleos"
  ]
}
```

#### Parámetros

- **payload** (required): Información del producto
  - `Categoria`: Categoría del producto (actualmente solo "Computadores")
  - `DescripcionProductoComprador`: Descripción del comprador
  - `DescripcionProductoProveedor`: Descripción del proveedor
  - `productoname`: Nombre específico del producto

- **codigo_cotizacion** (required): Código de la solicitud de cotización
- **rut_proveedor** (required): RUT del proveedor
- **use_diccionarios** (optional, default: true): Si usar normalización con diccionarios
- **llm_provider** (optional): Proveedor LLM específico ('openai', 'gemini', 'deepseek')
- **campos_manuales** (optional): Lista de campos adicionales a extraer en paralelo

#### Response

```json
{
  "success": true,
  "resultado": {
    "Tipo": "Laptop",
    "Part Number": "15-eh3027la",
    "Modelo": "HP 255 G9",
    "campos_manuales": {
      "Pantalla (Pulgadas)": "13.3",
      "Procesador": "AMD Ryzen 7 7735U",
      "Marca": "HP",
      "Tipo RAM": "DDR5",
      "RAM (GB)": "16 GB",
      "Tipo Almacenamiento": "SSD M.2",
      "Almacenamiento (GB)": "1000 GB",
      "Hilos": "16",
      "Nucleos": "8"
    }
  },
  "errores": [],
  "warnings": [],
  "metadata": {
    "codigo_cotizacion": "377-164-COT24",
    "rut_proveedor": "76292976-7",
    "categoria": "Computadores",
    "adjuntos_descargados": true,
    "adjuntos_procesados": true,
    "use_diccionarios": false,
    "llm_provider": "openai"
  }
}
```

### Ejemplo con cURL

```bash
curl -X POST "http://localhost:8000/catalogar" \
  -H "X-API-Key: 1234" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "Categoria": "Computadores",
      "DescripcionProductoComprador": "NOTEBOOK RYZEN 7, 16GB DE RAM",
      "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION",
      "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
    },
    "codigo_cotizacion": "377-164-COT24",
    "rut_proveedor": "76292976-7",
    "campos_manuales": ["Procesador", "RAM (GB)", "Pantalla (Pulgadas)"]
  }'
```

### Ejemplo con Python

```python
import requests

url = "http://localhost:8000/catalogar"
headers = {
    "X-API-Key": "1234",
    "Content-Type": "application/json"
}
data = {
    "payload": {
        "Categoria": "Computadores",
        "DescripcionProductoComprador": "NOTEBOOK RYZEN 7, 16GB DE RAM",
        "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION",
        "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
    },
    "codigo_cotizacion": "377-164-COT24",
    "rut_proveedor": "76292976-7",
    "campos_manuales": ["Procesador", "RAM (GB)", "Pantalla (Pulgadas)"]
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## Campos Manuales - Sistema Genérico

El sistema soporta **cualquier campo personalizado** que el usuario defina. Las reglas de extracción y limpieza se aplican automáticamente según patrones detectados en el nombre del campo:

### Patrones Soportados

| Patrón del Campo | Comportamiento | Ejemplos |
|-----------------|----------------|----------|
| `Campo (GB)`, `Campo (TB)` | Extrae número + unidad | "RAM (GB)" → "16 GB" |
| `Campo (Pulgadas)`, `Campo (Inches)` | Solo número | "Pantalla (Pulgadas)" → "13.3" |
| `Tipo Campo` | Solo tipo, elimina capacidades | "Tipo RAM" → "DDR5" |
| `Nucleos`, `Hilos`, `Cores` | Solo número | "Nucleos" → "8" |
| `Marca`, `Modelo` | Limpia capacidades numéricas | "Marca" → "HP" |
| `Procesador`, `CPU` | Modelo completo | "Procesador" → "AMD Ryzen 7 7735U" |
| `Campo (MHz)`, `Campo (GHz)` | Número + unidad | "Frecuencia (GHz)" → "3.5 GHz" |

### Ejemplos de Campos Personalizados

```json
{
  "campos_manuales": [
    "Pantalla (Pulgadas)",      // → "13.3"
    "Procesador",               // → "AMD Ryzen 7 7735U"
    "Marca",                    // → "HP"
    "Tipo RAM",                 // → "DDR5"
    "RAM (GB)",                 // → "16 GB"
    "Almacenamiento (GB)",      // → "1000 GB"
    "Frecuencia (GHz)",         // → "3.5 GHz"
    "Puertos USB",              // → "4"
    "Garantía (Años)",          // → "3 Años"
    "Color",                    // → "Negro"
    "Tipo Conexión"             // → "Wi-Fi 6"
  ]
}
```

## Configuración Avanzada

### Variables de Entorno - Campos Manuales

```bash
# Extracción Paralela - Configuración de Concurrencia
CAMPOS_MANUALES_MAX_WORKERS=3        # Máximo de agentes en paralelo
CAMPOS_MANUALES_MAX_RETRIES=3        # Reintentos por campo
CAMPOS_MANUALES_INITIAL_DELAY=2.0    # Delay inicial entre reintentos (segundos)
```

### Variables de Entorno - Vector Store

```bash
# FAISS Configuration
VECTORSTORE_PERSIST_DIR=./faiss_db
VECTORSTORE_COLLECTION_NAME=catalogacion_default
EMBEDDING_MODEL=models/text-embedding-004
```

### Variables de Entorno - Retrieval

```bash
# Configuración de búsqueda semántica
SEARCH_K_ADJUNTOS=5          # Documentos a recuperar de adjuntos
SEARCH_K_DICCIONARIOS=2      # Documentos a recuperar de diccionarios

# Temperaturas de LLM
TEMPERATURE_ADJUNTOS=0.7     # Creatividad en extracción de adjuntos
TEMPERATURE_DICCIONARIOS=0.3 # Precisión en normalización
```

## Estructura del Proyecto

```
AIChileCompra/
├── agents/                      # Agentes y workflows
│   ├── catalogacion_comp.py     # Agente de catalogación base
│   ├── grafo_comp.py            # Definición del grafo LangGraph
│   ├── nodos_comp.py            # Nodos del workflow
│   ├── state_comp.py            # Estado compartido
│   ├── get_agent.py             # Factory de LLMs
│   └── get_vectorstore.py       # Factory de FAISS
├── utils/                       # Utilidades
│   ├── get_attachments.py       # Descarga de adjuntos
│   └── process_attachments.py   # Procesamiento de archivos
├── main.py                      # API FastAPI
├── extraer_atributos.py         # Función principal de extracción
├── requirements.txt             # Dependencias
├── .env.example                 # Plantilla de configuración
└── README.md                    # Este archivo
```

## Manejo de Errores

### Rate Limiting

El sistema incluye **retry automático con backoff exponencial** para manejar límites de API:

- Máximo 3 reintentos por campo (configurable)
- Delay inicial de 2 segundos, duplicándose en cada reintento
- Detección inteligente del tiempo de espera desde mensajes de error
- Staggered submission de 500ms entre campos

