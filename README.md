# AIChileCompra — API de Catalogación Automática

API REST para extracción automática de atributos de productos en compras públicas chilenas, usando LLM, RAG y LangGraph.

---

## Inicio rápido

### 1. Configurar el entorno

Copia `.env.example` a `.env` y edita las claves necesarias:

```bash
cp .env.example .env
```

Campos mínimos a cambiar:

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | API key de OpenAI — requerida siempre (usada para embeddings) |
| `API_KEY` | Clave para autenticar las llamadas a esta API |
| `GOOGLE_API_KEY` | Solo si usas `gemini` como proveedor LLM |
| `DEEPSEEK_API_KEY` | Solo si usas `deepseek` como proveedor LLM |

El proveedor por defecto se controla con `DEFAULT_LLM_PROVIDER` (`openai`, `gemini` o `deepseek`).

> **Nota:** `OPENAI_API_KEY` es siempre necesaria aunque uses Gemini o DeepSeek como LLM, porque los embeddings FAISS usan `text-embedding-3-small` de OpenAI.

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Iniciar la API

```bash
python main.py
```

La API queda disponible en `http://localhost:8000`.
Documentación interactiva: `http://localhost:8000/docs`

---

## Autenticación

Todos los endpoints requieren el header `x-api-key` con el valor configurado en `API_KEY` del `.env`.

```bash
curl -H "x-api-key: tu-clave" ...
```

---

## Endpoints

### `POST /catalogar` — Compra Ágil

Cataloga un producto de una **solicitud de cotización** (Compra Ágil). Descarga los adjuntos del proveedor desde Mercado Público y extrae atributos usando RAG.

**Request body:**

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
  "use_diccionarios": true,
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
  ],
  "token_bearer" : "TOKEN OBTENIDO AL LOGEAR"
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `payload` | objeto | — | Datos del producto (ver estructura arriba) |
| `codigo_cotizacion` | string | — | Código de la cotización (ej: `1058094-1307-COT23`) |
| `rut_proveedor` | string | — | RUT del proveedor (ej: `76.274.027-3`) |
| `use_diccionarios` | bool | `false` | Normalizar procesador/RAM/almacenamiento con diccionarios |
| `llm_provider` | string\|null | `null` | `"openai"`, `"gemini"` o `"deepseek"`. `null` usa `DEFAULT_LLM_PROVIDER` del `.env` |
| `campos_manuales` | lista\|null | `null` | Campos adicionales a extraer (ej: `["Pantalla (Pulgadas)", "Procesador"]`). `null` = solo atributos fijos |
| `token_bearer` | string\|null | `null` | Token Bearer de Mercado Público. Si se provee, usa la API autenticada; si no, usa el Buscador público |

**Categorías soportadas:** `Computadores`

**Ejemplo:**

```bash
curl -X POST http://localhost:8000/catalogar \
  -H "x-api-key: tu-clave" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "Categoria": "Computadores",
      "DescripcionProductoComprador": "NOTEBOOK 512GB SSD 8GB RAM PROCESADOR AMD RYZEN 7",
      "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION 15-EH1005LA",
      "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
    },
    "codigo_cotizacion": "1058094-1307-COT23",
    "rut_proveedor": "76.274.027-3",
    "use_diccionarios": false,
    "campos_manuales": ["Procesador", "RAM (GB)", "Pantalla (Pulgadas)"]
  }'
```

---

### `POST /catalogar/licitacion` — Licitación

Cataloga un producto de una **licitación pública**. Los anexos técnicos y económicos se descargan desde la API pública de Mercado Público (no requiere token).

**Request body:**

```json
{
  "payload": {
    "Categoria": "Computadores",
    "DescripcionProductoComprador": "SERVICIO DE ARRIENDO DE COMPUTADORES DESDE 4179-44-LR22ARRIENDO 30 COMPUTADORES ,CUOTA 29 DE 36",
    "DescripcionProductoProveedor": "SERVICIO DE ARRIENDO DE COMPUTADORES DESDE 4179-44-LR22ARRIENDO 30 COMPUTADORES CUOTA 29 DE 36FACTURA DEBE INDICAR EL NUMERO DE ORDEN DE COMPRA PARA SER ACEPTADA.",
    "productoname": "Computadores de escritorio"
  },
  "codigo_licitacion": "4179-44-LR22",
  "rut_proveedor": "96.523.180-3",
  "use_diccionarios": true,
  "llm_provider": "openai",
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

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `payload` | objeto | — | Datos del producto |
| `codigo_licitacion` | string | — | Código de la licitación (ej: `1234-567-LE24`) |
| `rut_proveedor` | string | — | RUT del proveedor |
| `use_diccionarios` | bool | `true` | Normalizar con diccionarios |
| `llm_provider` | string\|null | `null` | Proveedor LLM. `null` usa `DEFAULT_LLM_PROVIDER` del `.env` |
| `campos_manuales` | lista\|null | `null` | Campos adicionales a extraer |

**Ejemplo:**

```bash
curl -X POST http://localhost:8000/catalogar/licitacion \
  -H "x-api-key: tu-clave" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "Categoria": "Computadores",
      "DescripcionProductoComprador": "COMPUTADOR TODO EN UNO 24 PULGADAS",
      "DescripcionProductoProveedor": "AIO LENOVO IDEACENTRE 3",
      "productoname": "Computadores de escritorio"
    },
    "codigo_licitacion": "1234-567-LE24",
    "rut_proveedor": "76.123.456-7",
    "use_diccionarios": true,
    "campos_manuales": ["Procesador", "RAM (GB)", "Pantalla (Pulgadas)"]
  }'
```

---

### `POST /catalogar/texto` — Texto Directo (sin adjuntos)

Cataloga un producto directamente desde texto, **sin descargar adjuntos ni usar RAG**. El nombre, descripción y/o tabla de atributos del producto se pasan directo al LLM como contexto.

Útil para catalogación masiva desde bases de datos existentes, normalización de catálogos de proveedores, o pruebas sin acceso a la API de Mercado Público.

**Request body:**

```json
{
  "nombre_producto": "Laptop HP Pavilion 15-EH1005LA",
  "descripcion": "Notebook HP con procesador AMD Ryzen 7, 8GB RAM, 512GB SSD NVMe, pantalla 15.6\"",
  "atributos": {
    "Procesador": "AMD Ryzen 7 5700U",
    "RAM": "8 GB DDR4",
    "Almacenamiento": "512 GB SSD M.2 NVMe",
    "Pantalla": "15.6 pulgadas FHD",
    "Sistema Operativo": "Windows 11 Home"
  },
  "categoria": "Computadores",
  "use_diccionarios": true,
  "llm_provider": "gemini",
  "campos_manuales": ["Procesador", "RAM (GB)", "Tipo RAM", "Almacenamiento (GB)", "Pantalla (Pulgadas)"]
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `nombre_producto` | string | — | Nombre del producto (**requerido**) |
| `descripcion` | string\|null | `null` | Descripción libre del producto |
| `atributos` | objeto\|null | `null` | Tabla de especificaciones técnicas en formato `{atributo: valor}` |
| `categoria` | string | `"Computadores"` | Categoría del producto. Solo `"Computadores"` soportado |
| `use_diccionarios` | bool | `true` | Normalizar con diccionarios técnicos |
| `llm_provider` | string\|null | `null` | `"openai"`, `"gemini"` o `"deepseek"`. `null` usa `DEFAULT_LLM_PROVIDER` del `.env` |
| `campos_manuales` | lista\|null | `null` | Campos adicionales a extraer (ej: `["Pantalla (Pulgadas)", "Procesador"]`). `null` = solo atributos fijos |

**Categorías soportadas:** `Computadores`

**Ejemplo:**

```bash
curl -X POST http://localhost:8000/catalogar/texto \
  -H "x-api-key: tu-clave" \
  -H "Content-Type: application/json" \
  -d '{
    "nombre_producto": "Laptop HP Pavilion 15-EH1005LA",
    "descripcion": "Notebook HP con procesador AMD Ryzen 7, 8GB RAM, 512GB SSD NVMe",
    "atributos": {
      "Procesador": "AMD Ryzen 7 5700U",
      "RAM": "8 GB DDR4",
      "Almacenamiento": "512 GB SSD M.2 NVMe"
    },
    "use_diccionarios": true,
    "campos_manuales": ["Procesador", "RAM (GB)", "Pantalla (Pulgadas)"]
  }'
```

---

## Diferencias entre endpoints

| | `/catalogar` (Compra Ágil) | `/catalogar/licitacion` | `/catalogar/texto` |
|---|---|---|---|
| Tipo de proceso | Solicitud de cotización | Licitación pública | Texto directo |
| Código requerido | `codigo_cotizacion` | `codigo_licitacion` | Ninguno |
| Autenticación Mercado Público | Opcional (`token_bearer`) | No requerida | No requerida |
| `use_diccionarios` default | `false` | `true` | `true` |
| Adjuntos | Archivos del proveedor (ofertas) | Anexos técnicos y económicos | No usa adjuntos |
| RAG / embeddings | Sí | Sí | No (texto directo al LLM) |

---

## Response

Los tres endpoints devuelven la misma estructura:

```json
{
  "success": true,
  "resultado": {
    "ROWNUM": "1",
    "Tipo": "Laptop",
    "Part Number": "15-EH1005LA",
    "Modelo": "HP Pavilion 15-EH1005LA",
    "Pantalla (Pulgadas)": "15.6",
    "Procesador": "AMD Ryzen 7 5700U",
    "RAM (GB)": "8 GB"
  },
  "errores": [],
  "warnings": [],
  "metadata": {
    "codigo_cotizacion": "1058094-1307-COT23",
    "rut_proveedor": "76.274.027-3",
    "categoria": "Computadores",
    "adjuntos_descargados": true,
    "adjuntos_procesados": true,
    "use_diccionarios": false,
    "llm_provider": "openai"
  }
}
```

Los atributos fijos siempre presentes en `resultado` son:

| Atributo | Valores posibles |
|---|---|
| `Tipo` | `"Laptop"`, `"Desktop"`, `"AIO"`, `"Otro"` |
| `Part Number` | Código del fabricante o `"No disponible"` |
| `Modelo` | Nombre completo del modelo o `"No disponible"` |

Los campos en `campos_manuales` se agregan directamente en `resultado` junto a los anteriores.

---

## Campos manuales

El sistema acepta **cualquier campo personalizado** definido en `campos_manuales`. Algunos ejemplos:

```json
"campos_manuales": [
  "Pantalla (Pulgadas)",
  "Procesador",
  "Marca",
  "Tipo RAM",
  "RAM (GB)",
  "Almacenamiento (GB)",
  "Nucleos",
  "Hilos",
  "Color"
]
```

---

## Configuración del `.env`

Ver `.env.example` para la lista completa de variables. Las más relevantes:

```env
# Proveedor LLM por defecto
DEFAULT_LLM_PROVIDER=openai          # openai | gemini | deepseek

# Claves API — cambiar según el proveedor usado
OPENAI_API_KEY=sk-...                # Siempre requerida (embeddings)
GOOGLE_API_KEY=AIza...               # Solo si DEFAULT_LLM_PROVIDER=gemini
DEEPSEEK_API_KEY=...                 # Solo si DEFAULT_LLM_PROVIDER=deepseek

# Modelos LLM (opcionales, tienen valores por defecto razonables)
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.5-flash
DEEPSEEK_MODEL=deepseek-chat

# Seguridad de esta API
API_KEY=cambia-esto
```
