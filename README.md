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
    "DescripcionProductoComprador": "NOTEBOOK RYZEN 7, 16GB DE RAM, SSD M.2 1TB, PANTALLA 13.3\"",
    "DescripcionProductoProveedor": "NOTEBOOK RYZEN 7, 16GB DE RAM, SSD M.2 1TB, PANTALLA 13.3\"",
    "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
  },
  "codigo_cotizacion": "377-164-COT24",
  "rut_proveedor": "76292976-7",
  "use_diccionarios": true,
  "campos_manuales": [
    { "campo": "Tipo", "contexto": "Responde SOLO una de estas opciones exactas: 'Laptop', 'Desktop', 'AIO', 'Otro'. 'Laptop' para notebooks/portátiles, 'Desktop' para PC de escritorio torres, 'AIO' para All-in-One, 'Otro' si no es ninguno de los anteriores." },
    { "campo": "Part Number", "contexto": "Extrae el número de parte del fabricante (P/N, SKU, Código de Producto). Es un código alfanumérico único que identifica el modelo (ej: '15-EH1005LA', '82KT00GMUS'). NO es el número de serie ni el Asset Tag. Si no se encuentra, responde 'No disponible'." },
    { "campo": "Modelo", "contexto": "Extrae el nombre completo del modelo del producto (ej: 'HP Pavilion 15-EH1005LA', 'Lenovo IdeaCentre AIO 3'). Si no se encuentra, responde 'No disponible'." },
    { "campo": "Procesador", "contexto": "Extrae el modelo exacto y completo. Si la ficha técnica y la descripción difieren, usa siempre la ficha." },
    { "campo": "Marca", "contexto": "Extrae solo el nombre de la marca fabricante." },
    { "campo": "Tipo RAM", "contexto": "Copia el sufijo exacto: LPDDR4X ≠ LPDDR4, LPDDR5X ≠ LPDDR5." },
    { "campo": "Tipo Almacenamiento", "contexto": "Solo el tipo, sin capacidad (ej: 'SSD', 'HDD', 'SSD/eMMC')." },
    { "campo": "Sistema Operativo", "contexto": "Nombre completo del SO. Si no está mencionado explícitamente, responde 'No disponible'." },
    { "campo": "RAM (GB)", "contexto": "Solo el número seguido de 'GB' (ej: '16 GB')." },
    { "campo": "Almacenamiento (GB)", "contexto": "Solo el número seguido de 'GB'. Convierte TB a GB si es necesario." },
    { "campo": "Pantalla (Pulgadas)", "contexto": "Solo el número en pulgadas, sin unidades (ej: '15.6')." },
    { "campo": "Nucleos", "contexto": "Responde solo el número de núcleos del procesador (ej: '8'). Si el Procesador está en el diccionario, este campo se completará automáticamente." },
    { "campo": "Hilos", "contexto": "Responde solo el número de hilos del procesador (ej: '16'). Si el Procesador está en el diccionario, este campo se completará automáticamente." }
  ],
  "diccionario_similarity_threshold": 0.85,
  "diccionario_llm_fallback": true,
  "token_bearer": "TOKEN_OBTENIDO_AL_LOGEAR"
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `payload` | objeto | — | Datos del producto (ver estructura arriba) |
| `codigo_cotizacion` | string | — | Código de la cotización (ej: `377-164-COT24`) |
| `rut_proveedor` | string | — | RUT del proveedor (ej: `76292976-7`) |
| `use_diccionarios` | bool | `false` | Normalizar campos con diccionarios usando similitud de embeddings |
| `llm_provider` | string\|null | `null` | `"openai"`, `"gemini"` o `"deepseek"`. `null` usa `DEFAULT_LLM_PROVIDER` del `.env` |
| `campos_manuales` | lista\|null | `null` | Campos adicionales a extraer. Acepta strings simples o dicts `{campo, contexto}` (ver abajo) |
| `diccionario_similarity_threshold` | float | `0.85` | Score mínimo de similitud coseno para aceptar un match del diccionario (0–1) |
| `diccionario_llm_fallback` | bool | `true` | Si no hay match automático, usar LLM para decidir entre top-3 candidatos |
| `token_bearer` | string\|null | `null` | Token Bearer de Mercado Público. Si se omite, usa el token seteado en `/set-token`; si tampoco hay, usa API pública |

**Categorías soportadas:** `Computadores`

---

### `POST /catalogar/licitacion` — Licitación

Cataloga un producto de una **licitación pública**. Los anexos técnicos se descargan desde la API pública de Mercado Público (no requiere token).

**Request body:**

```json
{
  "payload": {
    "Categoria": "Computadores",
    "DescripcionProductoComprador": "COMPUTADOR TODO EN UNO 24 PULGADAS",
    "DescripcionProductoProveedor": "AIO LENOVO IDEACENTRE 3",
    "productoname": "Computadores de escritorio"
  },
  "codigo_licitacion": "4179-44-LR22",
  "rut_proveedor": "96523180-3",
  "use_diccionarios": true,
  "campos_manuales": [
    { "campo": "Procesador" },
    "RAM (GB)",
    "Pantalla (Pulgadas)"
  ]
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `payload` | objeto | — | Datos del producto |
| `codigo_licitacion` | string | — | Código de la licitación (ej: `4179-44-LR22`) |
| `rut_proveedor` | string | — | RUT del proveedor |
| `use_diccionarios` | bool | `true` | Normalizar con diccionarios |
| `llm_provider` | string\|null | `null` | Proveedor LLM |
| `campos_manuales` | lista\|null | `null` | Campos adicionales a extraer |
| `diccionario_similarity_threshold` | float | `0.85` | Score mínimo de similitud |
| `diccionario_llm_fallback` | bool | `true` | LLM fallback para matches dudosos |

---

### `POST /set-token` — Guardar token en memoria

Guarda el token Bearer de Mercado Público en memoria del servidor. Una vez seteado, todos los llamados a `/catalogar` lo usan automáticamente si no viene en el body.

```json
{ "access_token": "eyJ...", "obtained_at": "2026-03-13T10:00:00" }
```

> El token persiste en RAM hasta que el servidor se reinicie o se llame `/set-token` nuevamente.

### `GET /get-token` — Consultar token guardado

Retorna el token actualmente en memoria.

---

## Campos manuales

`campos_manuales` acepta strings simples o dicts `{ "campo": "...", "contexto": "..." }`. El `contexto` se inyecta en el prompt del agente de ese campo como instrucción adicional, sin reemplazar las instrucciones base.

Los campos marcados con ★ tienen diccionario de valores canónicos — si `use_diccionarios: true`, el valor extraído se normaliza por similitud de embeddings contra ese diccionario.

```json
"campos_manuales": [
  {
    "campo": "Tipo",
    "contexto": "Responde SOLO una de estas opciones exactas: 'Laptop', 'Desktop', 'AIO', 'Otro'. 'Laptop' para notebooks/portátiles, 'Desktop' para PC de escritorio torres, 'AIO' para All-in-One, 'Otro' si no es ninguno de los anteriores."
  },
  {
    "campo": "Part Number",
    "contexto": "Extrae el número de parte del fabricante (P/N, SKU, Código de Producto). Es un código alfanumérico único que identifica el modelo (ej: '15-EH1005LA', '82KT00GMUS'). NO es el número de serie ni el Asset Tag. Si no se encuentra, responde 'No disponible'."
  },
  {
    "campo": "Modelo",
    "contexto": "Extrae el nombre completo del modelo del producto (ej: 'HP Pavilion 15-EH1005LA', 'Lenovo IdeaCentre AIO 3'). Si no se encuentra, responde 'No disponible'."
  },
  {
    "campo": "Procesador",
    "contexto": "Extrae el modelo exacto y completo (ej: 'Intel Core i7-1355U', 'AMD Ryzen 7 7735U'). Si la ficha técnica y la descripción del proveedor difieren, usa siempre la ficha técnica."
  },
  {
    "campo": "Marca",
    "contexto": "Extrae solo el nombre de la marca fabricante (ej: 'HP', 'Lenovo', 'Dell'). Puedes inferirla desde el nombre del modelo si no está escrita explícitamente."
  },
  {
    "campo": "Tipo RAM",
    "contexto": "Extrae solo el tipo/tecnología de RAM, SIN capacidades ni números. Copia el sufijo exacto tal como aparece en el documento: LPDDR4X ≠ LPDDR4, LPDDR5X ≠ LPDDR5. Para Apple con memoria unificada responde 'Memoria Unificada'."
  },
  {
    "campo": "Tipo Almacenamiento",
    "contexto": "Extrae solo el tipo de almacenamiento sin capacidad (ej: 'SSD', 'HDD', 'SSD/eMMC'). No incluyas GB ni TB en la respuesta."
  },
  {
    "campo": "Sistema Operativo",
    "contexto": "Extrae el nombre completo y exacto del SO (ej: 'Microsoft Windows 11 Home', 'FreeDOS', 'macOS'). No asumas el SO por la marca o modelo — si no está mencionado explícitamente, responde 'No disponible'."
  },
  {
    "campo": "RAM (GB)",
    "contexto": "Responde solo el número seguido de 'GB' (ej: '16 GB'). Si encuentras TB, conviértelo a GB."
  },
  {
    "campo": "Almacenamiento (GB)",
    "contexto": "Responde solo el número seguido de 'GB' (ej: '512 GB'). Si encuentras TB, conviértelo a GB (1 TB = 1000 GB)."
  },
  {
    "campo": "Pantalla (Pulgadas)",
    "contexto": "Responde solo el número en pulgadas, sin unidades (ej: '15.6', '13.3')."
  },
  {
    "campo": "Nucleos",
    "contexto": "Responde solo el número de núcleos del procesador (ej: '8'). Si el Procesador está en el diccionario, este campo se completará automáticamente."
  },
  {
    "campo": "Hilos",
    "contexto": "Responde solo el número de hilos del procesador (ej: '16'). Si el Procesador está en el diccionario, este campo se completará automáticamente."
  }
]
```

**Campos con diccionario de normalización (★):** `Procesador`, `Marca`, `Tipo RAM`, `Tipo Almacenamiento`, `Sistema Operativo`

> Si el Procesador tiene match en el diccionario de procesadores, `Nucleos` e `Hilos` se completan automáticamente aunque no estén en `campos_manuales`.

---

## Normalización con diccionarios (`use_diccionarios: true`)

Cuando está habilitada, después de la extracción RAG se aplica una capa de normalización por similitud de embeddings para los campos cubiertos por `diccionario_features.xlsx` (Tipo RAM, Sistema Operativo, Tipo Almacenamiento, Marca, Procesador).

**Flujo por campo:**

```
valor extraído
      ↓
embedding similarity vs. diccionario del campo (k=3)
      ↓
¿score ≥ diccionario_similarity_threshold?
   SÍ → usa valor normalizado del diccionario
   NO → ¿diccionario_llm_fallback = true?
           SÍ → LLM evalúa los top-3 candidatos → normaliza o mantiene original
           NO → mantiene el valor original
```

Adicionalmente, si el campo `Procesador` tiene match en el diccionario, los campos `Nucleos` e `Hilos` se completan automáticamente desde `diccionario_procesador.xlsx`.

---

## Response

Todos los endpoints de catalogación devuelven:

```json
{
  "success": true,
  "resultado": {
    "ROWNUM": "abc12345",
    "Tipo": "Laptop",
    "Part Number": "15-EH1005LA",
    "Modelo": "HP Pavilion 15-EH1005LA",
    "Procesador": "AMD Ryzen 7 5700U",
    "RAM (GB)": "16 GB",
    "Tipo RAM": "LPDDR4X",
    "Almacenamiento (GB)": "512 GB",
    "Tipo Almacenamiento": "SSD NVMe",
    "Pantalla (Pulgadas)": "15.6",
    "Marca": "HP",
    "Nucleos": "8",
    "Hilos": "16"
  },
  "errores": [],
  "warnings": [],
  "metadata": {
    "codigo_cotizacion": "377-164-COT24",
    "rut_proveedor": "76292976-7",
    "categoria": "Computadores",
    "adjuntos_descargados": true,
    "adjuntos_procesados": true,
    "use_diccionarios": true,
    "llm_provider": "openai"
  }
}
```

El campo `ROWNUM` siempre está presente en `resultado`. El resto de los atributos depende de los `campos_manuales` enviados en el request.

---

## Diferencias entre endpoints

| | `/catalogar` (Compra Ágil) | `/catalogar/licitacion` |
|---|---|---|
| Tipo de proceso | Solicitud de cotización | Licitación pública |
| Código requerido | `codigo_cotizacion` | `codigo_licitacion` |
| Autenticación Mercado Público | Opcional (`token_bearer` o `/set-token`) | No requerida |
| `use_diccionarios` default | `false` | `true` |
| Adjuntos | Archivos del proveedor (ofertas) | Anexos técnicos y económicos |

---

## Configuración del `.env`

Ver `.env.example` para la lista completa. Las más relevantes:

```env
# Proveedor LLM por defecto
DEFAULT_LLM_PROVIDER=openai          # openai | gemini | deepseek

# Claves API
OPENAI_API_KEY=sk-...                # Siempre requerida (embeddings)
GOOGLE_API_KEY=AIza...               # Solo si DEFAULT_LLM_PROVIDER=gemini
DEEPSEEK_API_KEY=...                 # Solo si DEFAULT_LLM_PROVIDER=deepseek

# Modelos LLM
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.5-flash
DEEPSEEK_MODEL=deepseek-chat

# Seguridad
API_KEY=cambia-esto
```
