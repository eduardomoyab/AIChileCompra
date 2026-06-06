"""
API REST para Sistema de Catalogación Automática - ChileCompra GenAI

FastAPI endpoint para extracción y catalogación de atributos de productos
en licitaciones públicas usando LLM, RAG y LangGraph.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraer_atributos import extraer_atributos, extraer_atributos_lote, validar_payload
from extraer_atributos_licitaciones import extraer_atributos_licitacion
from agents.grafo import ejecutar_catalogacion_texto
from utils.get_attachments import TokenAttachmentDownloader
from utils.langsmith_utils import set_langsmith

set_langsmith()

# Cargar variables de entorno
load_dotenv()

# Token persistido: PostgreSQL si DATABASE_URL está configurado, archivo como fallback local
_token_dir = os.getenv("TOKEN_STORE_PATH") or os.getenv("ATTACHMENTS_OUTPUT_PATH", "attachments")
_TOKEN_FILE = os.path.join(_token_dir, ".token_store.json")
_DB_URL = os.getenv("DATABASE_URL", "")

_TOKEN_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS token_store (
    id INTEGER PRIMARY KEY DEFAULT 1,
    access_token TEXT NOT NULL,
    obtained_at TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
)
"""

# Cache en memoria: evita ir a la DB en cada request
_token_cache: dict = {}
_token_cache_ts: float = 0.0
_TOKEN_CACHE_TTL = int(os.getenv("TOKEN_CACHE_TTL_SECONDS", "300"))  # 5 min por defecto

def _read_token(bypass_cache: bool = False) -> dict:
    global _token_cache, _token_cache_ts
    if not bypass_cache and time.time() - _token_cache_ts < _TOKEN_CACHE_TTL and _token_cache:
        return _token_cache
    if _DB_URL:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(_DB_URL)
            with engine.connect() as conn:
                conn.execute(text(_TOKEN_TABLE_DDL))
                row = conn.execute(text("SELECT access_token, obtained_at FROM token_store WHERE id = 1")).fetchone()
            if row:
                _token_cache = {"access_token": row[0], "obtained_at": row[1]}
                _token_cache_ts = time.time()
                return _token_cache
            return {}
        except Exception as e:
            logging.warning(f"[token] Error leyendo de DB: {e} — usando archivo")
    try:
        with open(_TOKEN_FILE, "r") as f:
            _token_cache = json.load(f)
            _token_cache_ts = time.time()
            return _token_cache
    except Exception:
        return {}


#set_langsmith()
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Crear aplicación FastAPI
app = FastAPI(
    title="ChileCompra GenAI - Catalogación API",
    description="API para extracción automática de atributos de productos en licitaciones públicas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys desde .env (soporta múltiples separadas por coma)
_raw_keys = os.getenv("API_KEY", "your-secret-api-key-here")
API_KEYS = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# Categorías soportadas desde .env (separadas por coma)
_raw_cats = os.getenv("CATEGORIAS_SOPORTADAS", "Computadores,Medicamentos")
CATEGORIAS_SOPORTADAS = [c.strip() for c in _raw_cats.split(",") if c.strip()]

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return x_api_key

# ========== SECURITY ==========

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Verifica que el API key sea válido.

    Args:
        x_api_key: API key desde el header X-API-Key

    Raises:
        HTTPException: Si el API key es inválido

    Example:
        curl -H "X-API-Key: your-api-key" http://localhost:8000/catalogar
    """
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return x_api_key


# ========== MODELS ==========

class ProductoPayload(BaseModel):
    """Modelo del payload para un producto individual"""

    Categoria: str = Field(..., description="Categoría del producto (Computadores, Medicamentos, etc.)")
    DescripcionProductoComprador: str = Field(..., description="Descripción del producto por el comprador")
    DescripcionProductoProveedor: str = Field(..., description="Descripción del producto por el proveedor")
    productoname: str = Field(..., description="Nombre específico del producto (diferente a Categoría)")

    @validator('Categoria')
    def validar_categoria(cls, v):
        """Valida que la categoría sea soportada"""
        if v not in CATEGORIAS_SOPORTADAS:
            raise ValueError(
                f"Categoría '{v}' no soportada. "
                f"Categorías disponibles: {CATEGORIAS_SOPORTADAS}"
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "Categoria": "Computadores",
                "DescripcionProductoComprador": "NOTEBOOK 512GB SSD 8GB RAM PROCESADOR AMD RYZEN 7",
                "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION 15-EH1005LA",
                "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
            }
        }


class CampoManualItem(BaseModel):
    """Campo manual con instrucciones opcionales para el agente de extracción"""
    campo: str = Field(..., description="Nombre del campo a extraer (ej: 'principio_activo_1')")
    contexto: str = Field("", description="Instrucciones adicionales para el agente de este campo")
    diccionario: Optional[str] = Field(None, description="Nombre del diccionario a usar para normalización. Si no se provee, se usa el nombre del campo.")


class CatalogarRequest(BaseModel):
    """Modelo de request para catalogación de un producto (Compra Ágil)"""

    payload: ProductoPayload = Field(..., description="Datos del producto a catalogar")
    codigo_cotizacion: str = Field(..., description="Código de la solicitud de cotización")
    rut_proveedor: str = Field(..., description="RUT del proveedor")
    use_diccionarios: bool = Field(False, description="Si usar normalización con diccionarios")
    llm_provider: Optional[str] = Field(None, description="Proveedor de LLM (openai, gemini, deepseek). None = usa DEFAULT_LLM_PROVIDER del .env")
    campos_manuales: Optional[List[Union[str, CampoManualItem]]] = Field(None, description="Lista de campos a extraer. Puede ser strings simples o dicts con {campo, contexto}")
    diccionario_similarity_threshold: float = Field(0.85, description="Score mínimo de similitud coseno para aceptar match del diccionario (0-1)")
    diccionario_llm_fallback: bool = Field(True, description="Si no hay match sobre el threshold, usar LLM para decidir entre top-3 candidatos")
    token_bearer: Optional[str] = Field(None, description="Token Bearer de Mercado Público. Si se provee, usa la API autenticada (servicios-compra-agil) en lugar del Buscador público.")
    clasificacion_prompt: Optional[str] = Field(None, description="Descripción de la categoría buscada para filtrar accesorios. Ej: 'Computadores portátiles y de escritorio. No incluye monitores, teclados, mouse ni bolsos.' Si no se provee, no se filtra.")


class CatalogarLicitacionRequest(BaseModel):
    """Modelo de request para catalogación de un producto de Licitación"""

    payload: ProductoPayload = Field(..., description="Datos del producto a catalogar")
    codigo_licitacion: str = Field(..., description="Código de la licitación (ej: 1234-567-LE24)")
    rut_proveedor: str = Field(..., description="RUT del proveedor")
    use_diccionarios: bool = Field(True, description="Si usar normalización con diccionarios")
    llm_provider: Optional[str] = Field(None, description="Proveedor de LLM (openai, gemini, deepseek). None = usa DEFAULT_LLM_PROVIDER del .env")
    campos_manuales: Optional[List[Union[str, CampoManualItem]]] = Field(None, description="Lista de campos a extraer. Puede ser strings simples o dicts con {campo, contexto}")
    diccionario_similarity_threshold: float = Field(0.85, description="Score mínimo de similitud coseno para aceptar match del diccionario (0-1)")
    diccionario_llm_fallback: bool = Field(True, description="Si no hay match sobre el threshold, usar LLM para decidir entre top-3 candidatos")
    clasificacion_prompt: Optional[str] = Field(None, description="Descripción de la categoría buscada para filtrar accesorios. Ej: 'Computadores portátiles y de escritorio. No incluye monitores, teclados, mouse ni bolsos.' Si no se provee, no se filtra.")

    @validator('llm_provider')
    def validar_llm_provider(cls, v):
        """Valida el proveedor de LLM"""
        if v is not None:
            providers_validos = ['openai', 'gemini', 'deepseek']
            if v not in providers_validos:
                raise ValueError(
                    f"LLM provider '{v}' no válido. "
                    f"Opciones: {providers_validos}"
                )
        return v

    class Config:
        schema_extra = {
            "example": {
                "payload": {
                    "Categoria": "Computadores",
                    "DescripcionProductoComprador": "NOTEBOOK 512GB SSD 8GB RAM",
                    "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION",
                    "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
                },
                "codigo_licitacion": "1234-567-LE24",
                "rut_proveedor": "76123456-7",
                "use_diccionarios": True,
                "llm_provider": "gemini"
            }
        }




class CatalogarTextoRequest(BaseModel):
    """Modelo de request para catalogación desde texto directo (sin adjuntos)"""

    nombre_producto: str = Field(..., description="Nombre del producto")
    descripcion: Optional[str] = Field(None, description="Descripción libre del producto")
    atributos: Optional[Dict[str, Any]] = Field(None, description="Especificaciones técnicas {campo: valor}")
    categoria: str = Field("Computadores", description="Categoría del producto. Solo 'Computadores' por ahora")
    use_diccionarios: bool = Field(True, description="Si normalizar con diccionarios técnicos")
    llm_provider: Optional[str] = Field(None, description="Proveedor de LLM (openai, gemini, deepseek). None = usa DEFAULT_LLM_PROVIDER del .env")
    campos_manuales: Optional[List[Union[str, CampoManualItem]]] = Field(None, description="Campos a extraer. Strings simples o dicts {campo, contexto}")
    diccionario_similarity_threshold: float = Field(0.85, description="Score mínimo de similitud coseno para aceptar match del diccionario (0-1)")
    diccionario_llm_fallback: bool = Field(True, description="Si no hay match sobre el threshold, usar LLM para decidir entre top-3 candidatos")

    @validator('categoria')
    def validar_categoria(cls, v):
        if v not in CATEGORIAS_SOPORTADAS:
            raise ValueError(f"Categoría '{v}' no soportada. Categorías disponibles: {CATEGORIAS_SOPORTADAS}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "nombre_producto": "Notebook HP Pavilion 15-EH1005LA",
                "descripcion": "Notebook con AMD Ryzen 7 5700U, 16GB RAM LPDDR4X, 512GB SSD NVMe",
                "atributos": {"Procesador": "AMD Ryzen 7 5700U", "RAM": "16 GB LPDDR4X", "Almacenamiento": "512 GB SSD NVMe"},
                "categoria": "Computadores",
                "use_diccionarios": True,
                "campos_manuales": ["Procesador", "Marca", "RAM (GB)", "Tipo RAM", "Almacenamiento (GB)"]
            }
        }


class CatalogarLoteRequest(BaseModel):
    """Modelo de request para catalogación de múltiples productos"""

    payloads: List[ProductoPayload] = Field(..., description="Lista de productos a catalogar")
    codigo_cotizacion: str = Field(..., description="Código de la solicitud de cotización")
    rut_proveedor: str = Field(..., description="RUT del proveedor")
    use_diccionarios: bool = Field(True, description="Si usar normalización con diccionarios")
    llm_provider: Optional[str] = Field(None, description="Proveedor de LLM")

    @validator('payloads')
    def validar_payloads_no_vacio(cls, v):
        """Valida que haya al menos un producto"""
        if len(v) == 0:
            raise ValueError("La lista de payloads no puede estar vacía")
        return v

    class Config:
        schema_extra = {
            "example": {
                "payloads": [
                    {
                        "ROWNUM": "1",
                        "Categoria": "Computadores",
                        "DescripcionProductoComprador": "NOTEBOOK 512GB SSD 8GB RAM",
                        "DescripcionProductoProveedor": "NOTEBOOK HP",
                        "productoname": "Computadores"
                    },
                    {
                        "ROWNUM": "2",
                        "Categoria": "Computadores",
                        "DescripcionProductoComprador": "DESKTOP INTEL CORE I7 16GB",
                        "DescripcionProductoProveedor": "DESKTOP LENOVO",
                        "productoname": "Computadores"
                    }
                ],
                "codigo_cotizacion": "12345678",
                "rut_proveedor": "76123456-7",
                "use_diccionarios": True,
                "llm_provider": "gemini"
            }
        }


class CatalogarResponse(BaseModel):
    """Modelo de response para catalogación"""

    success: bool = Field(..., description="Si la catalogación fue exitosa")
    resultado: Optional[Dict[str, Any]] = Field(None, description="Resultado de la catalogación")
    errores: List[str] = Field(default_factory=list, description="Lista de errores")
    warnings: List[str] = Field(default_factory=list, description="Lista de warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class HealthResponse(BaseModel):
    """Modelo de response para health check"""

    status: str = Field(..., description="Estado del servicio")
    timestamp: str = Field(..., description="Timestamp del check")
    version: str = Field(..., description="Versión de la API")


# ========== ENDPOINTS ==========

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz - información básica de la API"""
    return {
        "message": "ChileCompra GenAI - API de Catalogación",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/catalogar", response_model=CatalogarResponse)
async def catalogar_producto(request: CatalogarRequest,
                             api_key: str = Depends(require_api_key)):
    """
    Cataloga un producto individual.

    Realiza el flujo completo:
    1. Descarga adjuntos desde Mercado Público
    2. Procesa adjuntos (extracción de texto)
    3. Extrae atributos usando RAG con adjuntos
    4. Normaliza con diccionarios (opcional)
    5. Completa especificaciones

    **Requiere header:** `X-API-Key`

    **Returns:**
    - `success`: Si la catalogación fue exitosa
    - `resultado`: Diccionario con todos los atributos extraídos
    - `errores`: Lista de errores encontrados
    - `warnings`: Lista de advertencias
    - `metadata`: Información adicional del proceso
    """
    try:
        logging.info(f"Catalogando producto - Cotización: {request.codigo_cotizacion}, Proveedor: {request.rut_proveedor}")

        # Seleccionar downloader: payload > token en disco > sin token
        token = request.token_bearer or _read_token().get("access_token")
        downloader = None
        if token:
            downloader = TokenAttachmentDownloader(token)
            fuente = "payload" if request.token_bearer else "db"
            logging.info(f"Usando TokenAttachmentDownloader (token desde {fuente})")
        else:
            logging.info("Usando BuscadorAttachmentDownloader (API pública)")

        # Normalizar campos_manuales: List[str | CampoManualItem] → List[Dict]
        campos_manuales_norm = []
        for item in (request.campos_manuales or []):
            if isinstance(item, str):
                campos_manuales_norm.append({"campo": item, "contexto": ""})
            else:
                d = {"campo": item.campo, "contexto": item.contexto}
                if item.diccionario:
                    d["diccionario"] = item.diccionario
                campos_manuales_norm.append(d)

        # Ejecutar catalogación en thread pool para no bloquear el event loop
        resultado_completo = await run_in_threadpool(
            extraer_atributos,
            payload=request.payload.dict(),
            codigo_cotizacion=request.codigo_cotizacion,
            rut_proveedor=request.rut_proveedor,
            use_diccionarios=request.use_diccionarios,
            llm_provider=request.llm_provider,
            downloader=downloader,
            campos_manuales_lista=campos_manuales_norm,
            diccionario_similarity_threshold=request.diccionario_similarity_threshold,
            diccionario_llm_fallback=request.diccionario_llm_fallback,
            clasificacion_prompt=request.clasificacion_prompt,
        )

        # Preparar response
        response = CatalogarResponse(
            success=resultado_completo.get('resultado_final') is not None,
            resultado=resultado_completo.get('resultado_final'),
            errores=resultado_completo.get('errores', []),
            warnings=resultado_completo.get('warnings', []),
            metadata={
                "codigo_cotizacion": request.codigo_cotizacion,
                "rut_proveedor": request.rut_proveedor,
                "categoria": request.payload.Categoria,
                "adjuntos_descargados": resultado_completo.get('adjuntos_descargados', False),
                "adjuntos_procesados": resultado_completo.get('adjuntos_procesados', False),
                "use_diccionarios": request.use_diccionarios,
                "llm_provider": request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini"),
                "tiempos": resultado_completo.get('tiempos', []),
            }
        )

        logging.info(f"Catalogación completada - Cotización: {request.codigo_cotizacion}, Success: {response.success}")

        return response

    except ValueError as e:
        logging.error(f"Error de validación: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logging.exception(f"Error catalogando producto: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )


@app.post("/catalogar/licitacion", response_model=CatalogarResponse)
async def catalogar_licitacion(request: CatalogarLicitacionRequest,
                                api_key: str = Depends(require_api_key)):
    """
    Cataloga un producto de LICITACIÓN.

    Realiza el flujo completo:
    1. Descarga adjuntos desde Mercado Público (anexos técnicos y económicos)
    2. Procesa adjuntos (extracción de texto con OCR si es necesario)
    3. Extrae atributos usando un LLM único con RAG sobre adjuntos
    4. Normaliza con diccionarios (opcional)

    **Requiere header:** `X-API-Key`

    **Diferencias con /catalogar (Compra Ágil):**
    - No requiere autenticación (licitaciones son públicas)
    - Usa un único LLM en lugar de sistema de agentes
    - Descarga anexos técnicos y económicos

    **Returns:**
    - `success`: Si la catalogación fue exitosa
    - `resultado`: Diccionario con todos los atributos extraídos
    - `errores`: Lista de errores encontrados
    - `warnings`: Lista de advertencias
    - `metadata`: Información adicional del proceso
    """
    try:
        logging.info(f"Catalogando licitación - Código: {request.codigo_licitacion}, Proveedor: {request.rut_proveedor}")

        # Normalizar campos_manuales: List[str | CampoManualItem] → List[Dict]
        campos_manuales_norm = []
        for item in (request.campos_manuales or []):
            if isinstance(item, str):
                campos_manuales_norm.append({"campo": item, "contexto": ""})
            else:
                d = {"campo": item.campo, "contexto": item.contexto}
                if item.diccionario:
                    d["diccionario"] = item.diccionario
                campos_manuales_norm.append(d)

        # Ejecutar catalogación de licitación en thread pool para no bloquear el event loop
        resultado_completo = await run_in_threadpool(
            extraer_atributos_licitacion,
            payload=request.payload.dict(),
            codigo_licitacion=request.codigo_licitacion,
            rut_proveedor=request.rut_proveedor,
            use_diccionarios=request.use_diccionarios,
            llm_provider=request.llm_provider,
            campos_manuales=campos_manuales_norm,
            diccionario_similarity_threshold=request.diccionario_similarity_threshold,
            diccionario_llm_fallback=request.diccionario_llm_fallback,
            clasificacion_prompt=request.clasificacion_prompt,
        )

        # Preparar response
        response = CatalogarResponse(
            success=resultado_completo.get('resultado_final') is not None,
            resultado=resultado_completo.get('resultado_final'),
            errores=resultado_completo.get('errores', []),
            warnings=resultado_completo.get('warnings', []),
            metadata={
                "codigo_licitacion": request.codigo_licitacion,
                "rut_proveedor": request.rut_proveedor,
                "categoria": request.payload.Categoria,
                "adjuntos_descargados": resultado_completo.get('adjuntos_descargados', False),
                "adjuntos_procesados": resultado_completo.get('adjuntos_procesados', False),
                "use_diccionarios": request.use_diccionarios,
                "llm_provider": request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini"),
                "tiempos": resultado_completo.get('tiempos', []),
                "tipo": "licitacion"
            }
        )

        logging.info(f"Catalogación licitación completada - Código: {request.codigo_licitacion}, Success: {response.success}")

        return response

    except ValueError as e:
        logging.error(f"Error de validación: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logging.exception(f"Error catalogando licitación: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )


@app.post("/catalogar/texto", response_model=CatalogarResponse)
async def catalogar_texto(request: CatalogarTextoRequest,
                          api_key: str = Depends(require_api_key)):
    """
    Cataloga un producto a partir de texto directo, sin descargar adjuntos.

    Recibe el nombre del producto, una descripción libre y/o una tabla de
    especificaciones técnicas, y extrae atributos usando el mismo LLM y
    diccionarios de normalización que `/catalogar`.

    **Diferencias con /catalogar:**
    - No requiere `codigo_cotizacion` ni `rut_proveedor`
    - No descarga ni procesa archivos de Mercado Público
    - El texto de entrada reemplaza al contenido de los adjuntos como contexto RAG

    **Requiere header:** `X-API-Key`
    """
    try:
        logging.info(f"Catalogando desde texto — producto: {request.nombre_producto[:60]}")

        # Normalizar campos_manuales: List[str | CampoManualItem] → List[Dict]
        campos_manuales_norm = []
        for item in (request.campos_manuales or []):
            if isinstance(item, str):
                campos_manuales_norm.append({"campo": item, "contexto": ""})
            else:
                d = {"campo": item.campo, "contexto": item.contexto}
                if item.diccionario:
                    d["diccionario"] = item.diccionario
                campos_manuales_norm.append(d)

        resultado_completo = await run_in_threadpool(
            ejecutar_catalogacion_texto,
            nombre_producto=request.nombre_producto,
            descripcion=request.descripcion,
            atributos=request.atributos,
            categoria=request.categoria,
            campos_manuales_lista=campos_manuales_norm,
            use_diccionarios=request.use_diccionarios,
            llm_provider=request.llm_provider,
            diccionario_similarity_threshold=request.diccionario_similarity_threshold,
            diccionario_llm_fallback=request.diccionario_llm_fallback,
        )

        response = CatalogarResponse(
            success=resultado_completo.get('resultado_final') is not None,
            resultado=resultado_completo.get('resultado_final'),
            errores=resultado_completo.get('errores', []),
            warnings=resultado_completo.get('warnings', []),
            metadata={
                "nombre_producto": request.nombre_producto,
                "categoria": request.categoria,
                "use_diccionarios": request.use_diccionarios,
                "llm_provider": request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini"),
                "modo": "texto",
                "tiempos": resultado_completo.get('tiempos', []),
            }
        )

        logging.info(f"Catalogación texto completada — success: {response.success}")
        return response

    except ValueError as e:
        logging.error(f"Error de validación: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logging.exception(f"Error catalogando desde texto: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )


@app.get("/get-token")
async def get_token(api_key: str = Depends(require_api_key)):
    data = _read_token(bypass_cache=True)
    if not data:
        raise HTTPException(status_code=404, detail="Token no disponible")
    return data



# @app.post("/catalogar/lote", response_model=List[CatalogarResponse], dependencies=[Depends(verify_api_key)])
# async def catalogar_lote(request: CatalogarLoteRequest):
#     """
#     Cataloga un lote de productos.
#
#     Procesa múltiples productos secuencialmente, cada uno con el flujo completo
#     de catalogación.
#
#     **Requiere header:** `X-API-Key`
#
#     **Returns:**
#     Lista de resultados, uno por cada producto en el lote.
#     """
#     try:
#         logging.info(f"Catalogando lote de {len(request.payloads)} productos")
#
#         # Convertir payloads a dict
#         payloads_dict = [p.dict() for p in request.payloads]
#
#         # Ejecutar catalogación en lote
#         resultados_completos = extraer_atributos_lote(
#             payloads=payloads_dict,
#             codigo_cotizacion=request.codigo_cotizacion,
#             rut_proveedor=request.rut_proveedor,
#             use_diccionarios=request.use_diccionarios,
#             llm_provider=request.llm_provider
#         )
#
#         # Preparar responses
#         responses = []
#         for i, (resultado_completo, payload) in enumerate(zip(resultados_completos, request.payloads), 1):
#             response = CatalogarResponse(
#                 success=resultado_completo.get('resultado_final') is not None,
#                 resultado=resultado_completo.get('resultado_final'),
#                 errores=resultado_completo.get('errores', []),
#                 warnings=resultado_completo.get('warnings', []),
#                 metadata={
#                     "producto_numero": i,
#                     "categoria": payload.Categoria
#                 }
#             )
#             responses.append(response)
#
#         logging.info(f"Lote completado - {len(responses)} productos procesados")
#
#         return responses
#
#     except ValueError as e:
#         logging.error(f"Error de validación: {e}")
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#
#     except Exception as e:
#         logging.exception(f"Error catalogando lote: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error interno: {str(e)}"
#         )


# ========== STARTUP & SHUTDOWN ==========

@app.on_event("startup")
async def startup_event():
    """Evento al iniciar la aplicación"""
    logging.info("="*80)
    logging.info("ChileCompra GenAI - API de Catalogación")
    logging.info("Versión: 1.0.0")
    logging.info("="*80)
    logging.info(f"LLM Provider por defecto: {os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')}")
    logging.info(f"API Keys configuradas: {len(API_KEYS)} ({'✓' if 'your-secret-api-key-here' not in API_KEYS else '✗'})")

    # Pre-cargar FAISS diccionarios: desde disco si existen, si no construye y persiste
    try:
        from agents.retriever_diccionario import warm_dictionaries
        warm_dictionaries()
        logging.info("✓ FAISS diccionarios listos")
    except Exception as e:
        logging.warning(f"No se pudo pre-cargar FAISS diccionarios: {e}")

    logging.info("API iniciada correctamente")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento al cerrar la aplicación"""
    logging.info("API cerrándose...")


# ========== RUN ==========

if __name__ == "__main__":
    import uvicorn

    # Configuración del servidor
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))

    logging.info(f"\nIniciando servidor en http://{HOST}:{PORT}")
    logging.info(f"Documentación en http://{HOST}:{PORT}/docs\n")

    reload = os.getenv("DEV_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=reload,  # Solo en desarrollo: DEV_RELOAD=true
        log_level="info"
    )
