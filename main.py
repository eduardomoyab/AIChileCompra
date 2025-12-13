"""
API REST para Sistema de Catalogación Automática - ChileCompra GenAI

FastAPI endpoint para extracción y catalogación de atributos de productos
en licitaciones públicas usando LLM, RAG y LangGraph.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraer_atributos import extraer_atributos, extraer_atributos_lote, validar_payload
from utils.get_attachments import MercadoPublicoAttachmentDownloader

# Cargar variables de entorno
load_dotenv()

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

# API Key desde .env
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def require_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return x_api_key

# ========== GLOBAL SESSION ==========

# Instancia global del downloader con sesión persistente
global_downloader = None


@app.on_event("startup")
async def startup_event():
    """Inicializa la sesión global al inicio del servidor"""
    global global_downloader

    logging.info("🚀 Iniciando servidor...")
    logging.info("🔐 Iniciando sesión en ChileCompra...")

    try:
        # Crear downloader global
        global_downloader = MercadoPublicoAttachmentDownloader(headless=True)

        # Hacer login una sola vez
        global_downloader.login_mercado_publico()

        if global_downloader.token_bearer:
            logging.info("✓ Sesión iniciada correctamente")
            logging.info(f"✓ Token obtenido: {global_downloader.token_bearer[:20]}...")
        else:
            logging.error("❌ No se pudo obtener el token")

    except Exception as e:
        logging.error(f"❌ Error en login inicial: {e}")
        logging.warning("⚠️  El servidor continuará pero necesitará login en cada request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cierra la sesión al apagar el servidor"""
    global global_downloader

    if global_downloader and global_downloader.driver:
        logging.info("🔒 Cerrando sesión...")
        global_downloader.driver.quit()
        logging.info("✓ Sesión cerrada")


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
    if x_api_key != API_KEY:
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
        categorias_soportadas = ['Computadores']  # Por ahora solo Computadores
        if v not in categorias_soportadas:
            raise ValueError(
                f"Categoría '{v}' no soportada. "
                f"Categorías disponibles: {categorias_soportadas}"
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


class CatalogarRequest(BaseModel):
    """Modelo de request para catalogación de un producto"""

    payload: ProductoPayload = Field(..., description="Datos del producto a catalogar")
    codigo_cotizacion: str = Field(..., description="Código de la solicitud de cotización")
    rut_proveedor: str = Field(..., description="RUT del proveedor")
    use_diccionarios: bool = Field(True, description="Si usar normalización con diccionarios")
    llm_provider: Optional[str] = Field(None, description="Proveedor de LLM (openai, gemini, deepseek). None = usa DEFAULT_LLM_PROVIDER del .env")
    campos_manuales: Optional[List[str]] = Field(None, description="Lista de campos adicionales a extraer en paralelo (ej: ['pantalla', 'procesador'])")

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
                "codigo_cotizacion": "12345678",
                "rut_proveedor": "76123456-7",
                "use_diccionarios": True,
                "llm_provider": "gemini",
                "campos_manuales": ["pantalla", "procesador"]
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

        # Ejecutar catalogación (usar downloader global si está disponible)
        resultado_completo = extraer_atributos(
            payload=request.payload.dict(),
            codigo_cotizacion=request.codigo_cotizacion,
            rut_proveedor=request.rut_proveedor,
            use_diccionarios=request.use_diccionarios,
            llm_provider=request.llm_provider,
            downloader=global_downloader,
            campos_manuales_lista=request.campos_manuales
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
                "llm_provider": request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
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
    logging.info(f"API Key configurada: {'✓' if API_KEY != 'your-secret-api-key-here' else '✗'}")
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

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
