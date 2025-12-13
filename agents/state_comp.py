"""
State definition for the Computadores catalogation workflow using LangGraph.

This module defines the TypedDict that represents the shared state
across all nodes in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated


class CatalogacionState(TypedDict, total=False):
    """
    State for the Computadores catalogation workflow.

    This state is shared across all nodes in the LangGraph and represents
    the complete context of a catalogation request.

    Attributes:
        # Input fields (set at the beginning)
        codigo_cotizacion (str): Código de la solicitud de cotización
        rut_proveedor (str): RUT del proveedor
        payload (dict): Producto a catalogar con estructura:
            {
                'ROWNUM': str,
                'DescripcionProductoComprador': str,
                'DescripcionProductoProveedor': str,
                'productoname': str
            }

        # Configuration
        use_diccionarios (bool): Si usar normalización con diccionarios
        llm_provider (str): Proveedor de LLM ('openai', 'gemini', 'deepseek')

        # Processing state (updated by nodes)
        adjuntos_descargados (bool): Si los adjuntos ya fueron descargados
        adjuntos_procesados (bool): Si los adjuntos ya fueron procesados
        adjuntos_path (str): Ruta donde se descargaron los adjuntos
        processed_path (str): Ruta donde se procesaron los adjuntos
        vector_store_created (bool): Si el vector store de adjuntos fue creado

        # Intermediate results
        resultado_adjuntos (dict): Resultado del RAG con adjuntos (paso 1)
        resultado_diccionarios (dict): Resultado del RAG con diccionarios (paso 2)

        # Final result
        resultado_final (dict): Resultado final de la catalogación con todos los atributos

        # Metadata & logging
        errores (list): Lista de errores encontrados durante el proceso
        warnings (list): Lista de warnings
        tiempo_total (float): Tiempo total de procesamiento en segundos
    """
    # ========== INPUT FIELDS ==========
    codigo_cotizacion: str
    rut_proveedor: str
    payload: Dict[str, Any]

    # ========== CONFIGURATION ==========
    use_diccionarios: bool
    llm_provider: Optional[str]
    downloader: Optional[Any]  # Instancia global de AttachmentDownloader (opcional)

    # ========== PROCESSING STATE ==========
    adjuntos_descargados: bool
    adjuntos_procesados: bool
    adjuntos_path: Optional[str]
    processed_path: Optional[str]
    vector_store_created: bool

    # ========== INTERMEDIATE RESULTS ==========
    resultado_adjuntos: Optional[Dict[str, Any]]
    resultado_diccionarios: Optional[Dict[str, Any]]

    # ========== FINAL RESULT ==========
    resultado_final: Optional[Dict[str, Any]]

    # ========== METADATA & LOGGING ==========
    errores: List[str]
    warnings: List[str]
    tiempo_total: Optional[float]


def create_initial_state(
    codigo_cotizacion: str,
    rut_proveedor: str,
    payload: Dict[str, Any],
    use_diccionarios: bool = True,
    llm_provider: Optional[str] = None
) -> CatalogacionState:
    """
    Crea el estado inicial para el workflow de catalogación.

    Args:
        codigo_cotizacion: Código de la solicitud de cotización
        rut_proveedor: RUT del proveedor
        payload: Información del producto a catalogar
        use_diccionarios: Si usar normalización con diccionarios
        llm_provider: Proveedor de LLM (None = usa DEFAULT_LLM_PROVIDER del .env)

    Returns:
        CatalogacionState: Estado inicial con valores por defecto

    Example:
        >>> state = create_initial_state(
        ...     "12345678",
        ...     "76123456-7",
        ...     {'ROWNUM': '1', 'DescripcionProductoComprador': '...', ...}
        ... )
    """
    return CatalogacionState(
        # Input
        codigo_cotizacion=codigo_cotizacion,
        rut_proveedor=rut_proveedor,
        payload=payload,

        # Configuration
        use_diccionarios=use_diccionarios,
        llm_provider=llm_provider,

        # Processing state (all False/None initially)
        adjuntos_descargados=False,
        adjuntos_procesados=False,
        adjuntos_path=None,
        processed_path=None,
        vector_store_created=False,

        # Results (all None initially)
        resultado_adjuntos=None,
        resultado_diccionarios=None,
        resultado_final=None,

        # Metadata
        errores=[],
        warnings=[],
        tiempo_total=None
    )


# ========== HELPER FUNCTIONS FOR STATE VALIDATION ==========

def validate_payload(payload: Dict[str, Any]) -> bool:
    """
    Valida que el payload tenga los campos requeridos.

    Args:
        payload: Payload a validar

    Returns:
        bool: True si el payload es válido
    """
    required_fields = ['ROWNUM', 'DescripcionProductoComprador', 'DescripcionProductoProveedor', 'productoname']
    return all(field in payload for field in required_fields)


def validate_state(state: CatalogacionState) -> tuple[bool, List[str]]:
    """
    Valida el estado completo y retorna errores encontrados.

    Args:
        state: Estado a validar

    Returns:
        tuple: (is_valid, list_of_errors)

    Example:
        >>> is_valid, errors = validate_state(state)
        >>> if not is_valid:
        ...     print(f"Errores: {errors}")
    """
    errors = []

    # Validar campos requeridos
    if not state.get('codigo_cotizacion'):
        errors.append("codigo_cotizacion es requerido")

    if not state.get('rut_proveedor'):
        errors.append("rut_proveedor es requerido")

    if not state.get('payload'):
        errors.append("payload es requerido")
    elif not validate_payload(state['payload']):
        errors.append("payload no tiene todos los campos requeridos")

    return len(errors) == 0, errors


def add_error(state: CatalogacionState, error: str) -> CatalogacionState:
    """
    Helper para añadir un error al estado.

    Args:
        state: Estado actual
        error: Mensaje de error

    Returns:
        CatalogacionState: Estado actualizado con el error añadido
    """
    if 'errores' not in state:
        state['errores'] = []
    state['errores'].append(error)
    return state


def add_warning(state: CatalogacionState, warning: str) -> CatalogacionState:
    """
    Helper para añadir un warning al estado.

    Args:
        state: Estado actual
        warning: Mensaje de warning

    Returns:
        CatalogacionState: Estado actualizado con el warning añadido
    """
    if 'warnings' not in state:
        state['warnings'] = []
    state['warnings'].append(warning)
    return state
