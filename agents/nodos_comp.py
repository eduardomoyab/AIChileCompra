"""
Nodes for the Computadores catalogation workflow using LangGraph.

Each node is a function that takes the state and returns an updated state.
Nodes wrap existing functionality from utils and agents modules.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state_comp import CatalogacionState, add_error, add_warning
from agents.catalogacion_comp import CatalogacionComputadores
from utils.get_attachments import download_attachments_simple
from utils.process_attachments import process_attachments_simple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ========== NODO 1: DESCARGAR ADJUNTOS ==========

def descargar_adjuntos_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que descarga los adjuntos de una cotización.

    Este nodo utiliza la función download_attachments_simple de utils.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado con información de descarga

    Updates state:
        - adjuntos_descargados: True si la descarga fue exitosa
        - adjuntos_path: Ruta donde se descargaron los archivos
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 1/5] Descargando adjuntos para {state['codigo_cotizacion']}")

    try:
        resultado = download_attachments_simple(
            codigo_cotizacion=state['codigo_cotizacion'],
            rut_proveedor=state['rut_proveedor'],
            headless=True
        )

        if resultado['success']:
            state['adjuntos_descargados'] = True
            state['adjuntos_path'] = resultado['output_path']
            logging.info(f"✓ Adjuntos descargados: {resultado['total_files']} archivos")
        else:
            state['adjuntos_descargados'] = False
            error_msg = f"Error descargando adjuntos: {resultado.get('error', 'Unknown error')}"
            state = add_error(state, error_msg)
            logging.error(error_msg)

    except Exception as e:
        state['adjuntos_descargados'] = False
        error_msg = f"Excepción descargando adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    return state


# ========== NODO 2: PROCESAR ADJUNTOS ==========

def procesar_adjuntos_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que procesa los adjuntos descargados (extracción de texto).

    Este nodo utiliza la función process_attachments_simple de utils.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado con información de procesamiento

    Updates state:
        - adjuntos_procesados: True si el procesamiento fue exitoso
        - processed_path: Ruta donde se guardaron los archivos procesados
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 2/5] Procesando adjuntos de {state['codigo_cotizacion']}")

    # Verificar que se hayan descargado los adjuntos
    if not state.get('adjuntos_descargados'):
        error_msg = "No se pueden procesar adjuntos porque no fueron descargados"
        state = add_error(state, error_msg)
        logging.error(error_msg)
        return state

    try:
        resultado = process_attachments_simple(
            attachments_path=state['adjuntos_path'],
            output_path=None,  # Usa ruta por defecto
            blacklist=None,
            use_gpu=False
        )

        if resultado['success']:
            state['adjuntos_procesados'] = True
            state['processed_path'] = resultado['output_path']
            logging.info(
                f"✓ Adjuntos procesados: {resultado['processed_files']} archivos, "
                f"{resultado['skipped_files']} omitidos"
            )

            if resultado['skipped_files'] > 0:
                warning_msg = f"{resultado['skipped_files']} archivos fueron omitidos durante el procesamiento"
                state = add_warning(state, warning_msg)

        else:
            state['adjuntos_procesados'] = False
            error_msg = f"Error procesando adjuntos: {resultado.get('error', 'Unknown error')}"
            state = add_error(state, error_msg)
            logging.error(error_msg)

    except Exception as e:
        state['adjuntos_procesados'] = False
        error_msg = f"Excepción procesando adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    return state


# ========== NODO 3: CREAR VECTOR STORE (OPCIONAL) ==========

def crear_vectorstore_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que crea un vector store con los adjuntos procesados.

    NOTA: Este nodo es opcional. Si ya existe un vector store global de adjuntos,
    este nodo puede ser omitido del grafo.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado

    Updates state:
        - vector_store_created: True si el vector store fue creado
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 3/5] Creando vector store para adjuntos procesados")

    # TODO: Implementar creación de vector store si es necesario
    # Por ahora asumimos que existe un vector store global

    state['vector_store_created'] = True
    warning_msg = "Vector store creation not implemented - assuming global store exists"
    state = add_warning(state, warning_msg)

    return state


# ========== NODO 4: RAG CON ADJUNTOS ==========

def rag_adjuntos_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que ejecuta RAG con los adjuntos procesados.

    Utiliza CatalogacionComputadores para extraer atributos del producto
    usando los adjuntos procesados.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado con resultado del RAG

    Updates state:
        - resultado_adjuntos: Diccionario con atributos extraídos
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 4/5] Ejecutando RAG con adjuntos (ROWNUM: {state['payload'].get('ROWNUM')})")

    try:
        catalogador = CatalogacionComputadores(llm_provider=state.get('llm_provider'))

        # Solo ejecutar paso 1 (RAG con adjuntos, sin diccionarios)
        # Lo hacemos internamente llamando directamente al método
        catalogador._initialize_llms()

        # Aquí podríamos llamar a un método específico para solo RAG adjuntos
        # Por ahora usamos el método completo pero guardamos el resultado intermedio
        resultado = catalogador.catalogar_producto(
            payload=state['payload'],
            codigo_cotizacion=state['codigo_cotizacion'],
            rut_proveedor=state['rut_proveedor'],
            use_diccionarios=False  # Solo adjuntos, sin diccionarios
        )

        state['resultado_adjuntos'] = resultado
        logging.info(f"✓ RAG adjuntos completado para ROWNUM {state['payload'].get('ROWNUM')}")

    except Exception as e:
        error_msg = f"Error en RAG con adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    return state


# ========== NODO 5: RAG CON DICCIONARIOS ==========

def rag_diccionarios_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que ejecuta RAG con diccionarios para normalizar atributos.

    Utiliza CatalogacionComputadores para normalizar y completar los atributos
    usando los diccionarios técnicos.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado con resultado normalizado

    Updates state:
        - resultado_diccionarios: Diccionario con atributos normalizados
        - resultado_final: Resultado final de la catalogación
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 5/5] Ejecutando RAG con diccionarios (ROWNUM: {state['payload'].get('ROWNUM')})")

    # Verificar si se debe usar diccionarios
    if not state.get('use_diccionarios', True):
        # Si no se usan diccionarios, el resultado final es el de adjuntos
        state['resultado_final'] = state.get('resultado_adjuntos')
        logging.info("⊘ Diccionarios deshabilitados, usando resultado de adjuntos")
        return state

    # Verificar que exista resultado de adjuntos
    if not state.get('resultado_adjuntos'):
        error_msg = "No se puede ejecutar RAG con diccionarios sin resultado de adjuntos"
        state = add_error(state, error_msg)
        logging.error(error_msg)
        return state

    try:
        catalogador = CatalogacionComputadores(llm_provider=state.get('llm_provider'))

        # Ejecutar catalogación completa (incluye diccionarios)
        resultado = catalogador.catalogar_producto(
            payload=state['payload'],
            codigo_cotizacion=state['codigo_cotizacion'],
            rut_proveedor=state['rut_proveedor'],
            use_diccionarios=True  # Con diccionarios
        )

        state['resultado_diccionarios'] = resultado
        state['resultado_final'] = resultado
        logging.info(f"✓ RAG diccionarios completado para ROWNUM {state['payload'].get('ROWNUM')}")

    except Exception as e:
        error_msg = f"Error en RAG con diccionarios: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

        # Si falla el RAG con diccionarios, usar resultado de adjuntos
        state['resultado_final'] = state.get('resultado_adjuntos')
        warning_msg = "Usando resultado de adjuntos debido a error en diccionarios"
        state = add_warning(state, warning_msg)

    return state


# ========== NODO FINAL: CONSOLIDAR RESULTADO ==========

def consolidar_resultado_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo final que consolida el resultado y calcula métricas.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado final consolidado

    Updates state:
        - tiempo_total: Tiempo total de procesamiento
    """
    logging.info(f"[FINAL] Consolidando resultado para ROWNUM {state['payload'].get('ROWNUM')}")

    # Verificar que haya un resultado final
    if not state.get('resultado_final'):
        error_msg = "No hay resultado final disponible"
        state = add_error(state, error_msg)
        logging.error(error_msg)

    # Logging de resumen
    if state.get('errores'):
        logging.warning(f"⚠️  Catalogación completada con {len(state['errores'])} errores")
        for error in state['errores']:
            logging.warning(f"  - {error}")

    if state.get('warnings'):
        logging.info(f"ℹ️  {len(state['warnings'])} warnings generados")

    logging.info(f"✓ Catalogación finalizada para ROWNUM {state['payload'].get('ROWNUM')}")

    return state


# ========== CONDITIONAL EDGE FUNCTIONS ==========

def should_continue_after_download(state: CatalogacionState) -> str:
    """
    Determina si continuar después de la descarga de adjuntos.

    Args:
        state: Estado actual

    Returns:
        str: Nombre del siguiente nodo o "END"
    """
    if state.get('adjuntos_descargados'):
        return "procesar_adjuntos"
    else:
        return "END"


def should_continue_after_processing(state: CatalogacionState) -> str:
    """
    Determina si continuar después del procesamiento de adjuntos.

    Args:
        state: Estado actual

    Returns:
        str: Nombre del siguiente nodo o "END"
    """
    if state.get('adjuntos_procesados'):
        return "rag_adjuntos"
    else:
        return "END"


def should_use_diccionarios(state: CatalogacionState) -> str:
    """
    Determina si usar diccionarios para normalización.

    Args:
        state: Estado actual

    Returns:
        str: Nombre del siguiente nodo
    """
    if state.get('use_diccionarios', True) and state.get('resultado_adjuntos'):
        return "rag_diccionarios"
    else:
        return "consolidar_resultado"
