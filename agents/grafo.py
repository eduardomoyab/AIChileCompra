"""
LangGraph workflow for Computadores catalogation.

This module defines the complete workflow graph using LangGraph,
connecting all nodes in the proper sequence with conditional edges.
"""

import json
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from langgraph.graph import StateGraph, END

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import CatalogacionState, create_initial_state
from agents.nodos import (
    clasificar_categoria_node,
    descargar_adjuntos_node,
    procesar_adjuntos_node,
    rag_adjuntos_node,
    campos_manuales_node,
    rag_diccionarios_node,
    consolidar_resultado_node,
    should_continue_after_clasificacion,
    should_continue_after_download,
    should_continue_after_processing,
    should_extract_campos_manuales,
    should_use_diccionarios
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_catalogacion_graph():
    """
    Crea el grafo de LangGraph para catalogación de computadores.

    El flujo del grafo es:

    START
      ↓
    descargar_adjuntos
      ↓ (si exitoso)
    procesar_adjuntos
      ↓ (si exitoso)
    rag_adjuntos
      ↓
    ┌────────────────────┐
    │ ¿campos_manuales?  │
    └────────────────────┘
      ↓ Sí                ↓ No
    campos_manuales     rag_adjuntos (continúa)
      ↓
    ┌─────────────┐
    │ ¿usar_dic?  │
    └─────────────┘
      ↓ Sí         ↓ No
    rag_diccionarios  consolidar_resultado
      ↓
    consolidar_resultado
      ↓
    END

    Returns:
        StateGraph: Grafo compilado listo para ejecutar

    Example:
        >>> graph = create_catalogacion_graph()
        >>> result = graph.invoke(initial_state)
    """
    # Crear el grafo
    workflow = StateGraph(CatalogacionState)

    # ========== AGREGAR NODOS ==========

    # Nodo 0: Clasificar categoría (filtro accesorios)
    workflow.add_node("clasificar_categoria", clasificar_categoria_node)

    # Nodo 1: Descargar adjuntos
    workflow.add_node("descargar_adjuntos", descargar_adjuntos_node)

    # Nodo 2: Procesar adjuntos (extracción de texto)
    workflow.add_node("procesar_adjuntos", procesar_adjuntos_node)

    # Nodo 3: RAG con adjuntos
    workflow.add_node("rag_adjuntos", rag_adjuntos_node)

    # Nodo 4: Extracción paralela de campos manuales
    workflow.add_node("campos_manuales", campos_manuales_node)

    # Nodo 5: RAG con diccionarios (normalización)
    workflow.add_node("rag_diccionarios", rag_diccionarios_node)

    # Nodo 6: Consolidar resultado final
    workflow.add_node("consolidar_resultado", consolidar_resultado_node)

    # ========== DEFINIR PUNTO DE ENTRADA ==========

    workflow.set_entry_point("clasificar_categoria")

    # ========== AGREGAR EDGES (Conexiones) ==========

    # Edge 0: clasificar_categoria -> descargar_adjuntos o END
    workflow.add_conditional_edges(
        "clasificar_categoria",
        should_continue_after_clasificacion,
        {
            "descargar_adjuntos": "descargar_adjuntos",
            "END": END
        }
    )

    # Edge 1: descargar_adjuntos -> procesar_adjuntos (condicional) o rag_adjuntos si falló la descarga
    workflow.add_conditional_edges(
        "descargar_adjuntos",
        should_continue_after_download,
        {
            "procesar_adjuntos": "procesar_adjuntos",
            "rag_adjuntos": "rag_adjuntos",
        }
    )

    # Edge 2: procesar_adjuntos -> rag_adjuntos (condicional)
    workflow.add_conditional_edges(
        "procesar_adjuntos",
        should_continue_after_processing,
        {
            "rag_adjuntos": "rag_adjuntos",
            "END": END
        }
    )

    # Edge 3: rag_adjuntos -> campos_manuales o rag_diccionarios o consolidar (condicional)
    workflow.add_conditional_edges(
        "rag_adjuntos",
        should_extract_campos_manuales,
        {
            "campos_manuales": "campos_manuales",
            "rag_diccionarios": "rag_diccionarios",
            "consolidar_resultado": "consolidar_resultado"
        }
    )

    # Edge 4: campos_manuales -> rag_diccionarios o consolidar (condicional)
    workflow.add_conditional_edges(
        "campos_manuales",
        should_use_diccionarios,
        {
            "rag_diccionarios": "rag_diccionarios",
            "consolidar_resultado": "consolidar_resultado"
        }
    )

    # Edge 5: rag_diccionarios -> consolidar_resultado (siempre)
    workflow.add_edge("rag_diccionarios", "consolidar_resultado")

    # Edge 6: consolidar_resultado -> END (siempre)
    workflow.add_edge("consolidar_resultado", END)

    # ========== COMPILAR GRAFO ==========

    graph = workflow.compile()

    logging.info("✓ Grafo de catalogación compilado exitosamente")

    return graph


# Grafo compilado una sola vez al importar el módulo — se reutiliza en todos los requests
_GRAPH: object = None
_GRAPH_LOCK = __import__("threading").Lock()

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = create_catalogacion_graph()
    return _GRAPH


# ========== FUNCIÓN HELPER PARA EJECUTAR EL GRAFO ==========

def ejecutar_catalogacion(
    codigo_cotizacion: str,
    rut_proveedor: str,
    payload: Dict[str, Any],
    use_diccionarios: bool = True,
    llm_provider: str = None,
    downloader: Any = None,
    campos_manuales_lista: List[Dict] = None,
    diccionario_similarity_threshold: float = 0.85,
    diccionario_llm_fallback: bool = True,
    clasificacion_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el workflow completo de catalogación usando el grafo.

    Args:
        codigo_cotizacion: Código de la solicitud de cotización
        rut_proveedor: RUT del proveedor
        payload: Información del producto con estructura:
            {
                'ROWNUM': str,
                'DescripcionProductoComprador': str,
                'DescripcionProductoProveedor': str,
                'productoname': str
            }
        use_diccionarios: Si usar normalización con diccionarios
        llm_provider: Proveedor de LLM (None = usa DEFAULT_LLM_PROVIDER del .env)

    Returns:
        dict: Estado final con el resultado de la catalogación

    Example:
        >>> payload = {
        ...     'ROWNUM': '1',
        ...     'DescripcionProductoComprador': 'NOTEBOOK HP 8GB RAM',
        ...     'DescripcionProductoProveedor': 'Notebook HP Pavilion',
        ...     'productoname': 'Computadores'
        ... }
        >>> resultado = ejecutar_catalogacion(
        ...     "12345678",
        ...     "76123456-7",
        ...     payload
        ... )
        >>> print(resultado['resultado_final'])
    """
    # Crear estado inicial
    initial_state = create_initial_state(
        codigo_cotizacion=codigo_cotizacion,
        rut_proveedor=rut_proveedor,
        payload=payload,
        use_diccionarios=use_diccionarios,
        llm_provider=llm_provider,
        campos_manuales_lista=campos_manuales_lista,
        diccionario_similarity_threshold=diccionario_similarity_threshold,
        diccionario_llm_fallback=diccionario_llm_fallback,
        clasificacion_prompt=clasificacion_prompt,
    )

    # Agregar downloader al state si se proporcionó
    if downloader is not None:
        initial_state['downloader'] = downloader

    # Reutilizar grafo compilado (singleton)
    graph = _get_graph()

    logging.info(f"\n{'='*60}")
    logging.info(f"INICIANDO CATALOGACIÓN")
    logging.info(f"Cotización: {codigo_cotizacion}")
    logging.info(f"Proveedor: {rut_proveedor}")
    logging.info(f"ROWNUM: {payload.get('ROWNUM')}")
    logging.info(f"{'='*60}\n")

    # Ejecutar workflow
    final_state = graph.invoke(initial_state)

    logging.info(f"\n{'='*60}")
    logging.info(f"CATALOGACIÓN FINALIZADA")
    logging.info(f"ROWNUM: {payload.get('ROWNUM')}")
    logging.info(f"Errores: {len(final_state.get('errores', []))}")
    logging.info(f"Warnings: {len(final_state.get('warnings', []))}")
    logging.info(f"{'='*60}\n")

    return final_state


# ========== GRAFO SIMPLIFICADO PARA MODO TEXTO ==========

def create_catalogacion_texto_graph():
    """
    Crea un grafo simplificado para catalogación desde texto directo.

    Flujo:
        START → campos_manuales → rag_diccionarios → consolidar_resultado → END

    Los nodos de descarga/procesamiento/RAG de adjuntos se omiten porque el texto
    ya viene como input en state['texto_directo'].
    """
    workflow = StateGraph(CatalogacionState)

    workflow.add_node("campos_manuales", campos_manuales_node)
    workflow.add_node("rag_diccionarios", rag_diccionarios_node)
    workflow.add_node("consolidar_resultado", consolidar_resultado_node)

    workflow.set_entry_point("campos_manuales")

    workflow.add_conditional_edges(
        "campos_manuales",
        should_use_diccionarios,
        {
            "rag_diccionarios": "rag_diccionarios",
            "consolidar_resultado": "consolidar_resultado",
        }
    )
    workflow.add_edge("rag_diccionarios", "consolidar_resultado")
    workflow.add_edge("consolidar_resultado", END)

    graph = workflow.compile()
    logging.info("✓ Grafo texto compilado exitosamente")
    return graph


# Singleton para el grafo de texto
_GRAPH_TEXTO: object = None
_GRAPH_TEXTO_LOCK = __import__("threading").Lock()

def _get_graph_texto():
    global _GRAPH_TEXTO
    if _GRAPH_TEXTO is None:
        with _GRAPH_TEXTO_LOCK:
            if _GRAPH_TEXTO is None:
                _GRAPH_TEXTO = create_catalogacion_texto_graph()
    return _GRAPH_TEXTO


def ejecutar_catalogacion_texto(
    nombre_producto: str,
    descripcion: Optional[str],
    atributos: Optional[Dict[str, Any]],
    categoria: str = "Computadores",
    campos_manuales_lista: Optional[List[Dict]] = None,
    use_diccionarios: bool = True,
    llm_provider: Optional[str] = None,
    diccionario_similarity_threshold: float = 0.85,
    diccionario_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta catalogación desde texto directo (sin adjuntos de Mercado Público).

    Construye el texto de contexto, crea un payload sintético compatible con el
    estado del grafo y ejecuta el grafo simplificado (campos_manuales →
    rag_diccionarios → consolidar_resultado).

    Args:
        nombre_producto: Nombre del producto (requerido)
        descripcion: Descripción libre del producto
        atributos: Tabla de especificaciones técnicas {campo: valor}
        categoria: Categoría del producto
        campos_manuales_lista: Lista de {campo, contexto} a extraer
        use_diccionarios: Si normalizar con diccionarios
        llm_provider: Proveedor LLM. None = usa DEFAULT_LLM_PROVIDER del .env
        diccionario_similarity_threshold: Score mínimo para match de diccionario
        diccionario_llm_fallback: Si usar LLM cuando no hay match sobre threshold

    Returns:
        dict: Estado final con resultado_final, errores, warnings, etc.
    """
    rownum = str(uuid4())[:8]

    # Construir texto de contexto plano
    partes = [f"Nombre del producto: {nombre_producto}"]
    if descripcion:
        partes.append(f"Descripción: {descripcion}")
    if atributos:
        partes.append(f"Especificaciones técnicas: {json.dumps(atributos, ensure_ascii=False)}")
    texto_directo = "\n".join(partes)

    # Payload sintético compatible con los nodos downstream (rag_diccionarios, consolidar)
    payload = {
        "ROWNUM": rownum,
        "Categoria": categoria,
        "DescripcionProductoComprador": nombre_producto,
        "DescripcionProductoProveedor": descripcion or nombre_producto,
        "productoname": nombre_producto,
    }

    initial_state = CatalogacionState(
        # Input
        codigo_cotizacion="texto",
        rut_proveedor="texto",
        payload=payload,
        # Modo texto
        texto_directo=texto_directo,
        # Config
        use_diccionarios=use_diccionarios,
        llm_provider=llm_provider,
        campos_manuales_lista=campos_manuales_lista or [],
        diccionario_similarity_threshold=diccionario_similarity_threshold,
        diccionario_llm_fallback=diccionario_llm_fallback,
        # Estado de adjuntos: marcados como False (no aplica en modo texto)
        adjuntos_descargados=False,
        adjuntos_procesados=False,
        adjuntos_path=None,
        processed_path=None,
        vector_store_created=False,
        # resultado_adjuntos inicializado con ROWNUM para que rag_diccionarios_node lo encuentre
        resultado_adjuntos={"ROWNUM": rownum},
        resultado_diccionarios=None,
        resultado_campos_manuales=None,
        resultado_final=None,
        errores=[],
        warnings=[],
        tiempo_total=None,
        tiempos=[],
        adjuntos_vectorstore=None,
    )

    graph = _get_graph_texto()

    logging.info(f"\n{'='*60}")
    logging.info(f"INICIANDO CATALOGACIÓN DESDE TEXTO")
    logging.info(f"Producto: {nombre_producto}")
    logging.info(f"ROWNUM: {rownum}")
    logging.info(f"{'='*60}\n")

    final_state = graph.invoke(initial_state)

    logging.info(f"\n{'='*60}")
    logging.info(f"CATALOGACIÓN TEXTO FINALIZADA")
    logging.info(f"Errores: {len(final_state.get('errores', []))}")
    logging.info(f"{'='*60}\n")

    return final_state


# ========== VISUALIZACIÓN DEL GRAFO (OPCIONAL) ==========

def visualizar_grafo(output_path: str = "grafo_catalogacion.png"):
    """
    Genera una visualización del grafo y la guarda como imagen.

    Args:
        output_path: Ruta donde guardar la imagen

    Note:
        Requiere tener instalado graphviz y python-graphviz:
        - pip install graphviz
        - Instalar Graphviz system package (https://graphviz.org/download/)

    Example:
        >>> visualizar_grafo("docs/workflow.png")
    """
    try:
        graph = create_catalogacion_graph()

        # Intentar generar visualización
        from IPython.display import Image, display

        # Generar imagen del grafo
        img = graph.get_graph().draw_mermaid_png()

        # Guardar imagen
        with open(output_path, 'wb') as f:
            f.write(img)

        logging.info(f"✓ Visualización del grafo guardada en: {output_path}")

    except ImportError:
        logging.warning(
            "⚠️  No se pudo generar visualización. "
            "Instala: pip install graphviz && apt-get install graphviz"
        )
    except Exception as e:
        logging.error(f"Error generando visualización: {e}")


