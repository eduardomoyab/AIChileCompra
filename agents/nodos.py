"""
Nodes for the Computadores catalogation workflow using LangGraph.

Each node is a function that takes the state and returns an updated state.
Nodes wrap existing functionality from utils and agents modules.
"""

import os
import sys
import json
import logging
import time
import asyncio
import re
import shutil
import threading
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lock por (codigo_cotizacion, rut_proveedor) para evitar descargas concurrentes del mismo recurso
_download_locks: Dict[str, threading.Lock] = {}
_download_locks_mutex = threading.Lock()

# Semáforo global: limita LLM calls simultáneas para no disparar ráfagas que quemen el RPM
_llm_semaphore = threading.Semaphore(int(os.getenv("LLM_MAX_CONCURRENT", "2")))

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import CatalogacionState, add_error, add_warning, add_tiempo
from agents.normalizador import Normalizador
from agents.get_agent import get_llm
from agents.get_vectorstore import create_faiss_from_files, create_faiss_from_texts
from utils.get_attachments import download_attachments_simple
from utils.process_attachments import process_attachments_simple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ========== NODO 0: CLASIFICAR CATEGORÍA (FILTRO DE ACCESORIOS) ==========

def clasificar_categoria_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo previo que determina si el producto es de la categoría buscada o es un accesorio.

    Si clasificacion_prompt no está configurado, se omite la clasificación y el flujo continúa.
    Si está configurado, un LLM evalúa las descripciones del producto y decide si es accesorio.
    En caso de accesorio, el flujo termina con resultado vacío.
    """
    clasificacion_prompt = state.get('clasificacion_prompt')
    if not clasificacion_prompt:
        state['es_accesorio'] = False
        return state

    _NODO = "nodo_0_clasificar_categoria"
    t_nodo = time.time()
    logging.info(f"[NODO 0] Clasificando si el producto es accesorio o categoría principal")

    payload = state.get('payload', {})
    nombre = payload.get('productoname', '')
    desc_comprador = payload.get('DescripcionProductoComprador', '')
    desc_proveedor = payload.get('DescripcionProductoProveedor', '')
    categoria = payload.get('Categoria', '')

    prompt = f"""Eres un clasificador de productos de compras públicas. Tu única tarea es determinar si el producto descrito es de la categoría buscada o es un accesorio/producto diferente.

CATEGORÍA BUSCADA:
{clasificacion_prompt}

PRODUCTO A CLASIFICAR:
- Categoría declarada: {categoria}
- Nombre: {nombre}
- Descripción comprador: {desc_comprador}
- Descripción proveedor: {desc_proveedor}

INSTRUCCIONES:
- Si el producto corresponde a la categoría buscada → responde EXACTAMENTE: NO_ACCESORIO
- Si el producto es un accesorio, periférico, o no corresponde a la categoría → responde EXACTAMENTE: ACCESORIO
- Responde SOLO una de esas dos palabras, sin explicación."""

    try:
        llm_provider = state.get('llm_provider') or os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')
        llm = get_llm(model_provider=llm_provider, temperature=0.0)
        response = llm.invoke(prompt)
        decision = response.content.strip().upper()
        es_accesorio = decision == 'ACCESORIO'
        state['es_accesorio'] = es_accesorio
        if es_accesorio:
            logging.info(f"⛔ Producto clasificado como ACCESORIO — se omite catalogación")
            state['resultado_final'] = {"ROWNUM": payload.get('ROWNUM'), "es_accesorio": True}
            warning_msg = f"Producto '{nombre}' clasificado como accesorio — flujo omitido"
            state = add_warning(state, warning_msg)
        else:
            logging.info(f"✓ Producto clasificado como categoría principal — continúa flujo")
    except Exception as e:
        logging.warning(f"Error en clasificación de categoría: {e} — se continúa el flujo por defecto")
        state['es_accesorio'] = False

    state = add_tiempo(state, _NODO, "total", "Clasificación categoría vs accesorio", time.time() - t_nodo)
    return state


def should_continue_after_clasificacion(state: CatalogacionState) -> str:
    """Si es accesorio, termina. Si no, continúa a descargar adjuntos."""
    if state.get('es_accesorio'):
        return "END"
    return "descargar_adjuntos"


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
    _NODO = "nodo_1_descargar_adjuntos"
    t_nodo = time.time()

    try:
        # Sub-fase 1a: verificar caché
        t0 = time.time()
        attachments_dir = os.path.join(
            os.getenv('ATTACHMENTS_OUTPUT_PATH', 'attachments'),
            state['codigo_cotizacion'],
            state['rut_proveedor']
        )

        # TTL en segundos: 0 = sin caché (siempre re-descarga), >0 = usar caché si archivos son frescos
        cache_ttl = int(os.getenv('ATTACHMENTS_CACHE_TTL_SECONDS', '0'))

        processed_dir = os.path.join(
            os.getenv('ATTACHMENTS_OUTPUT_PATH', 'attachments'),
            state['codigo_cotizacion'],
            'processed'
        )

        def _clear_processed():
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir, ignore_errors=True)
                logging.info(f"  Carpeta processed limpiada: {processed_dir}")

        def _check_cache(directory):
            if not os.path.exists(directory):
                return False, []
            files = [f for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
            if not files:
                return False, []
            if cache_ttl <= 0:
                # Sin caché: borrar descargas y processed para forzar reprocesamiento completo
                shutil.rmtree(directory, ignore_errors=True)
                _clear_processed()
                logging.info(f"  Cache TTL=0 — directorio limpiado: {directory}")
                return False, []
            # Con TTL: verificar antigüedad del archivo más reciente
            newest_mtime = max(
                os.path.getmtime(os.path.join(directory, f)) for f in files
            )
            age = time.time() - newest_mtime
            if age > cache_ttl:
                shutil.rmtree(directory, ignore_errors=True)
                _clear_processed()
                logging.info(f"  Cache expirado ({age:.0f}s > TTL {cache_ttl}s) — directorio limpiado")
                return False, []
            return True, files

        cache_hit, existing_files = _check_cache(attachments_dir)
        state = add_tiempo(state, _NODO, "check_cache",
                           f"Verificar caché local de adjuntos (hit={cache_hit}, ttl={cache_ttl}s)",
                           time.time() - t0)

        if cache_hit:
            logging.info(f"✓ Archivos ya descargados: {len(existing_files)} archivos encontrados")
            state['adjuntos_descargados'] = True
            state['adjuntos_path'] = attachments_dir
            state = add_tiempo(state, _NODO, "total",
                               "Total nodo descargar_adjuntos (caché)",
                               time.time() - t_nodo)
            return state

        # Sub-fase 1b: descarga real — serializada por (cotizacion, proveedor) para evitar race conditions
        lock_key = f"{state['codigo_cotizacion']}::{state['rut_proveedor']}"
        with _download_locks_mutex:
            if lock_key not in _download_locks:
                _download_locks[lock_key] = threading.Lock()
        download_lock = _download_locks[lock_key]

        with download_lock:
            # Re-verificar caché: puede que otro thread ya haya descargado mientras esperábamos
            cache_hit, existing_files = _check_cache(attachments_dir)
            if cache_hit:
                logging.info(f"✓ Archivos descargados por otro request concurrente: {len(existing_files)} archivos")
                state['adjuntos_descargados'] = True
                state['adjuntos_path'] = attachments_dir
                state = add_tiempo(state, _NODO, "total",
                                   "Total nodo descargar_adjuntos (caché post-lock)",
                                   time.time() - t_nodo)
                return state

            t0 = time.time()
            downloader = state.get('downloader', None)
            resultado = download_attachments_simple(
                codigo_cotizacion=state['codigo_cotizacion'],
                rut_proveedor=state['rut_proveedor'],
                headless=True,
                downloader=downloader
            )
        state = add_tiempo(state, _NODO, "download_api",
                           f"Descarga desde Mercado Público (success={resultado.get('success')},"
                           f" archivos={resultado.get('total_files', 0)})",
                           time.time() - t0)

        if resultado['success']:
            state['adjuntos_descargados'] = True
            state['adjuntos_path'] = resultado['output_path']
            logging.info(f"✓ Adjuntos descargados: {resultado['total_files']} archivos")
            # Registrar detalle de archivos por carpeta
            _path = resultado['output_path']
            _tech = sorted(os.listdir(os.path.join(_path, 'tech'))) if os.path.exists(os.path.join(_path, 'tech')) else []
            _econ = sorted(os.listdir(os.path.join(_path, 'econ'))) if os.path.exists(os.path.join(_path, 'econ')) else []
            _otros = sorted(f for f in os.listdir(_path) if os.path.isfile(os.path.join(_path, f))) if os.path.exists(_path) else []
            diag = state.get('diagnostico') or {}
            diag['descarga'] = {
                'tecnicos': _tech, 'num_tecnicos': len(_tech),
                'economicos': _econ, 'num_economicos': len(_econ),
                'otros': _otros,
                'total': len(_tech) + len(_econ) + len(_otros),
            }
            state['diagnostico'] = diag
        else:
            state['adjuntos_descargados'] = False
            _raw_error = resultado.get('error') or 'Sin detalle de error'
            _parciales = resultado.get('files_downloaded') or []
            error_msg = f"Error descargando adjuntos: {_raw_error}"
            state = add_error(state, error_msg)
            logging.error(error_msg)
            diag = state.get('diagnostico') or {}
            diag['descarga'] = {
                'error': _raw_error,
                'archivos_parciales': _parciales,
                'num_parciales': len(_parciales),
                'total': len(_parciales),
            }
            state['diagnostico'] = diag

    except Exception as e:
        state['adjuntos_descargados'] = False
        error_msg = f"Excepción descargando adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo descargar_adjuntos",
                       time.time() - t_nodo)
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
    _NODO = "nodo_2_procesar_adjuntos"
    t_nodo = time.time()

    # Verificar que se hayan descargado los adjuntos
    if not state.get('adjuntos_descargados'):
        error_msg = "No se pueden procesar adjuntos porque no fueron descargados"
        state = add_error(state, error_msg)
        logging.error(error_msg)
        return state

    try:
        # Sub-fase 2a: extracción de texto de PDFs/imágenes
        t0 = time.time()
        resultado = process_attachments_simple(
            attachments_path=state['adjuntos_path'],
            output_path=None,
            blacklist=None,
            use_gpu=False
        )
        state = add_tiempo(state, _NODO, "process_attachments",
                           f"Extracción de texto (procesados={resultado.get('processed_files', 0)},"
                           f" omitidos={resultado.get('skipped_files', 0)},"
                           f" success={resultado.get('success')})",
                           time.time() - t0)

        if resultado['success']:
            state['adjuntos_procesados'] = True
            state['processed_path'] = resultado['output_path']
            if resultado['processed_files'] > 0:
                logging.info(
                    f"✓ Adjuntos procesados: {resultado['processed_files']} archivos, "
                    f"{resultado['skipped_files']} omitidos"
                )
            else:
                logging.info("✓ Archivos ya procesados previamente")
            if resultado['skipped_files'] > 0:
                warning_msg = f"{resultado['skipped_files']} archivos fueron omitidos durante el procesamiento"
                state = add_warning(state, warning_msg)
            # Registrar detalle de archivos procesados
            _ppath = resultado['output_path']
            _txts = sorted(f for f in os.listdir(_ppath) if f.endswith('.txt') and f != 'skipped_files.txt') if os.path.exists(_ppath) else []
            diag = state.get('diagnostico') or {}
            _arch = resultado.get('archivos_comprimidos') or {}
            diag['proceso'] = {
                'txt_generados': _txts,
                'num_procesados': resultado.get('processed_files', 0),
                'num_omitidos': resultado.get('skipped_files', 0),
                'errores_proceso': resultado.get('errors') or [],
                'comprimidos_encontrados': _arch.get('archivos_comprimidos_encontrados', 0),
                'comprimidos_extraidos': _arch.get('extraidos', 0),
                'comprimidos_fallidos': _arch.get('fallidos') or [],
            }
            state['diagnostico'] = diag
        else:
            state['adjuntos_procesados'] = False
            error_msg = f"Error procesando adjuntos: {resultado.get('error', 'Unknown error')}"
            state = add_error(state, error_msg)
            logging.error(error_msg)
            diag = state.get('diagnostico') or {}
            diag['proceso'] = {'error': error_msg, 'num_procesados': 0, 'num_omitidos': 0}
            state['diagnostico'] = diag

    except Exception as e:
        state['adjuntos_procesados'] = False
        error_msg = f"Excepción procesando adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo procesar_adjuntos",
                       time.time() - t_nodo)
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

    Utiliza Normalizador para extraer atributos del producto
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
    _NODO = "nodo_3_rag_adjuntos"
    t_nodo = time.time()

    try:
        # Crear FAISS una sola vez y guardarlo en estado para reutilizar en campos_manuales_node
        t0 = time.time()
        processed_path = state['processed_path']
        all_txt_files = [
            os.path.join(processed_path, f)
            for f in os.listdir(processed_path)
            if f.endswith('.txt') and f != 'skipped_files.txt'
        ] if processed_path and os.path.exists(processed_path) else []

        skip_patterns = [p.strip().lower() for p in os.getenv('SKIP_FILENAME_PATTERNS', 'anexo,formulario,propuesta,simulacion').split(',') if p.strip()]
        tech_keywords = [kw.strip().lower() for kw in os.getenv(
            'TECH_FILENAME_KEYWORDS',
            'procesador,cpu,ram,memoria,ssd,hdd,disco,pantalla,monitor,display,'
            'intel,amd,nvidia,gpu,ficha,especificacion,tecnica,tecnico,almacenamiento,bateria,ghz'
        ).split(',') if kw.strip()]

        def _is_admin(fname):
            f = fname.lower()
            return any(p in f for p in skip_patterns) and not any(kw in f for kw in tech_keywords)

        txt_files = [f for f in all_txt_files if not _is_admin(os.path.basename(f))] or all_txt_files

        if txt_files:
            metadatas = [{"codigo_cotizacion": state['codigo_cotizacion'], "rut_proveedor": state['rut_proveedor'], "source_file": os.path.basename(f)} for f in txt_files]
            chunk_size = int(os.getenv('ADJUNTOS_CHUNK_SIZE', '500'))
            chunk_overlap = int(os.getenv('ADJUNTOS_CHUNK_OVERLAP', '100'))
            vectorstore_shared = create_faiss_from_files(txt_files, metadatas, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            logging.info(f"✓ FAISS creado con {len(txt_files)} archivos de adjuntos")
        else:
            # Fallback: no hay adjuntos procesados → crear FAISS desde la descripción del producto
            payload = state.get('payload', {})
            fallback_text = (
                f"Categoría: {payload.get('Categoria', '')}\n"
                f"Nombre: {payload.get('productoname', '')}\n"
                f"Descripción comprador: {payload.get('DescripcionProductoComprador', '')}\n"
                f"Descripción proveedor: {payload.get('DescripcionProductoProveedor', '')}"
            )
            vectorstore_shared = create_faiss_from_texts(
                [fallback_text],
                [{"source": "descripcion_producto"}]
            )
            warning_msg = "No se encontraron adjuntos procesados — usando descripción del producto como contexto"
            state = add_warning(state, warning_msg)
            logging.warning(f"⚠️  {warning_msg}")

        state['adjuntos_vectorstore'] = vectorstore_shared
        state = add_tiempo(state, _NODO, "crear_faiss",
                           f"Crear FAISS ({len(txt_files)} archivos adjuntos)",
                           time.time() - t0)

        # Registrar qué archivos txt se usaron en el FAISS y el modo (adjuntos vs fallback)
        diag = state.get('diagnostico') or {}
        _modo = "adjuntos" if txt_files else "descripcion_fallback"
        diag['rag'] = {
            'modo': _modo,
            'archivos_txt_usados': [os.path.basename(f) for f in txt_files],
            'num_archivos_txt': len(txt_files),
        }
        state['diagnostico'] = diag

        state['resultado_adjuntos'] = {"ROWNUM": state['payload'].get('ROWNUM')}
        logging.info(f"✓ Nodo RAG adjuntos completado — todos los campos se extraen via campos_manuales")

    except Exception as e:
        error_msg = f"Error en RAG con adjuntos: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo rag_adjuntos",
                       time.time() - t_nodo)
    return state


# ========== NODO 5: RAG CON DICCIONARIOS ==========

def rag_diccionarios_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que ejecuta RAG con diccionarios para normalizar atributos.

    Utiliza Normalizador para normalizar y completar los atributos
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
    _NODO = "nodo_5_rag_diccionarios"
    t_nodo = time.time()

    if not state.get('use_diccionarios', True):
        state['resultado_final'] = state.get('resultado_adjuntos')
        logging.info("⊘ Diccionarios deshabilitados, usando resultado de adjuntos")
        diag = state.get('diagnostico') or {}
        diag['normalizacion'] = {'use_diccionarios': False}
        state['diagnostico'] = diag
        state = add_tiempo(state, _NODO, "total",
                           "Total nodo rag_diccionarios (deshabilitado)",
                           time.time() - t_nodo)
        return state

    if not state.get('resultado_adjuntos'):
        error_msg = "No se puede ejecutar RAG con diccionarios sin resultado de adjuntos"
        state = add_error(state, error_msg)
        logging.error(error_msg)
        return state

    try:
        # Sub-fase 5a: instanciar catalogador + mezcla de campos manuales
        t0 = time.time()
        catalogador = Normalizador(llm_provider=state.get('llm_provider'))
        resultado_previo = dict(state['resultado_adjuntos'])
        if state.get('resultado_campos_manuales'):
            logging.info("Mezclando campos manuales con resultado de adjuntos...")
            for campo, valor in state['resultado_campos_manuales'].items():
                resultado_previo[campo] = valor
            logging.info(f"  Campos mezclados: {list(state['resultado_campos_manuales'].keys())}")
        state = add_tiempo(state, _NODO, "merge_campos_manuales",
                           "Mezclar resultado adjuntos + campos manuales",
                           time.time() - t0)

        # Sub-fases 5b..5d: dentro de aplicar_diccionarios
        tiempos_locales = state.get('tiempos', [])
        dic_overrides = {
            cm['campo']: cm['diccionario']
            for cm in (state.get('campos_manuales_lista') or [])
            if cm.get('diccionario')
        }
        resultado = catalogador.aplicar_diccionarios(
            resultado_adjuntos=resultado_previo,
            categoria=state['payload'].get('Categoria', ''),
            tiempos=tiempos_locales,
            nodo_nombre=_NODO,
            similarity_threshold=state.get('diccionario_similarity_threshold', 0.85),
            llm_fallback=state.get('diccionario_llm_fallback', True),
            dic_overrides=dic_overrides,
        )
        state['tiempos'] = tiempos_locales

        # Registrar diagnóstico de normalización
        _sim, _llm, _sin_match = [], [], []
        for t in tiempos_locales:
            if t.get('nodo') != _NODO:
                continue
            fase = t.get('fase', '')
            if fase.startswith('sim_'):
                _sim.append(fase[4:])
            elif fase.startswith('llm_fallback_'):
                _llm.append(fase[13:])
        # Campos que cambiaron de valor respecto al input
        _cambiados = {
            k: {"antes": str(resultado_previo.get(k)), "despues": str(v)}
            for k, v in resultado.items()
            if k != 'ROWNUM' and str(resultado_previo.get(k)) != str(v)
        }
        diag = state.get('diagnostico') or {}
        diag['normalizacion'] = {
            'use_diccionarios': True,
            'campos_similitud': _sim,
            'campos_llm_fallback': _llm,
            'num_similitud': len(_sim),
            'num_llm_fallback': len(_llm),
            'campos_cambiados': _cambiados,
            'num_cambiados': len(_cambiados),
        }
        state['diagnostico'] = diag

        state['resultado_diccionarios'] = resultado
        state['resultado_final'] = resultado
        logging.info(f"✓ RAG diccionarios completado para ROWNUM {state['payload'].get('ROWNUM')}")

    except Exception as e:
        error_msg = f"Error en RAG con diccionarios: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)
        state['resultado_final'] = state.get('resultado_adjuntos')
        warning_msg = "Usando resultado de adjuntos debido a error en diccionarios"
        state = add_warning(state, warning_msg)
        diag = state.get('diagnostico') or {}
        diag['normalizacion'] = {'use_diccionarios': True, 'error': error_msg}
        state['diagnostico'] = diag

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo rag_diccionarios",
                       time.time() - t_nodo)
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
        - resultado_final: Resultado consolidado con campos_manuales si existen
        - tiempo_total: Tiempo total de procesamiento
    """
    logging.info(f"[FINAL] Consolidando resultado para ROWNUM {state['payload'].get('ROWNUM')}")
    _NODO = "nodo_6_consolidar_resultado"
    t_nodo = time.time()

    # Sub-fase 6a: merge de campos manuales en resultado final
    t0 = time.time()
    if not state.get('resultado_final'):
        if state.get('resultado_adjuntos'):
            logging.info("Usando resultado_adjuntos como resultado_final (tipo 'Otro' o sin diccionarios)")
            state['resultado_final'] = state['resultado_adjuntos']
        else:
            error_msg = "No hay resultado final disponible"
            state = add_error(state, error_msg)
            logging.error(error_msg)

    if state.get('resultado_final') and state.get('resultado_campos_manuales'):
        logging.info("Consolidando campos manuales en resultado final...")
        resultado_consolidado = dict(state['resultado_final'])
        for campo, valor in state['resultado_campos_manuales'].items():
            valor_actual = resultado_consolidado.get(campo)
            if not valor_actual or \
               (valor_actual in ['No disponible', 'No especificado'] and valor not in ['No disponible', 'No especificado']):
                resultado_consolidado[campo] = valor
            elif valor_actual not in ['No disponible', 'No especificado']:
                logging.debug(f"  Manteniendo valor actualizado para {campo}: {valor_actual}")
        state['resultado_final'] = resultado_consolidado
        logging.info(f"✓ Campos manuales consolidados en raíz: {list(state['resultado_campos_manuales'].keys())}")
    state = add_tiempo(state, _NODO, "merge_campos",
                       "Merge de campos manuales en resultado final",
                       time.time() - t0)

    if state.get('errores'):
        logging.warning(f"⚠️  Catalogación completada con {len(state['errores'])} errores")
        for error in state['errores']:
            logging.warning(f"  - {error}")
    if state.get('warnings'):
        logging.info(f"ℹ️  {len(state['warnings'])} warnings generados")

    # Registrar resumen de campos del resultado final
    _VACIOS = {"no disponible", "no especificado", "false", "", "none"}
    rf = state.get('resultado_final') or {}
    campos_con_valor = {k: v for k, v in rf.items()
                        if k != 'ROWNUM' and v is not None and str(v).strip().lower() not in _VACIOS}
    campos_vacios = [k for k, v in rf.items()
                     if k != 'ROWNUM' and (v is None or str(v).strip().lower() in _VACIOS)]
    diag = state.get('diagnostico') or {}
    diag['campos'] = {
        'extraidos': campos_con_valor,
        'vacios': campos_vacios,
        'num_extraidos': len(campos_con_valor),
        'num_vacios': len(campos_vacios),
    }
    state['diagnostico'] = diag

    # Sub-fase 6b: limpiar archivos temporales
    t0 = time.time()
    try:
        if state.get('adjuntos_path') and os.path.exists(state['adjuntos_path']):
            shutil.rmtree(state['adjuntos_path'])
            logging.info(f"✓ Adjuntos descargados eliminados: {state['adjuntos_path']}")
        if state.get('processed_path') and os.path.exists(state['processed_path']):
            shutil.rmtree(state['processed_path'])
            logging.info(f"✓ Adjuntos procesados eliminados: {state['processed_path']}")
    except Exception as e:
        logging.warning(f"⚠️  No se pudieron eliminar archivos temporales: {e}")
    state = add_tiempo(state, _NODO, "limpieza_archivos",
                       "Eliminar adjuntos descargados y procesados",
                       time.time() - t0)

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo consolidar_resultado",
                       time.time() - t_nodo)
    logging.info(f"✓ Catalogación finalizada para ROWNUM {state['payload'].get('ROWNUM')}")
    return state


# ========== NODO 6: EXTRACCIÓN PARALELA DE CAMPOS MANUALES ==========

def _limpiar_valor(valor: str, campo: str = "") -> str:
    """
    Limpia caracteres especiales y formatea el valor extraído de forma genérica.

    Aplica reglas inteligentes según patrones detectados en el nombre del campo,
    funcionando con CUALQUIER campo que el usuario defina.

    Args:
        valor: Valor a limpiar
        campo: Nombre del campo (para detectar el tipo de dato esperado)

    Returns:
        str: Valor limpio y formateado
    """
    # Limpieza básica
    valor = ' '.join(valor.split())  # Eliminar saltos de línea y espacios extra
    valor = re.sub(r'[^\w\s.,()°"\'-]', '', valor, flags=re.UNICODE)  # Eliminar caracteres especiales
    valor = re.sub(r'\s+', ' ', valor)  # Normalizar espacios
    valor = valor.strip()

    campo_lower = campo.lower()

    # ========== REGLAS GENÉRICAS BASADAS EN PATRONES DEL NOMBRE DEL CAMPO ==========

    # 1. Si el campo contiene "(GB)" -> Extraer solo número + GB
    if "(gb)" in campo_lower or campo_lower.endswith("gb"):
        # Convertir TB a GB si es necesario
        match_tb = re.search(r'(\d+\.?\d*)\s*TB', valor, re.IGNORECASE)
        if match_tb:
            tb_value = float(match_tb.group(1))
            gb_value = int(tb_value * 1000)
            return f"{gb_value} GB"
        # Extraer GB directamente
        match_gb = re.search(r'(\d+)\s*(GB)?', valor, re.IGNORECASE)
        if match_gb:
            return f"{match_gb.group(1)} GB"

    # 2. Si el campo contiene "(TB)" -> Extraer solo número + TB
    if "(tb)" in campo_lower or campo_lower.endswith("tb"):
        match = re.search(r'(\d+\.?\d*)\s*(TB)?', valor, re.IGNORECASE)
        if match:
            return f"{match.group(1)} TB"

    # 3. Si el campo contiene "(Pulgadas)" o "(inches)" -> Solo número
    if "(pulgada" in campo_lower or "(inch" in campo_lower or "\"" in campo:
        match = re.search(r'(\d+\.?\d*)', valor)
        if match:
            return match.group(1)

    # 4. Si el campo contiene "Tipo" + otra palabra -> Eliminar números/capacidades
    if "tipo" in campo_lower:
        # Eliminar números seguidos de unidades (GB, TB, MHz, GHz, etc.)
        valor_limpio = re.sub(r'\d+\.?\d*\s*(GB|TB|MB|MHz|GHz|rpm|gb|tb|mb|mhz|ghz)', '', valor, flags=re.IGNORECASE)
        valor_limpio = ' '.join(valor_limpio.split())

        # Si es "Tipo RAM", buscar patrones DDR
        if "ram" in campo_lower:
            match = re.search(r'(LP)?DDR\d[X]?', valor_limpio, re.IGNORECASE)
            if match:
                return match.group(0).upper()

        return valor_limpio.strip()

    # 5. Si el campo termina con unidad entre paréntesis como (MHz), (GHz), (W), etc.
    unit_match = re.search(r'\(([A-Za-z]+)\)$', campo)
    if unit_match:
        unit = unit_match.group(1)
        # Extraer número + unidad
        match = re.search(rf'(\d+\.?\d*)\s*{unit}?', valor, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {unit}"

    # 5b. Si el campo es Procesador/CPU → eliminar velocidades de reloj, núcleos y texto extra
    if any(kw in campo_lower for kw in ["procesador", "cpu", "processor"]):
        # Cortar en la primera coma o paréntesis (ej: "Intel Core i7-1355U (1.7 GHz)" → "Intel Core i7-1355U")
        valor = re.sub(r'\s*[,\(].*$', '', valor)
        # Cortar velocidades de reloj si el LLM las dejó antes del paréntesis (ej: "... 5.0 GHz")
        valor = re.sub(r'\s+\d+\.?\d*\s*GHz.*$', '', valor, flags=re.IGNORECASE)
        return valor.strip()

    # 6. Si el campo contiene palabras clave numéricas (Nucleos, Hilos, Cores, Threads, etc.)
    numeric_keywords = ["nucleo", "hilo", "core", "thread", "puerto", "slot", "canal"]
    if any(kw in campo_lower for kw in numeric_keywords):
        match = re.search(r'\d+', valor)
        if match:
            return match.group(0)

    # 7. Si el campo es solo un sustantivo simple (Marca, Modelo, Color, etc.)
    #    y el valor tiene capacidades, eliminarlas
    simple_fields = ["marca", "modelo", "color", "serie", "version"]
    if any(campo_lower == sf or campo_lower.startswith(sf + " ") for sf in simple_fields):
        # Eliminar capacidades numéricas al final
        valor_limpio = re.sub(r'\s*\d+\.?\d*\s*(GB|TB|MB|MHz|GHz|W|rpm).*$', '', valor, flags=re.IGNORECASE)
        return valor_limpio.strip()

    # Si no se aplicó ninguna regla específica, devolver valor limpio básico
    return valor


def _extraer_campo_individual(
    campo: str,
    vectorstore: Any,
    payload: Dict[str, Any],
    llm_provider: str,
    k: int = 5,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    contexto_extra: str = "",
) -> Dict[str, str]:
    """
    Extrae un campo individual usando RAG con retry automático.

    Esta función se ejecutará en paralelo para cada campo manual.

    Args:
        campo: Nombre del campo a extraer (ej: 'pantalla', 'procesador')
        vectorstore: Vector store FAISS con los documentos
        payload: Información del producto
        llm_provider: Proveedor de LLM
        k: Número de documentos a recuperar
        max_retries: Número máximo de reintentos en caso de error
        initial_delay: Delay inicial en segundos (se duplica en cada reintento)

    Returns:
        dict: {"campo": nombre_campo, "valor": valor_extraído}
    """
    campo_lower = campo.lower()

    # Búsqueda semántica (solo una vez, fuera del loop de retry)
    try:
        retrieved_docs = vectorstore.similarity_search(campo, k=k)
        if not retrieved_docs:
            logging.warning(f"  [Agente {campo}] No se encontraron documentos relevantes")
            return {"campo": campo, "valor": "No disponible"}
    except Exception as e:
        logging.error(f"  [Agente {campo}] Error en búsqueda: {e}")
        return {"campo": campo, "valor": "Error en búsqueda"}

    # Construir contexto
    contexto = "\n\n".join([
        f"Fragmento {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])

    # Las instrucciones específicas vienen del contexto pasado por el llamador
    instrucciones_especificas = []
    if contexto_extra:
        instrucciones_especificas.append(f"- {contexto_extra}")
    instrucciones_texto = "\n".join(instrucciones_especificas) if instrucciones_especificas else "- Extrae el valor exacto y limpio"

    # Construir prompt genérico y adaptativo
    prompt = f"""Eres un agente especializado en extraer información sobre **{campo}** de productos.

Producto:
- Categoría: {payload.get('Categoria', 'N/A')}
- Nombre: {payload.get('productoname', 'N/A')}
- Descripción Comprador: {payload.get('DescripcionProductoComprador', 'N/A')}
- Descripción Proveedor: {payload.get('DescripcionProductoProveedor', 'N/A')}

Contexto de documentos:
{contexto}

Tu tarea es extraer ÚNICAMENTE la información sobre **{campo}** del producto.

REGLAS GENERALES:
1. Responde SOLO con el valor extraído, SIN explicaciones ni texto adicional
2. FUENTE PRINCIPAL: El "Contexto de documentos" (fichas técnicas adjuntas) es la fuente autorizada. Las descripciones del producto sirven ÚNICAMENTE para identificar el producto correcto dentro de los adjuntos — NO son fuente de valores. Si la ficha técnica dice un procesador o RAM diferente a la descripción del proveedor, usa SIEMPRE el valor de la ficha técnica.
3. FALLBACK: Solo si la información NO aparece en los documentos adjuntos, puedes usar la "Descripción Proveedor" como referencia secundaria. Si tampoco está ahí, responde "No especificado".
4. NO inventes información — extrae SOLO lo que aparece explícitamente en los documentos
5. Sé PRECISO y conciso
6. Elimina caracteres especiales innecesarios
7. FOCO EN EL PRODUCTO PRINCIPAL: el contexto puede contener fichas de múltiples productos (el computador principal más accesorios como mouse, teclado, mochila, etc., o varios modelos juntos como un Notebook + un AIO). Extrae el atributo ÚNICAMENTE del producto principal identificado por la "Descripción Comprador" y "Descripción Proveedor" (si dice AIO, extrae del AIO; si dice Notebook, del Notebook). Ignora especificaciones de accesorios u otros modelos.

INSTRUCCIONES ESPECÍFICAS PARA ESTE CAMPO:
{instrucciones_texto}

Valor de {campo}:"""

    # Crear LLM una sola vez (fuera del retry loop) — temperatura baja para máxima precisión en extracción
    llm = get_llm(model_provider=llm_provider, temperature=0.1)

    # Retry loop
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logging.info(f"  [Agente {campo}] Intento {attempt + 1}/{max_retries}...")

            # Ejecutar LLM
            response = llm.invoke(prompt)
            valor = response.content.strip()

            # Limpiar valor con contexto del campo
            valor = _limpiar_valor(valor, campo)

            logging.info(f"  [Agente {campo}] ✓ Extracción completada: {valor[:50]}...")

            return {"campo": campo, "valor": valor}

        except Exception as e:
            error_msg = str(e)

            # Verificar si es un error de rate limit (429)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Extraer tiempo de espera del mensaje si está disponible
                    wait_time = delay

                    # Buscar "retry in X.Xs" o "Please retry in Xs"
                    import re
                    match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 1  # +1 seg de margen

                    logging.warning(f"  [Agente {campo}] Rate limit alcanzado. Esperando {wait_time:.1f}s antes de reintentar...")
                    time.sleep(wait_time)
                    delay *= 2  # Backoff exponencial
                    continue
                else:
                    logging.error(f"  [Agente {campo}] Rate limit - reintentos agotados")
                    return {"campo": campo, "valor": "No disponible (rate limit)"}
            else:
                # Otro tipo de error
                if attempt < max_retries - 1:
                    logging.warning(f"  [Agente {campo}] Error: {error_msg}. Reintentando en {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logging.error(f"  [Agente {campo}] Error después de {max_retries} intentos: {error_msg}")
                    return {"campo": campo, "valor": "Error en extracción"}

    # Si llegamos aquí, todos los intentos fallaron
    return {"campo": campo, "valor": "Error en extracción"}


def campos_manuales_node(state: CatalogacionState) -> CatalogacionState:
    """
    Nodo que ejecuta extracción paralela de campos manuales.

    Crea un agente especializado para cada campo en la lista campos_manuales_lista,
    ejecuta todos en paralelo usando RAG, y sintetiza los resultados en un JSON.

    Args:
        state: Estado actual del workflow

    Returns:
        CatalogacionState: Estado actualizado con resultado_campos_manuales

    Updates state:
        - resultado_campos_manuales: Diccionario con campos extraídos
        - errores: Lista de errores si hubo fallas
    """
    logging.info(f"[NODO 6] Extrayendo campos manuales (1 llamada LLM consolidada)")
    _NODO = "nodo_4_campos_manuales"
    t_nodo = time.time()

    # Normalizar: acepta List[str] o List[Dict{campo, contexto}]
    raw_lista = state.get('campos_manuales_lista') or []
    campos_manuales = [
        item if isinstance(item, dict) else {"campo": item, "contexto": ""}
        for item in raw_lista
    ]

    if not campos_manuales:
        logging.info("⊘ No hay campos manuales para extraer")
        state['resultado_campos_manuales'] = {}
        return state

    logging.info(f"Campos a extraer: {[c['campo'] for c in campos_manuales]}")

    try:
        texto_directo = state.get('texto_directo')
        chunks_ancla = ""  # Se rellena en modo adjuntos; vacío en modo texto

        if texto_directo:
            # --- MODO TEXTO: sin FAISS, el texto completo es el contexto ---
            logging.info("✓ Modo texto_directo — saltando FAISS, usando texto como contexto")
            state = add_tiempo(state, _NODO, "faiss", "FAISS omitido (modo texto)", 0.0)

            t0 = time.time()
            campos_con_docs = []
            for item in campos_manuales:
                campos_con_docs.append({
                    "campo": item["campo"],
                    "contexto": item.get("contexto", ""),
                    "fragmentos": texto_directo,
                })
            state = add_tiempo(state, _NODO, "searches", "Contexto directo (sin similarity search)", time.time() - t0)

        else:
            # --- MODO ADJUNTOS: flujo original con FAISS ---
            t0 = time.time()
            vectorstore = state.get('adjuntos_vectorstore')

            if vectorstore is not None:
                logging.info(f"✓ Reutilizando FAISS compartido de nodo_3 (fallback o adjuntos)")
            else:
                # Sin vectorstore precargado: construir desde archivos procesados
                processed_path = state.get('processed_path')
                if not processed_path or not os.path.exists(processed_path):
                    error_msg = "No se encontró el directorio de archivos procesados"
                    state = add_error(state, error_msg)
                    logging.error(error_msg)
                    return state

                txt_files = [
                    os.path.join(processed_path, f)
                    for f in os.listdir(processed_path)
                    if f.endswith('.txt') and f != 'skipped_files.txt'
                ]

                if not txt_files:
                    error_msg = f"No se encontraron archivos procesados en {processed_path}"
                    state = add_error(state, error_msg)
                    logging.error(error_msg)
                    return state

                metadatas = [
                    {
                        "codigo_cotizacion": state['codigo_cotizacion'],
                        "rut_proveedor": state['rut_proveedor'],
                        "source_file": os.path.basename(f)
                    }
                    for f in txt_files
                ]
                chunk_size = int(os.getenv('CAMPOS_MANUALES_CHUNK_SIZE', '200'))
                chunk_overlap = int(os.getenv('CAMPOS_MANUALES_CHUNK_OVERLAP', '50'))
                vectorstore = create_faiss_from_files(
                    txt_files, metadatas,
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                logging.info(f"✓ FAISS creado ({len(txt_files)} archivos, chunk={chunk_size})")
            state = add_tiempo(state, _NODO, "faiss", "FAISS listo", time.time() - t0)

            # Sub-fase 4b: búsqueda de anclaje + similarity_search por campo
            t0 = time.time()
            k = int(os.getenv('CAMPOS_MANUALES_SEARCH_K', '3'))
            payload_tmp = state.get('payload', {})
            desc_proveedor = payload_tmp.get('DescripcionProductoProveedor', '')
            desc_comprador = payload_tmp.get('DescripcionProductoComprador', '')
            nombre_producto = payload_tmp.get('productoname', '')
            producto_contexto = " ".join(filter(None, [nombre_producto, desc_proveedor, desc_comprador]))[:300]

            # Búsqueda de anclaje: chunks que identifican ESTE producto específico
            # Se usa para construir un mini-FAISS con solo los chunks del producto correcto
            chunks_ancla = ""
            search_store = vectorstore  # fallback: FAISS global
            mejor_score_ancla = float('inf')
            try:
                k_ancla = int(os.getenv('CAMPOS_MANUALES_ANCLA_K', '8'))
                ancla_with_scores = vectorstore.similarity_search_with_score(producto_contexto, k=k_ancla)
                docs_ancla = [doc for doc, _ in ancla_with_scores]
                ancla_scores = [score for _, score in ancla_with_scores]
                if ancla_scores:
                    mejor_score_ancla = min(ancla_scores)
                if docs_ancla:
                    chunks_ancla = "\n\n".join([
                        f"  Fragmento {i+1}:\n  {doc.page_content}"
                        for i, doc in enumerate(docs_ancla)
                    ])
                    # Mini-FAISS con solo los chunks del producto: las búsquedas por campo
                    # ya no ven specs de otros productos del mismo documento
                    ancla_texts = [d.page_content for d in docs_ancla]
                    ancla_metas = [d.metadata for d in docs_ancla]
                    search_store = create_faiss_from_texts(ancla_texts, ancla_metas)
                    logging.info(f"  Anclaje: mini-FAISS creado con {len(docs_ancla)} chunks (mejor L2={mejor_score_ancla:.3f})")
            except Exception as e:
                logging.warning(f"  Error en búsqueda de anclaje, usando FAISS global: {e}")

            # Señal de baja relevancia: adjuntos no contienen specs del producto
            _diag_now = state.get('diagnostico') or {}
            _rag_modo = (_diag_now.get('rag') or {}).get('modo', '')
            _L2_THRESHOLD = float(os.getenv('ADJUNTOS_RELEVANCE_L2_THRESHOLD', '1.2'))
            _baja_relevancia = (_rag_modo == 'adjuntos') and (mejor_score_ancla > _L2_THRESHOLD)
            if _baja_relevancia:
                _warn = (
                    f"Los adjuntos no contienen especificaciones técnicas del producto "
                    f"(mejor similitud L2={mejor_score_ancla:.2f} > umbral {_L2_THRESHOLD}). "
                    f"Puede ser un contrato genérico o de suministro sin detalle técnico."
                )
                state = add_warning(state, _warn)
                logging.warning(f"⚠️  {_warn}")
            _rag_diag = (_diag_now.get('rag') or {})
            _rag_diag['mejor_score_ancla_l2'] = round(mejor_score_ancla, 3) if mejor_score_ancla != float('inf') else None
            _rag_diag['baja_relevancia'] = _baja_relevancia
            _diag_now['rag'] = _rag_diag
            state['diagnostico'] = _diag_now

            campos_con_docs = []
            for item in campos_manuales:
                campo = item["campo"]
                contexto_extra = item.get("contexto", "")
                k_campo = k + 1 if any(kw in campo.lower() for kw in ["procesador", "cpu", "processor"]) else k
                # Búsqueda sobre el mini-FAISS del producto (o FAISS global si falló el anclaje)
                k_campo = min(k_campo, len(docs_ancla) if search_store is not vectorstore else k_campo)
                try:
                    docs = search_store.similarity_search(campo, k=k_campo)
                    fragmentos = "\n\n".join([
                        f"  Fragmento {i+1}:\n  {doc.page_content}"
                        for i, doc in enumerate(docs)
                    ]) if docs else "  (sin documentos relevantes encontrados)"
                except Exception as e:
                    logging.warning(f"  Error buscando '{campo}': {e}")
                    fragmentos = "  (error en búsqueda)"
                campos_con_docs.append({
                    "campo": campo,
                    "contexto": contexto_extra,
                    "fragmentos": fragmentos,
                })
            logging.info(f"  {len(campos_con_docs)} campos con documentos RAG del producto")
            state = add_tiempo(state, _NODO, "searches", "Anclaje + mini-FAISS + searches", time.time() - t0)

        # Sub-fase 4c: 1 sola llamada LLM con contexto por campo
        t0 = time.time()
        payload = state.get('payload')
        llm_provider = state.get('llm_provider') or os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')

        campos_json_template = ", ".join([f'"{item["campo"]}": "..."' for item in campos_manuales])

        secciones_campos = "\n\n".join([
            f'CAMPO: {c["campo"]}\n'
            f'INSTRUCCIONES: {c["contexto"] or "Extrae el valor exacto y limpio"}\n'
            f'DOCUMENTOS RELEVANTES:\n{c["fragmentos"]}'
            for c in campos_con_docs
        ])

        seccion_ancla = f"""--- SECCIÓN DEL DOCUMENTO QUE CORRESPONDE A ESTE PRODUCTO ---
(Usa estos fragmentos para identificar con certeza de qué producto se habla en los documentos)

{chunks_ancla}

""" if chunks_ancla else ""

        prompt = f"""Eres un experto en catalogación de productos tecnológicos. Para cada campo se entregan sus documentos más relevantes de la ficha técnica. Extrae el valor de cada campo usando SOLO sus propios documentos.

PRODUCTO A CATALOGAR:
- Categoría: {payload.get('Categoria', 'N/A')}
- Nombre: {payload.get('productoname', 'N/A')}
- Descripción Comprador: {payload.get('DescripcionProductoComprador', 'N/A')}
- Descripción Proveedor: {payload.get('DescripcionProductoProveedor', 'N/A')}

{seccion_ancla}REGLAS GENERALES:
1. El "PRODUCTO A CATALOGAR" y la "SECCIÓN DEL DOCUMENTO" identifican exactamente cuál producto catalogar.
2. Si los documentos contienen múltiples productos, extrae ÚNICAMENTE los valores del producto identificado arriba. Ignora los demás.
3. La ficha técnica es la fuente principal de valores. Extrae SOLO lo que aparece explícitamente.
4. Si un valor no aparece en los documentos del campo, usa "No especificado".

--- CAMPOS A EXTRAER ---

{secciones_campos}

--- OUTPUT ---
Responde ÚNICAMENTE con un JSON válido, sin explicaciones ni markdown:
{{{campos_json_template}}}"""

        llm = get_llm(model_provider=llm_provider, temperature=0.1)
        max_retries = int(os.getenv('CAMPOS_MANUALES_MAX_RETRIES', '3'))
        delay = float(os.getenv('CAMPOS_MANUALES_INITIAL_DELAY', '2.0'))
        resultado_json = {}

        for attempt in range(max_retries):
            try:
                with _llm_semaphore:
                    response = llm.invoke(prompt)
                content = response.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                resultado_json = json.loads(content)
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    # Extraer tiempo real de espera del mensaje de OpenAI ("retry after Xs" o "Please retry in Xs")
                    match = re.search(r'(?:retry after|retry in|please retry in)\s*(\d+(?:\.\d+)?)', error_msg, re.IGNORECASE)
                    wait = float(match.group(1)) + 2 if match else max(delay, 30)
                    logging.warning(f"  Rate limit (429) intento {attempt+1}/{max_retries} — esperando {wait:.0f}s...")
                    time.sleep(wait)
                    delay *= 2
                    continue
                logging.error(f"  Error LLM intento {attempt+1}: {error_msg}")
                if attempt == max_retries - 1:
                    resultado_json = {item["campo"]: "Error en extracción" for item in campos_manuales}

        # Aplicar _limpiar_valor por campo
        resultado_final = {}
        for item in campos_manuales:
            campo = item["campo"]
            valor = resultado_json.get(campo, "No especificado")
            resultado_final[campo] = _limpiar_valor(str(valor), campo)

        state = add_tiempo(state, _NODO, "llm", "1 llamada LLM consolidada", time.time() - t0)
        logging.info(f"✓ Extracción completada ({len(resultado_final)} campos, 1 llamada LLM):")
        for campo, valor in resultado_final.items():
            logging.info(f"  - {campo}: {str(valor)[:50]}")

        state['resultado_campos_manuales'] = resultado_final

    except Exception as e:
        error_msg = f"Error en extracción de campos manuales: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)
        state['resultado_campos_manuales'] = {}

    state = add_tiempo(state, _NODO, "total", "Total nodo campos_manuales", time.time() - t_nodo)
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
        # Sin adjuntos: salta directo a rag_adjuntos, que usa la descripción del payload como fallback
        return "rag_adjuntos"


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
    use_dic = state.get('use_diccionarios', True)
    has_result = state.get('resultado_adjuntos') is not None

    logging.info(f"[DECISIÓN] ¿Usar diccionarios?")
    logging.info(f"  - use_diccionarios: {use_dic}")
    logging.info(f"  - tiene resultado_adjuntos: {has_result}")

    if use_dic and has_result:
        logging.info(f"  → Ir a: rag_diccionarios")
        return "rag_diccionarios"
    else:
        logging.info(f"  → Ir a: consolidar_resultado")
        return "consolidar_resultado"


def should_extract_campos_manuales(state: CatalogacionState) -> str:
    """
    Determina si se deben extraer campos manuales.

    Args:
        state: Estado actual

    Returns:
        str: Nombre del siguiente nodo
    """
    campos_manuales = state.get('campos_manuales_lista', [])

    # Verificar si el tipo de producto es "Otro"
    resultado_adjuntos = state.get('resultado_adjuntos') or {}
    tipo_producto = resultado_adjuntos.get('Tipo', '').strip()

    # Si el tipo es "Otro", saltar campos manuales Y diccionarios
    if tipo_producto.lower() == 'otro':
        logging.info("Tipo de producto es 'Otro' - saltando extracción de campos manuales y RAG diccionarios")
        return "consolidar_resultado"

    # Si hay campos manuales y hay contexto disponible (adjuntos procesados, vectorstore, o descripción del payload)
    has_context = (state.get('adjuntos_procesados')
                   or state.get('adjuntos_vectorstore') is not None
                   or state.get('resultado_adjuntos') is not None)
    if campos_manuales and len(campos_manuales) > 0 and has_context:
        return "campos_manuales"
    else:
        # Si NO hay campos manuales, ir directo a consolidar (sin diccionarios)
        # Los diccionarios solo se aplican DESPUÉS de extraer campos manuales
        return "consolidar_resultado"
