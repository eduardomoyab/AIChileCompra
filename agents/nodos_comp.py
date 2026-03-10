"""
Nodes for the Computadores catalogation workflow using LangGraph.

Each node is a function that takes the state and returns an updated state.
Nodes wrap existing functionality from utils and agents modules.
"""

import os
import sys
import logging
import time
import asyncio
import re
import shutil
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state_comp import CatalogacionState, add_error, add_warning, add_tiempo
from agents.catalogacion_comp import CatalogacionComputadores
from agents.get_agent import get_llm
from agents.get_vectorstore import create_faiss_from_files
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
        cache_hit = False
        if os.path.exists(attachments_dir):
            existing_files = [f for f in os.listdir(attachments_dir)
                            if os.path.isfile(os.path.join(attachments_dir, f))]
            if len(existing_files) > 0:
                cache_hit = True
        state = add_tiempo(state, _NODO, "check_cache",
                           f"Verificar caché local de adjuntos (hit={cache_hit})",
                           time.time() - t0)

        if cache_hit:
            logging.info(f"✓ Archivos ya descargados: {len(existing_files)} archivos encontrados")
            state['adjuntos_descargados'] = True
            state['adjuntos_path'] = attachments_dir
            state = add_tiempo(state, _NODO, "total",
                               "Total nodo descargar_adjuntos (caché)",
                               time.time() - t_nodo)
            return state

        # Sub-fase 1b: descarga real
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
    _NODO = "nodo_3_rag_adjuntos"
    t_nodo = time.time()

    try:
        # Sub-fase 3a: instanciar catalogador (carga configs y diccionario Excel)
        t0 = time.time()
        catalogador = CatalogacionComputadores(llm_provider=state.get('llm_provider'))
        state = add_tiempo(state, _NODO, "init_catalogador",
                           "Instanciar CatalogacionComputadores (cargar YAML + Excel)",
                           time.time() - t0)

        # Sub-fase 3b: inicializar LLMs
        t0 = time.time()
        catalogador._initialize_llms()
        state = add_tiempo(state, _NODO, "init_llm",
                           f"Inicializar LLM ({catalogador.llm_provider})",
                           time.time() - t0)

        # Sub-fases 3c..3f: dentro de catalogar_producto
        tiempos_locales = state.get('tiempos', [])
        resultado = catalogador.catalogar_producto(
            payload=state['payload'],
            codigo_cotizacion=state['codigo_cotizacion'],
            rut_proveedor=state['rut_proveedor'],
            processed_path=state['processed_path'],
            use_diccionarios=False,
            tiempos=tiempos_locales,
            nodo_nombre=_NODO,
        )
        # tiempos_locales fue mutado in-place por catalogar_producto
        state['tiempos'] = tiempos_locales

        state['resultado_adjuntos'] = resultado
        if not state.get('use_diccionarios', True):
            state['resultado_final'] = resultado
        logging.info(f"✓ RAG adjuntos completado para ROWNUM {state['payload'].get('ROWNUM')}")

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
    _NODO = "nodo_5_rag_diccionarios"
    t_nodo = time.time()

    if not state.get('use_diccionarios', True):
        state['resultado_final'] = state.get('resultado_adjuntos')
        logging.info("⊘ Diccionarios deshabilitados, usando resultado de adjuntos")
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
        catalogador = CatalogacionComputadores(llm_provider=state.get('llm_provider'))
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
        resultado = catalogador.aplicar_diccionarios(
            resultado_adjuntos=resultado_previo,
            tiempos=tiempos_locales,
            nodo_nombre=_NODO,
        )
        state['tiempos'] = tiempos_locales

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
    initial_delay: float = 2.0
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

    # Construir query semántica enriquecida con contexto del producto para campos clave
    query_busqueda = campo
    nombre_producto = (
        payload.get("productoname") or
        payload.get("DescripcionProductoComprador") or
        ""
    )
    desc_proveedor = payload.get("DescripcionProductoProveedor") or ""

    if any(kw in campo_lower for kw in ["marca", "fabricante", "manufacturer"]):
        # Para marca: usar el nombre del producto (la marca suele estar implícita en el modelo)
        query_busqueda = f"marca fabricante {nombre_producto[:100]}"
    elif any(kw in campo_lower for kw in ["procesador", "cpu", "processor"]):
        # Para procesador: usar keywords técnicos de marcas + producto para localizar el chunk correcto
        query_busqueda = f"procesador Intel AMD Apple Core Ryzen GHz {nombre_producto[:50]}"

    # Búsqueda semántica (solo una vez, fuera del loop de retry)
    try:
        retrieved_docs = vectorstore.similarity_search(query_busqueda, k=k)
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

    # Determinar instrucciones específicas según patrones en el nombre del campo
    instrucciones_especificas = []

    if "(gb)" in campo_lower or campo_lower.endswith(" gb"):
        instrucciones_especificas.append("- Responde SOLO el número seguido de 'GB' (ej: '16 GB', '512 GB')")
        instrucciones_especificas.append("- Si encuentras TB, conviértelo a GB (1 TB = 1000 GB)")
    elif "(tb)" in campo_lower or campo_lower.endswith(" tb"):
        instrucciones_especificas.append("- Responde SOLO el número seguido de 'TB' (ej: '1 TB', '2 TB')")
    elif "(pulgada" in campo_lower or "(inch" in campo_lower:
        instrucciones_especificas.append("- Responde SOLO el número, sin unidades (ej: '13.3', '15.6')")
    elif "(mhz)" in campo_lower or "(ghz)" in campo_lower or "(w)" in campo_lower:
        # Detectar cualquier unidad entre paréntesis
        unit_match = re.search(r'\(([A-Za-z]+)\)', campo)
        if unit_match:
            unit = unit_match.group(1)
            instrucciones_especificas.append(f"- Responde SOLO el número seguido de '{unit}' (ej: '3.5 {unit}')")
    elif "tipo" in campo_lower:
        instrucciones_especificas.append("- Responde SOLO el TIPO o tecnología, SIN capacidades ni números")
        instrucciones_especificas.append("- Elimina cantidades como GB, TB, MHz, etc. de tu respuesta")
        if "ram" in campo_lower:
            instrucciones_especificas.append("- Ejemplo de respuesta correcta: 'DDR4', 'DDR5', 'LPDDR4', 'LPDDR4X', 'LPDDR5', 'LPDDR5X'")
            instrucciones_especificas.append("- Para productos Apple con memoria unificada, responde 'Memoria Unificada'")
            instrucciones_especificas.append("- CRÍTICO: Si el documento NO menciona explícitamente el tipo de RAM, responde 'No disponible'. NUNCA lo deduzcas de la capacidad (GB), del procesador ni del modelo.")
            instrucciones_especificas.append("- Copia EXACTAMENTE la denominación con su sufijo: LPDDR4X ≠ LPDDR4, LPDDR5X ≠ LPDDR5. Si ves el sufijo X en el documento, inclúyelo.")
        elif "almacenamiento" in campo_lower or "disco" in campo_lower:
            instrucciones_especificas.append("- Ejemplo de respuesta correcta: 'SSD M.2', 'SSD NVMe', 'HDD SATA'")
    elif any(kw in campo_lower for kw in ["nucleo", "hilo", "core", "thread"]):
        instrucciones_especificas.append("- Responde SOLO el número (ej: '8', '16')")
    elif any(kw in campo_lower for kw in ["marca", "fabricante", "manufacturer"]):
        instrucciones_especificas.append("- Responde SOLO el nombre de la marca (ej: 'HP', 'Lenovo', 'Dell', 'Acer')")
        instrucciones_especificas.append("- La marca puede aparecer explícita ('HP', 'Lenovo') o implícita en el nombre o número de modelo del producto")
        instrucciones_especificas.append("- Usa tu conocimiento sobre computadores para inferir la marca a partir del nombre del modelo si no aparece escrita directamente")
        instrucciones_especificas.append("- Si no hay información suficiente, responde 'No especificado'")
    elif any(kw in campo_lower for kw in ["procesador", "cpu", "processor"]):
        instrucciones_especificas.append("- Responde SOLO el nombre estándar del modelo de procesador (ej: 'AMD Ryzen 7 7735U', 'Intel Core i7-1355U', 'Apple M3')")
        instrucciones_especificas.append("- Copia el número de modelo EXACTAMENTE: todos sus dígitos y letras (ej: '12450H', '1235U', '5700U'). NO aproximes ni sustituyas.")
        instrucciones_especificas.append("- NO agregues información extra: velocidades de reloj (GHz/MHz), número de núcleos, generación en texto, ni descripciones adicionales")
        instrucciones_especificas.append("- Formato correcto: 'Intel Core i7-1260P' (NO 'Intel Core i7-1260P de 12ª Generación' ni 'i7-1260P 2.1GHz 12-Core')")
        instrucciones_especificas.append("- Incluye siempre la palabra 'Core' para Intel si corresponde (ej: 'Intel Core i5' no 'Intel i5')")
        instrucciones_especificas.append("- Si el contexto menciona el procesador en varios formatos, elige la versión con el número de modelo más completo y específico (ej: 'Intel Core i7-1355U' sobre 'Intel Core i7 de 13ª Gen')")
    elif "sistema operativo" in campo_lower or ("sistema" in campo_lower and "operativo" in campo_lower):
        instrucciones_especificas.append("- Responde con el nombre EXACTO y COMPLETO del sistema operativo instalado (ej: 'Microsoft Windows 11 Pro', 'macOS', 'FreeDOS')")
        instrucciones_especificas.append("- Selecciona UNA sola versión y edición específica — NO listes varias opciones ni combines múltiples")
        instrucciones_especificas.append("- Si el documento menciona Windows, identifica: ¿versión 10 u 11? ¿edición Home, Pro, Enterprise, SE? Extrae lo que esté en las especificaciones del producto")
        instrucciones_especificas.append("- Para productos Apple, responde 'macOS' salvo que indique versión específica (ej: 'macOS Ventura')")
        instrucciones_especificas.append("- Para sistemas distintos de Windows o macOS (FreeDOS, Ubuntu, Chrome OS, Linux, etc.), extrae EXACTAMENTE lo que aparece en el documento")
        instrucciones_especificas.append("- CRÍTICO: Si los documentos adjuntos NO mencionan ningún SO instalado, responde 'No especificado'. NUNCA asumas el SO por la marca, modelo o tipo de equipo.")
    elif any(kw in campo_lower for kw in ["modelo", "model"]) and "procesador" not in campo_lower:
        instrucciones_especificas.append("- Responde el modelo o nombre específico del producto")
    elif "modalidad" in campo_lower:
        instrucciones_especificas.append("- Busca si el contexto o las descripciones mencionan 'arriendo', 'arrendamiento' o 'leasing'")
        instrucciones_especificas.append("- Si encuentras esas palabras, responde exactamente: 'Arriendo'")
        instrucciones_especificas.append("- Si no se menciona ninguna modalidad o se habla de 'compra', 'adquisición' o 'compra ágil', responde exactamente: 'Compra'")
        instrucciones_especificas.append("- Solo responde 'Arriendo' o 'Compra', nada más")
    else:
        instrucciones_especificas.append("- Extrae el valor exacto tal como aparece en el contexto")

    # Construir sección de instrucciones específicas
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
    logging.info(f"[NODO 6] Extrayendo campos manuales en paralelo")
    _NODO = "nodo_4_campos_manuales"
    t_nodo = time.time()

    campos_manuales = state.get('campos_manuales_lista', [])

    if not campos_manuales:
        logging.info("⊘ No hay campos manuales para extraer")
        state['resultado_campos_manuales'] = {}
        return state

    logging.info(f"Campos a extraer: {campos_manuales}")

    try:
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

        # Sub-fase 4a: crear FAISS para campos manuales
        t0 = time.time()
        logging.info(f"Creando FAISS en memoria con {len(txt_files)} archivos...")
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
        logging.info(f"Creando FAISS con chunks de {chunk_size} caracteres (overlap: {chunk_overlap})...")
        vectorstore = create_faiss_from_files(
            txt_files, metadatas,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        state = add_tiempo(state, _NODO, "crear_faiss",
                           f"Crear FAISS para campos manuales ({len(txt_files)} archivos,"
                           f" chunk={chunk_size})",
                           time.time() - t0)
        logging.info(f"✓ FAISS creado con chunking optimizado para campos manuales")

        # Sub-fase 4b: extracción paralela
        t0 = time.time()
        llm_provider = state.get('llm_provider') or os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')
        k = int(os.getenv('CAMPOS_MANUALES_SEARCH_K', '3'))
        payload = state.get('payload')
        max_workers_config = int(os.getenv('CAMPOS_MANUALES_MAX_WORKERS', '9'))
        max_workers = min(max_workers_config, len(campos_manuales))
        logging.info(f"Ejecutando {len(campos_manuales)} agentes (máx {max_workers} en paralelo)...")

        resultados = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, campo in enumerate(campos_manuales):
                max_retries = int(os.getenv('CAMPOS_MANUALES_MAX_RETRIES', '3'))
                initial_delay = float(os.getenv('CAMPOS_MANUALES_INITIAL_DELAY', '2.0'))
                # Procesador necesita más documentos de referencia para localizar el modelo exacto
                campo_lower_iter = campo.lower()
                k_campo = k + 1 if any(kw in campo_lower_iter for kw in ["procesador", "cpu", "processor"]) else k
                future = executor.submit(
                    _extraer_campo_individual,
                    campo, vectorstore, payload, llm_provider,
                    k_campo, max_retries, initial_delay
                )
                futures[future] = campo

            for future in as_completed(futures):
                campo = futures[future]
                try:
                    resultado = future.result()
                    resultados.append(resultado)
                except Exception as e:
                    logging.error(f"Error en agente {campo}: {e}")
                    resultados.append({"campo": campo, "valor": f"Error: {str(e)}"})

        state = add_tiempo(state, _NODO, "extraccion_paralela",
                           f"Extracción paralela de {len(campos_manuales)} campos"
                           f" ({max_workers} workers)",
                           time.time() - t0)

        resultado_json = {r["campo"]: r["valor"] for r in resultados}
        logging.info(f"✓ Extracción paralela completada:")
        for campo, valor in resultado_json.items():
            logging.info(f"  - {campo}: {valor[:50]}...")

        state['resultado_campos_manuales'] = resultado_json

    except Exception as e:
        error_msg = f"Error en extracción de campos manuales: {str(e)}"
        state = add_error(state, error_msg)
        logging.exception(error_msg)
        state['resultado_campos_manuales'] = {}

    state = add_tiempo(state, _NODO, "total",
                       "Total nodo campos_manuales",
                       time.time() - t_nodo)
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
    resultado_adjuntos = state.get('resultado_adjuntos', {})
    tipo_producto = resultado_adjuntos.get('Tipo', '').strip()

    # Si el tipo es "Otro", saltar campos manuales Y diccionarios
    if tipo_producto.lower() == 'otro':
        logging.info("Tipo de producto es 'Otro' - saltando extracción de campos manuales y RAG diccionarios")
        return "consolidar_resultado"

    # Si hay campos manuales y se procesaron adjuntos, ejecutar extracción paralela
    if campos_manuales and len(campos_manuales) > 0 and state.get('adjuntos_procesados'):
        return "campos_manuales"
    else:
        # Si NO hay campos manuales, ir directo a consolidar (sin diccionarios)
        # Los diccionarios solo se aplican DESPUÉS de extraer campos manuales
        return "consolidar_resultado"
