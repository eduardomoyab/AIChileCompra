"""
Extracción de atributos de productos directamente desde texto.

========================================================
  SIN RAG — SIN EMBEDDINGS — SIN FAISS
========================================================

El texto del producto (nombre + descripción + atributos) se pasa
directamente al LLM como contexto del prompt, sin necesidad de
crear vectorstores ni llamar a la API de embeddings de OpenAI.

Esto es posible porque el input de este endpoint es corto (~400-800
tokens) y cabe íntegro en el contexto del LLM, a diferencia del
flujo con adjuntos (PDFs grandes) donde el RAG sí es necesario.

Categorías soportadas: Computadores
"""

import os
import sys
import re
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extraer_atributos_texto(
    nombre_producto: str,
    descripcion: str = None,
    atributos: Dict[str, Any] = None,
    categoria: str = "Computadores",
    use_diccionarios: bool = True,
    llm_provider: str = None,
    campos_manuales: List[str] = None
) -> Dict[str, Any]:
    """
    Extrae y normaliza atributos de un producto a partir de texto directo.

    *** SIN RAG: el texto se pasa directamente al LLM, sin embeddings ni FAISS ***

    Reemplaza el flujo de descarga + OCR de adjuntos por texto provisto directamente.
    El output sigue el mismo formato y normalización que el endpoint /catalogar.

    Args:
        nombre_producto: Nombre del producto (ej: "Laptop HP Pavilion 15-EH1005LA")
        descripcion: Descripción libre del producto
        atributos: Tabla de atributos en formato {atributo: valor}
                   (ej: {"Procesador": "AMD Ryzen 7 5700U", "RAM": "8 GB DDR4"})
        categoria: Categoría del producto. Solo "Computadores" soportado.
        use_diccionarios: Si aplicar normalización con diccionarios técnicos (default True)
        llm_provider: Proveedor LLM ('openai', 'gemini'). None = usa DEFAULT_LLM_PROVIDER del .env
        campos_manuales: Lista de campos adicionales a extraer en paralelo
                         (ej: ['Procesador', 'RAM (GB)', 'Tipo RAM', 'Almacenamiento (GB)'])

    Returns:
        dict con la misma estructura que extraer_atributos():
            {
                'resultado_final': {
                    'ROWNUM': str,
                    'Tipo': str,          # AIO / Laptop / Desktop / Otro
                    'Part Number': str,
                    'Modelo': str,
                    # + campos normalizados si use_diccionarios=True
                    # + campos_manuales si se solicitaron
                },
                'errores': list[str],
                'warnings': list[str]
            }

    Raises:
        ValueError: Si la categoría no es soportada o falta nombre_producto
    """
    if not nombre_producto or not nombre_producto.strip():
        raise ValueError("nombre_producto es requerido y no puede estar vacío")

    if categoria != 'Computadores':
        raise ValueError(
            f"Categoría '{categoria}' no soportada. "
            f"Categorías disponibles: ['Computadores']"
        )

    errores = []
    warnings = []
    rownum = str(uuid.uuid4())[:8]

    logging.info(f"\n{'='*80}")
    logging.info(f"EXTRACCIÓN DESDE TEXTO DIRECTO  [SIN RAG / SIN EMBEDDINGS]")
    logging.info(f"Categoría: {categoria}")
    logging.info(f"Producto: {nombre_producto}")
    logging.info(f"use_diccionarios: {use_diccionarios}")
    logging.info(f"campos_manuales: {campos_manuales}")
    logging.info(f"{'='*80}\n")

    # Construir texto combinado para el contexto del LLM
    partes = [f"Nombre del producto: {nombre_producto}"]
    if descripcion:
        partes.append(f"\nDescripción:\n{descripcion}")
    if atributos:
        partes.append(
            f"\nEspecificaciones técnicas:\n"
            f"{json.dumps(atributos, indent=2, ensure_ascii=False)}"
        )
    texto_completo = "\n".join(partes)
    logging.info(f"Contexto para LLM ({len(texto_completo)} chars):\n{texto_completo[:400]}...")

    # Payload compatible con CatalogacionComputadores
    payload = {
        'ROWNUM': rownum,
        'Categoria': categoria,
        'DescripcionProductoComprador': nombre_producto,
        'DescripcionProductoProveedor': descripcion or nombre_producto,
        'productoname': nombre_producto
    }

    try:
        from agents.catalogacion_comp import CatalogacionComputadores

        catalogador = CatalogacionComputadores(llm_provider=llm_provider)

        # Paso 1: Extraer Tipo, Part Number, Modelo
        # SIN RAG: el texto completo va directo al LLM como contexto
        resultado = catalogador.catalogar_desde_texto_sin_rag(
            payload=payload,
            texto_completo=texto_completo
        )

        if not resultado:
            errores.append("No se pudo extraer atributos del texto proporcionado")
            return {'resultado_final': None, 'errores': errores, 'warnings': warnings}

        tipo = resultado.get('Tipo', '').strip().lower()

        # Paso 2 (opcional): Extraer campos manuales en paralelo
        # SIN RAG: se usa el texto completo directamente
        resultado_campos_manuales = {}
        if campos_manuales and len(campos_manuales) > 0 and tipo != 'otro':
            logging.info(f"Extrayendo {len(campos_manuales)} campos manuales [SIN RAG]...")
            resultado_campos_manuales = _extraer_campos_manuales_sin_rag(
                texto_completo=texto_completo,
                campos=campos_manuales,
                payload=payload,
                llm_provider=llm_provider
            )
        elif tipo == 'otro' and campos_manuales:
            warnings.append(
                "Tipo detectado como 'Otro' — se omitió extracción de campos manuales y diccionarios"
            )

        # Mezclar campos manuales antes de aplicar diccionarios
        resultado_previo = dict(resultado)
        for campo, valor in resultado_campos_manuales.items():
            resultado_previo[campo] = valor

        # Paso 3 (opcional): Normalizar con diccionarios (este sí usa FAISS, pero está cacheado)
        if use_diccionarios and tipo != 'otro':
            logging.info("Aplicando normalización con diccionarios...")
            resultado_final = catalogador.aplicar_diccionarios(resultado_previo)

            # Rellenar campos manuales que diccionarios no completaron
            for campo, valor in resultado_campos_manuales.items():
                valor_actual = resultado_final.get(campo)
                if not valor_actual or valor_actual in ['No disponible', 'No especificado']:
                    if valor not in ['No disponible', 'No especificado']:
                        resultado_final[campo] = valor
        else:
            resultado_final = resultado_previo

        logging.info(f"✓ Extracción desde texto completada [sin RAG]")
        return {
            'resultado_final': resultado_final,
            'errores': errores,
            'warnings': warnings
        }

    except Exception as e:
        logging.exception(f"Error extrayendo atributos desde texto: {e}")
        errores.append(str(e))
        return {'resultado_final': None, 'errores': errores, 'warnings': warnings}


def _extraer_campos_manuales_sin_rag(
    texto_completo: str,
    campos: List[str],
    payload: Dict[str, Any],
    llm_provider: str = None
) -> Dict[str, str]:
    """
    Extrae campos manuales en paralelo pasando el texto completo al LLM.

    SIN RAG: no crea FAISS ni llama a la API de embeddings.
    El texto_completo se pasa directamente como contexto del prompt.
    """
    provider = llm_provider or os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')
    max_workers_config = int(os.getenv('CAMPOS_MANUALES_MAX_WORKERS', '3'))
    max_workers = min(max_workers_config, len(campos))
    max_retries = int(os.getenv('CAMPOS_MANUALES_MAX_RETRIES', '3'))
    initial_delay = float(os.getenv('CAMPOS_MANUALES_INITIAL_DELAY', '2.0'))

    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, campo in enumerate(campos):
            if i > 0:
                time.sleep(0.5)
            future = executor.submit(
                _extraer_campo_directo,
                campo,
                texto_completo,
                payload,
                provider,
                max_retries,
                initial_delay
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

    resultado_json = {r["campo"]: r["valor"] for r in resultados}
    logging.info(f"✓ Campos manuales extraídos [sin RAG]: {list(resultado_json.keys())}")
    return resultado_json


def _extraer_campo_directo(
    campo: str,
    texto_completo: str,
    payload: Dict[str, Any],
    llm_provider: str,
    max_retries: int = 3,
    initial_delay: float = 2.0
) -> Dict[str, str]:
    """
    Extrae un campo individual usando el texto completo como contexto, SIN RAG.

    Replica la lógica de instrucciones específicas de _extraer_campo_individual
    (nodos_comp.py) pero sin vectorstore: el texto ya es el contexto completo.
    """
    from agents.get_agent import get_llm
    from agents.nodos_comp import _limpiar_valor

    campo_lower = campo.lower()
    instrucciones_especificas = []

    if "(gb)" in campo_lower or campo_lower.endswith(" gb"):
        instrucciones_especificas.append("- Responde SOLO el número seguido de 'GB' (ej: '16 GB', '512 GB')")
        instrucciones_especificas.append("- Si encuentras TB, conviértelo a GB (1 TB = 1000 GB)")
    elif "(tb)" in campo_lower or campo_lower.endswith(" tb"):
        instrucciones_especificas.append("- Responde SOLO el número seguido de 'TB' (ej: '1 TB', '2 TB')")
    elif "(pulgada" in campo_lower or "(inch" in campo_lower:
        instrucciones_especificas.append("- Responde SOLO el número, sin unidades (ej: '13.3', '15.6')")
    elif "(mhz)" in campo_lower or "(ghz)" in campo_lower or "(w)" in campo_lower:
        unit_match = re.search(r'\(([A-Za-z]+)\)', campo)
        if unit_match:
            unit = unit_match.group(1)
            instrucciones_especificas.append(
                f"- Responde SOLO el número seguido de '{unit}' (ej: '3.5 {unit}')"
            )
    elif "tipo" in campo_lower:
        instrucciones_especificas.append(
            "- Responde SOLO el TIPO o tecnología, SIN capacidades ni números"
        )
        instrucciones_especificas.append("- Elimina cantidades como GB, TB, MHz, etc. de tu respuesta")
        if "ram" in campo_lower:
            instrucciones_especificas.append(
                "- Ejemplo de respuesta correcta: 'DDR4', 'DDR5', 'LPDDR5'"
            )
        elif "almacenamiento" in campo_lower or "disco" in campo_lower:
            instrucciones_especificas.append(
                "- Ejemplo de respuesta correcta: 'SSD M.2', 'SSD NVMe', 'HDD SATA'"
            )
    elif any(kw in campo_lower for kw in ["nucleo", "hilo", "core", "thread"]):
        instrucciones_especificas.append("- Responde SOLO el número (ej: '8', '16')")
    elif any(kw in campo_lower for kw in ["marca", "fabricante", "manufacturer"]):
        instrucciones_especificas.append(
            "- Responde SOLO el nombre de la marca (ej: 'HP', 'Lenovo', 'Dell')"
        )
    elif any(kw in campo_lower for kw in ["procesador", "cpu", "processor"]):
        instrucciones_especificas.append(
            "- Responde el modelo completo del procesador "
            "(ej: 'AMD Ryzen 7 7735U', 'Intel Core i7-1355U')"
        )
    else:
        instrucciones_especificas.append("- Extrae el valor exacto tal como aparece en el contexto")

    instrucciones_texto = "\n".join(instrucciones_especificas)

    prompt = f"""Eres un agente especializado en extraer información sobre **{campo}** de productos.

Producto:
- Categoría: {payload.get('Categoria', 'N/A')}
- Nombre: {payload.get('productoname', 'N/A')}
- Descripción Comprador: {payload.get('DescripcionProductoComprador', 'N/A')}
- Descripción Proveedor: {payload.get('DescripcionProductoProveedor', 'N/A')}

Información del producto:
{texto_completo}

Tu tarea es extraer ÚNICAMENTE la información sobre **{campo}** del producto.

REGLAS GENERALES:
1. Responde SOLO con el valor extraído, SIN explicaciones ni texto adicional
2. Si la información no está clara o no existe, responde "No especificado"
3. NO inventes información - solo extrae lo que realmente está en el contexto
4. Sé PRECISO y conciso

INSTRUCCIONES ESPECÍFICAS PARA ESTE CAMPO:
{instrucciones_texto}

Valor de {campo}:"""

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logging.info(f"  [Campo {campo}] Intento {attempt + 1}/{max_retries} [sin RAG]...")
            llm = get_llm(model_provider=llm_provider, temperature=0.3)
            response = llm.invoke(prompt)
            valor = response.content.strip()
            valor = _limpiar_valor(valor, campo)
            logging.info(f"  [Campo {campo}] ✓ {valor[:50]}")
            return {"campo": campo, "valor": valor}

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = delay
                    match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 1
                    logging.warning(
                        f"  [Campo {campo}] Rate limit. Esperando {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= 2
                    continue
                else:
                    return {"campo": campo, "valor": "No disponible (rate limit)"}
            else:
                if attempt < max_retries - 1:
                    logging.warning(f"  [Campo {campo}] Error: {error_msg}. Reintentando en {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logging.error(f"  [Campo {campo}] Error tras {max_retries} intentos: {error_msg}")
                    return {"campo": campo, "valor": "Error en extracción"}

    return {"campo": campo, "valor": "Error en extracción"}
