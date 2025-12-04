"""
Script principal para extracción de atributos de productos.

Este script orquesta el flujo completo de catalogación usando LangGraph:
1. Descarga de adjuntos desde Mercado Público
2. Procesamiento de adjuntos (extracción de texto)
3. Catalogación con RAG usando adjuntos y diccionarios

Actualmente soporta:
- Computadores (Desktops, Notebooks, All in One)

Próximamente:
- Medicamentos
- Otros productos
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.grafo_comp import ejecutar_catalogacion

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validar_payload(payload: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Valida que el payload tenga los campos requeridos.

    Args:
        payload: Payload a validar

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errores = []

    # Campos requeridos básicos
    campos_requeridos = ['Categoria', 'DescripcionProductoComprador', 'DescripcionProductoProveedor', 'productoname']

    for campo in campos_requeridos:
        if campo not in payload:
            errores.append(f"Campo requerido faltante: {campo}")

    # Validar que Categoria sea Computadores (por ahora)
    if 'Categoria' in payload and payload['Categoria'] != 'Computadores':
        errores.append(f"Categoría '{payload['Categoria']}' no soportada. Solo 'Computadores' está disponible.")

    return len(errores) == 0, errores


def extraer_atributos(
    payload: Dict[str, Any],
    codigo_cotizacion: str,
    rut_proveedor: str,
    use_diccionarios: bool = True,
    llm_provider: str = None
) -> Dict[str, Any]:
    """
    Extrae atributos de un producto usando el flujo completo de catalogación.

    Args:
        payload: Información del producto con estructura:
            {
                'Categoria': str,  # Por ahora solo 'Computadores'
                'DescripcionProductoComprador': str,
                'DescripcionProductoProveedor': str,
                'productoname': str  # Diferente a Categoria, indica el tipo específico
            }
        codigo_cotizacion: Código de la solicitud de cotización
        rut_proveedor: RUT del proveedor
        use_diccionarios: Si usar normalización con diccionarios
        llm_provider: Proveedor de LLM ('openai', 'gemini', 'deepseek')
                     Si es None, usa DEFAULT_LLM_PROVIDER del .env

    Returns:
        dict: Estado final con el resultado de la catalogación

    Raises:
        ValueError: Si el payload no es válido o la categoría no es soportada

    Example:
        >>> payload = {
        ...     'Categoria': 'Computadores',
        ...     'DescripcionProductoComprador': 'NOTEBOOK HP 8GB RAM',
        ...     'DescripcionProductoProveedor': 'Notebook HP Pavilion',
        ...     'productoname': 'Notebook, laptop o computador portátil excepto Tablet PC'
        ... }
        >>> resultado = extraer_atributos(
        ...     payload,
        ...     "12345678",
        ...     "76123456-7"
        ... )
        >>> print(resultado['resultado_final'])
    """
    # Validar payload
    is_valid, errores = validar_payload(payload)
    if not is_valid:
        raise ValueError(f"Payload inválido: {', '.join(errores)}")

    # Añadir ROWNUM interno si no existe (solo para uso del LLM)
    if 'ROWNUM' not in payload:
        import uuid
        payload['ROWNUM'] = str(uuid.uuid4())[:8]  # ID único de 8 caracteres

    # Determinar categoría
    categoria = payload.get('Categoria', 'Computadores')

    logging.info(f"\n{'='*80}")
    logging.info(f"EXTRACCIÓN DE ATRIBUTOS")
    logging.info(f"Categoría: {categoria}")
    logging.info(f"Producto: {payload.get('productoname')}")
    logging.info(f"Cotización: {codigo_cotizacion}")
    logging.info(f"Proveedor: {rut_proveedor}")
    logging.info(f"{'='*80}\n")

    # Ejecutar workflow según categoría
    if categoria == 'Computadores':
        resultado = ejecutar_catalogacion(
            codigo_cotizacion=codigo_cotizacion,
            rut_proveedor=rut_proveedor,
            payload=payload,
            use_diccionarios=use_diccionarios,
            llm_provider=llm_provider
        )
    else:
        raise ValueError(
            f"Categoría '{categoria}' no soportada. "
            f"Categorías disponibles: ['Computadores']"
        )

    return resultado


def extraer_atributos_lote(
    payloads: List[Dict[str, Any]],
    codigo_cotizacion: str,
    rut_proveedor: str,
    use_diccionarios: bool = True,
    llm_provider: str = None
) -> List[Dict[str, Any]]:
    """
    Extrae atributos de un lote de productos.

    Args:
        payloads: Lista de payloads de productos
        codigo_cotizacion: Código de la solicitud de cotización
        rut_proveedor: RUT del proveedor
        use_diccionarios: Si usar normalización con diccionarios
        llm_provider: Proveedor de LLM

    Returns:
        list: Lista de estados finales con resultados

    Example:
        >>> payloads = [
        ...     {'Categoria': 'Computadores', 'DescripcionProductoComprador': '...', ...},
        ...     {'Categoria': 'Computadores', 'DescripcionProductoComprador': '...', ...}
        ... ]
        >>> resultados = extraer_atributos_lote(
        ...     payloads, "12345678", "76123456-7"
        ... )
    """
    resultados = []

    for i, payload in enumerate(payloads, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESANDO {i}/{len(payloads)}")
        logging.info(f"{'='*80}\n")

        try:
            resultado = extraer_atributos(
                payload,
                codigo_cotizacion,
                rut_proveedor,
                use_diccionarios=use_diccionarios,
                llm_provider=llm_provider
            )
            resultados.append(resultado)

        except Exception as e:
            logging.error(f"Error procesando producto {i}: {e}")
            resultados.append({
                'payload': payload,
                'resultado_final': None,
                'errores': [str(e)]
            })

    return resultados


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description='Extrae atributos de productos usando LangGraph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Extraer atributos de un producto desde JSON
  python extraer_atributos.py --payload payload.json --codigo 12345678 --rut 76123456-7

  # Extraer de un lote de productos
  python extraer_atributos.py --payload payloads.json --codigo 12345678 --rut 76123456-7 --lote

  # Sin usar diccionarios (solo adjuntos)
  python extraer_atributos.py --payload payload.json --codigo 12345678 --rut 76123456-7 --no-diccionarios

  # Usar modelo específico
  python extraer_atributos.py --payload payload.json --codigo 12345678 --rut 76123456-7 --llm openai

Formato del payload (JSON):
  {
    "Categoria": "Computadores",
    "DescripcionProductoComprador": "NOTEBOOK HP 8GB RAM",
    "DescripcionProductoProveedor": "Notebook HP Pavilion",
    "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
  }
        """
    )

    parser.add_argument(
        '--payload',
        type=str,
        required=True,
        help='Ruta al archivo JSON con el payload (o lista de payloads si --lote)'
    )

    parser.add_argument(
        '--codigo',
        '--codigo-cotizacion',
        type=str,
        required=True,
        help='Código de la solicitud de cotización'
    )

    parser.add_argument(
        '--rut',
        '--rut-proveedor',
        type=str,
        required=True,
        help='RUT del proveedor'
    )

    parser.add_argument(
        '--lote',
        action='store_true',
        help='Si el payload es un lote (lista de productos)'
    )

    parser.add_argument(
        '--no-diccionarios',
        action='store_true',
        help='No usar diccionarios para normalización (solo adjuntos)'
    )

    parser.add_argument(
        '--llm',
        '--llm-provider',
        type=str,
        choices=['openai', 'gemini', 'deepseek'],
        default=None,
        help='Proveedor de LLM (por defecto usa DEFAULT_LLM_PROVIDER del .env)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Ruta donde guardar el resultado (JSON). Si no se especifica, imprime en consola'
    )

    args = parser.parse_args()

    # Cargar payload
    try:
        with open(args.payload, 'r', encoding='utf-8') as f:
            payload_data = json.load(f)
    except Exception as e:
        logging.error(f"Error cargando payload: {e}")
        sys.exit(1)

    # Ejecutar extracción
    try:
        if args.lote:
            # Procesar lote
            if not isinstance(payload_data, list):
                logging.error("El payload debe ser una lista cuando se usa --lote")
                sys.exit(1)

            resultados = extraer_atributos_lote(
                payloads=payload_data,
                codigo_cotizacion=args.codigo,
                rut_proveedor=args.rut,
                use_diccionarios=not args.no_diccionarios,
                llm_provider=args.llm
            )

            # Extraer solo los resultados finales
            resultados_finales = [
                r.get('resultado_final') for r in resultados if r.get('resultado_final')
            ]

        else:
            # Procesar uno solo
            if isinstance(payload_data, list):
                logging.error("El payload no debe ser una lista. Usa --lote para procesar múltiples productos")
                sys.exit(1)

            resultado = extraer_atributos(
                payload=payload_data,
                codigo_cotizacion=args.codigo,
                rut_proveedor=args.rut,
                use_diccionarios=not args.no_diccionarios,
                llm_provider=args.llm
            )

            resultados_finales = resultado.get('resultado_final')

        # Guardar o mostrar resultado
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(resultados_finales, f, indent=2, ensure_ascii=False)
            logging.info(f"\n✓ Resultado guardado en: {args.output}")
        else:
            print("\n" + "="*80)
            print("RESULTADO FINAL")
            print("="*80 + "\n")
            print(json.dumps(resultados_finales, indent=2, ensure_ascii=False))

    except Exception as e:
        logging.exception(f"Error en la extracción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Si no hay argumentos, mostrar ejemplo
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("EJEMPLO DE USO")
        print("="*80 + "\n")

        payload_ejemplo = {
            "Categoria": "Computadores",
            "DescripcionProductoComprador": "NOTEBOOK 512 GB DISCO DURO 256 GB SSD 8GB DE RAM PROCESADOR AMD RYZEN 7 PANTALLA 15.6",
            "DescripcionProductoProveedor": "NOTEBOOK HP",
            "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
        }

        print("Payload de ejemplo:")
        print(json.dumps(payload_ejemplo, indent=2, ensure_ascii=False))

        print("\n" + "-"*80)
        print("Comando de ejemplo:")
        print("-"*80)
        print("python extraer_atributos.py --payload payload.json --codigo 12345678 --rut 76123456-7")

        print("\n" + "-"*80)
        print("Para ver todas las opciones:")
        print("-"*80)
        print("python extraer_atributos.py --help")
        print()

        sys.exit(0)

    main()
