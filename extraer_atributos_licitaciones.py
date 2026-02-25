"""
Script principal para extracción de atributos de productos en LICITACIONES.

Este script orquesta el flujo completo de catalogación para licitaciones:
1. Descarga de adjuntos desde Mercado Público (anexos técnicos y económicos)
2. Procesamiento de adjuntos (extracción de texto con OCR)
3. Catalogación usando un LLM único con RAG sobre adjuntos

A diferencia de compra_agil que usa agentes múltiples con LangGraph,
este módulo usa un enfoque más directo con un único LLM.

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

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def extraer_atributos_licitacion(
    payload: Dict[str, Any],
    codigo_licitacion: str,
    rut_proveedor: str,
    use_diccionarios: bool = True,
    llm_provider: str = None,
    campos_manuales: List[str] = None,
    downloader: Any = None
) -> Dict[str, Any]:
    """
    Extrae atributos de un producto de licitación reutilizando el pipeline
    completo de compra ágil (LangGraph + agentes).

    La única diferencia con compra ágil es el downloader de adjuntos:
    usa LicitacionDownloaderAdapter, que descarga los anexos técnicos y
    económicos del portal público sin necesidad de autenticación.

    El output es idéntico al de extraer_atributos (compra ágil).

    Args:
        payload: Información del producto.
        codigo_licitacion: Código de la licitación (ej: "1234-567-LE24").
        rut_proveedor: RUT del proveedor.
        use_diccionarios: Si usar normalización con diccionarios.
        llm_provider: Proveedor de LLM. None = usa DEFAULT_LLM_PROVIDER.
        campos_manuales: Lista de campos adicionales a extraer.
        downloader: Ignorado — licitaciones siempre usan LicitacionDownloaderAdapter.
    """
    # Validar payload
    is_valid, errores = validar_payload(payload)
    if not is_valid:
        raise ValueError(f"Payload inválido: {', '.join(errores)}")

    logging.info(f"\n{'='*80}")
    logging.info(f"EXTRACCIÓN DE ATRIBUTOS - LICITACIÓN")
    logging.info(f"Licitación: {codigo_licitacion} | Proveedor: {rut_proveedor}")
    logging.info(f"{'='*80}\n")

    from extraer_atributos import extraer_atributos
    from utils.get_attachments import LicitacionDownloaderAdapter

    adapter = LicitacionDownloaderAdapter()

    return extraer_atributos(
        payload=payload,
        codigo_cotizacion=codigo_licitacion,
        rut_proveedor=rut_proveedor,
        use_diccionarios=use_diccionarios,
        llm_provider=llm_provider,
        downloader=adapter,
        campos_manuales_lista=campos_manuales,
    )


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description='Extrae atributos de productos en licitaciones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Extraer atributos de una licitación
  python extraer_atributos_licitaciones.py --payload payload.json --codigo 1234-567-LE24 --rut 76123456-7

  # Sin usar diccionarios
  python extraer_atributos_licitaciones.py --payload payload.json --codigo 1234-567-LE24 --rut 76123456-7 --no-diccionarios

  # Usar modelo específico
  python extraer_atributos_licitaciones.py --payload payload.json --codigo 1234-567-LE24 --rut 76123456-7 --llm openai

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
        help='Ruta al archivo JSON con el payload'
    )

    parser.add_argument(
        '--codigo',
        '--codigo-licitacion',
        type=str,
        required=True,
        help='Código de la licitación'
    )

    parser.add_argument(
        '--rut',
        '--rut-proveedor',
        type=str,
        required=True,
        help='RUT del proveedor'
    )

    parser.add_argument(
        '--no-diccionarios',
        action='store_true',
        help='No usar diccionarios para normalización'
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
        help='Ruta donde guardar el resultado (JSON)'
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
        resultado = extraer_atributos_licitacion(
            payload=payload_data,
            codigo_licitacion=args.codigo,
            rut_proveedor=args.rut,
            use_diccionarios=not args.no_diccionarios,
            llm_provider=args.llm
        )

        resultado_final = resultado.get('resultado_final')

        # Guardar o mostrar resultado
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(resultado_final, f, indent=2, ensure_ascii=False)
            logging.info(f"\n✓ Resultado guardado en: {args.output}")
        else:
            print("\n" + "="*80)
            print("RESULTADO FINAL")
            print("="*80 + "\n")
            print(json.dumps(resultado_final, indent=2, ensure_ascii=False))

    except Exception as e:
        logging.exception(f"Error en la extracción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Si no hay argumentos, mostrar ejemplo
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("EJEMPLO DE USO - LICITACIONES")
        print("="*80 + "\n")

        payload_ejemplo = {
            "Categoria": "Computadores",
            "DescripcionProductoComprador": "NOTEBOOK 512 GB DISCO DURO 256 GB SSD 8GB DE RAM",
            "DescripcionProductoProveedor": "NOTEBOOK HP PAVILION",
            "productoname": "Notebook, laptop o computador portátil excepto Tablet PC"
        }

        print("Payload de ejemplo:")
        print(json.dumps(payload_ejemplo, indent=2, ensure_ascii=False))

        print("\n" + "-"*80)
        print("Comando de ejemplo:")
        print("-"*80)
        print("python extraer_atributos_licitaciones.py --payload payload.json --codigo 1234-567-LE24 --rut 76123456-7")

        print("\n" + "-"*80)
        print("Para ver todas las opciones:")
        print("-"*80)
        print("python extraer_atributos_licitaciones.py --help")
        print()

        sys.exit(0)

    main()
