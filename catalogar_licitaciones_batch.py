"""
Script para catalogar productos de licitaciones en lote.

Este script:
1. Lee el archivo Licitaciones/processed_data/Set_validacion.xlsx
2. Por cada registro, llama al endpoint /catalogar/licitacion
3. Guarda los resultados en un CSV con las columnas esperadas

Columnas de entrada:
- CodigoExterno: código de licitación
- EspecificacionComprador: descripción producto comprador
- EspecificacionProveedor: descripción producto proveedor
- NombreProductoGenerico: nombre del producto
- RutSucursal: RUT del proveedor
- Categoria: siempre "Computadores"

Columnas de salida:
- Todas las de entrada +
- Modalidad
- Tipo Producto
- Marca
- Procesador
- Núcleos
- Hilos Procesador
- RAM (GB)
- Tipo RAM
- Sistema Operativo
- Tipo Almacenamiento
- Almacenamiento (GB)
- Pantalla (Pulgadas)
- success: si la catalogación fue exitosa
- error: mensaje de error si falló
"""

import pandas as pd
import requests
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import time
from tqdm import tqdm
import os

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('catalogar_licitaciones_batch.log'),
        logging.StreamHandler()
    ]
)

# Configuración del endpoint
API_URL = "http://localhost:8000/catalogar/licitacion"
API_KEY = "1234"

# Rutas de archivos
INPUT_FILE = "../Licitaciones/processed_data/Set_validacion.xlsx"
OUTPUT_FILE = "resultados_licitaciones_catalogadas.csv"

# Configuración de procesamiento
SKIP_EXISTENTES = True  # Saltar registros ya procesados
DELAY_ENTRE_REQUESTS = 1  # Segundos entre requests para no saturar el API
MAX_REGISTROS = None  # None = procesar todos, o un número para testing


def leer_archivo_entrada() -> pd.DataFrame:
    """
    Lee el archivo Excel de entrada.

    Returns:
        DataFrame con los datos de licitaciones
    """
    try:
        df = pd.read_excel(INPUT_FILE)
        logging.info(f"Archivo leído: {len(df)} registros encontrados")

        # Verificar columnas requeridas
        columnas_requeridas = [
            'CodigoExterno',
            'EspecificacionComprador',
            'EspecificacionProveedor',
            'NombreProductoGenerico',
            'RutSucursal'
        ]

        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if columnas_faltantes:
            raise ValueError(f"Columnas faltantes en el archivo: {columnas_faltantes}")

        # Eliminar duplicados por CodigoExterno + RutSucursal
        df_original_len = len(df)
        df = df.drop_duplicates(subset=['CodigoExterno', 'RutSucursal'])
        if len(df) < df_original_len:
            logging.info(f"Se eliminaron {df_original_len - len(df)} duplicados")

        return df

    except Exception as e:
        logging.error(f"Error leyendo archivo de entrada: {e}")
        raise


def cargar_resultados_existentes() -> pd.DataFrame:
    """
    Carga resultados previamente procesados si existen.

    Returns:
        DataFrame con resultados existentes o DataFrame vacío
    """
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            logging.info(f"Resultados existentes cargados: {len(df)} registros")
            return df
        except Exception as e:
            logging.warning(f"No se pudieron cargar resultados existentes: {e}")
            return pd.DataFrame()
    else:
        logging.info("No hay archivo de resultados existente, se creará uno nuevo")
        return pd.DataFrame()


def catalogar_producto(row: pd.Series) -> Dict[str, Any]:
    """
    Llama al endpoint para catalogar un producto.

    Args:
        row: Fila del DataFrame con datos del producto

    Returns:
        dict: Resultado de la catalogación
    """
    try:
        # Preparar payload
        payload = {
            "payload": {
                "Categoria": "Computadores",
                "DescripcionProductoComprador": str(row['EspecificacionComprador']),
                "DescripcionProductoProveedor": str(row['EspecificacionProveedor']),
                "productoname": str(row['NombreProductoGenerico'])
            },
            "codigo_licitacion": str(row['CodigoExterno']),
            "rut_proveedor": str(row['RutSucursal']),
            "use_diccionarios": True,
            "llm_provider": "gemini"
        }

        # Headers
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }

        # Hacer request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            result = response.json()
            return {
                'success': result.get('success', False),
                'resultado': result.get('resultado', {}),
                'metadata': result.get('metadata', {}),
                'error': None
            }
        else:
            return {
                'success': False,
                'resultado': {},
                'metadata': {},
                'error': f"HTTP {response.status_code}: {response.text[:200]}"
            }

    except requests.exceptions.Timeout:
        return {
            'success': False,
            'resultado': {},
            'metadata': {},
            'error': "Timeout - el request tardó más de 5 minutos"
        }
    except Exception as e:
        return {
            'success': False,
            'resultado': {},
            'metadata': {},
            'error': str(e)
        }


def extraer_atributos_resultado(resultado: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae los atributos del resultado en el formato esperado.

    Args:
        resultado: Resultado de la catalogación

    Returns:
        dict: Atributos extraídos
    """
    if not resultado or not isinstance(resultado, dict):
        return {
            'Modalidad': None,
            'Tipo Producto': None,
            'Marca': None,
            'Procesador': None,
            'Núcleos': None,
            'Hilos Procesador': None,
            'RAM (GB)': None,
            'Tipo RAM': None,
            'Sistema Operativo': None,
            'Tipo Almacenamiento': None,
            'Almacenamiento (GB)': None,
            'Pantalla (Pulgadas)': None
        }

    # Extraer atributos con varios posibles nombres
    def get_attr(names):
        for name in names if isinstance(names, list) else [names]:
            if name in resultado:
                return resultado[name]
            # Buscar en nested dict
            for key, value in resultado.items():
                if isinstance(value, dict) and name in value:
                    return value[name]
        return None

    atributos = {
        'Modalidad': get_attr(['modalidad', 'Modalidad', 'tipo_producto']),
        'Tipo Producto': get_attr(['tipo_producto', 'Tipo Producto', 'tipo']),
        'Marca': get_attr(['marca', 'Marca']),
        'Procesador': get_attr(['procesador', 'Procesador', 'cpu']),
        'Núcleos': get_attr(['nucleos', 'Núcleos', 'nucleos_procesador', 'cores']),
        'Hilos Procesador': get_attr(['hilos', 'Hilos Procesador', 'hilos_procesador', 'threads']),
        'RAM (GB)': get_attr(['ram', 'RAM (GB)', 'memoria_ram', 'memoria']),
        'Tipo RAM': get_attr(['tipo_ram', 'Tipo RAM', 'ram_tipo']),
        'Sistema Operativo': get_attr(['sistema_operativo', 'Sistema Operativo', 'os', 'so']),
        'Tipo Almacenamiento': get_attr(['tipo_almacenamiento', 'Tipo Almacenamiento', 'storage_type']),
        'Almacenamiento (GB)': get_attr(['almacenamiento', 'Almacenamiento (GB)', 'disco', 'storage']),
        'Pantalla (Pulgadas)': get_attr(['pantalla', 'Pantalla (Pulgadas)', 'tamaño_pantalla', 'screen'])
    }

    return atributos


def procesar_lote():
    """
    Procesa el lote completo de licitaciones.
    """
    logging.info("="*80)
    logging.info("INICIO DE CATALOGACIÓN DE LICITACIONES EN LOTE")
    logging.info("="*80)

    # Leer archivo de entrada
    df_input = leer_archivo_entrada()

    # Cargar resultados existentes
    df_existentes = cargar_resultados_existentes()

    # Filtrar registros ya procesados
    if SKIP_EXISTENTES and not df_existentes.empty:
        # Crear columna de identificación
        df_input['_id'] = df_input['CodigoExterno'].astype(str) + '_' + df_input['RutSucursal'].astype(str)
        df_existentes['_id'] = df_existentes['CodigoExterno'].astype(str) + '_' + df_existentes['RutSucursal'].astype(str)

        # Filtrar los ya procesados exitosamente
        ids_exitosos = set(df_existentes[df_existentes['success'] == True]['_id'].tolist())
        df_pendientes = df_input[~df_input['_id'].isin(ids_exitosos)].copy()
        df_pendientes = df_pendientes.drop('_id', axis=1)

        logging.info(f"Registros ya procesados: {len(ids_exitosos)}")
        logging.info(f"Registros pendientes: {len(df_pendientes)}")
    else:
        df_pendientes = df_input.copy()

    # Limitar si se especificó MAX_REGISTROS
    if MAX_REGISTROS:
        df_pendientes = df_pendientes.head(MAX_REGISTROS)
        logging.info(f"Limitando procesamiento a {MAX_REGISTROS} registros")

    if len(df_pendientes) == 0:
        logging.info("No hay registros pendientes para procesar")
        return

    # Procesar cada registro
    resultados = []

    logging.info(f"\nIniciando procesamiento de {len(df_pendientes)} registros...")

    for idx, row in tqdm(df_pendientes.iterrows(), total=len(df_pendientes), desc="Catalogando"):
        try:
            logging.info(f"\n--- Procesando {idx + 1}/{len(df_pendientes)} ---")
            logging.info(f"Licitación: {row['CodigoExterno']}, RUT: {row['RutSucursal']}")

            # Catalogar
            resultado_api = catalogar_producto(row)

            # Extraer atributos
            atributos = extraer_atributos_resultado(resultado_api.get('resultado', {}))

            # Combinar datos originales + atributos + metadata
            resultado_completo = {
                'CodigoExterno': row['CodigoExterno'],
                'RutSucursal': row['RutSucursal'],
                'EspecificacionComprador': row['EspecificacionComprador'],
                'EspecificacionProveedor': row['EspecificacionProveedor'],
                'NombreProductoGenerico': row['NombreProductoGenerico'],
                **atributos,
                'success': resultado_api['success'],
                'error': resultado_api['error'],
                'adjuntos_descargados': resultado_api.get('metadata', {}).get('adjuntos_descargados', False),
                'adjuntos_procesados': resultado_api.get('metadata', {}).get('adjuntos_procesados', False),
                'fecha_catalogacion': datetime.now().isoformat()
            }

            resultados.append(resultado_completo)

            # Log resultado
            if resultado_api['success']:
                logging.info(f"✓ Catalogación exitosa")
            else:
                logging.warning(f"✗ Error: {resultado_api['error']}")

            # Guardar progreso cada 10 registros
            if len(resultados) % 10 == 0:
                guardar_resultados_parciales(df_existentes, resultados)

            # Delay entre requests
            time.sleep(DELAY_ENTRE_REQUESTS)

        except Exception as e:
            logging.exception(f"Error procesando registro {idx}: {e}")
            # Agregar registro con error
            resultados.append({
                'CodigoExterno': row['CodigoExterno'],
                'RutSucursal': row['RutSucursal'],
                'EspecificacionComprador': row['EspecificacionComprador'],
                'EspecificacionProveedor': row['EspecificacionProveedor'],
                'NombreProductoGenerico': row['NombreProductoGenerico'],
                'Modalidad': None,
                'Tipo Producto': None,
                'Marca': None,
                'Procesador': None,
                'Núcleos': None,
                'Hilos Procesador': None,
                'RAM (GB)': None,
                'Tipo RAM': None,
                'Sistema Operativo': None,
                'Tipo Almacenamiento': None,
                'Almacenamiento (GB)': None,
                'Pantalla (Pulgadas)': None,
                'success': False,
                'error': str(e),
                'adjuntos_descargados': False,
                'adjuntos_procesados': False,
                'fecha_catalogacion': datetime.now().isoformat()
            })

    # Guardar resultados finales
    guardar_resultados_finales(df_existentes, resultados)

    # Resumen
    logging.info("\n" + "="*80)
    logging.info("RESUMEN DE CATALOGACIÓN")
    logging.info("="*80)
    exitosos = sum(1 for r in resultados if r['success'])
    fallidos = len(resultados) - exitosos
    logging.info(f"Total procesados: {len(resultados)}")
    logging.info(f"Exitosos: {exitosos} ({exitosos/len(resultados)*100:.1f}%)")
    logging.info(f"Fallidos: {fallidos} ({fallidos/len(resultados)*100:.1f}%)")
    logging.info(f"Archivo de salida: {OUTPUT_FILE}")
    logging.info("="*80)


def guardar_resultados_parciales(df_existentes: pd.DataFrame, nuevos_resultados: list):
    """Guarda resultados parciales (cada 10 registros)."""
    try:
        df_nuevos = pd.DataFrame(nuevos_resultados)
        df_total = pd.concat([df_existentes, df_nuevos], ignore_index=True)
        df_total.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Progreso guardado: {len(nuevos_resultados)} registros nuevos")
    except Exception as e:
        logging.error(f"Error guardando progreso: {e}")


def guardar_resultados_finales(df_existentes: pd.DataFrame, nuevos_resultados: list):
    """Guarda todos los resultados al final."""
    try:
        df_nuevos = pd.DataFrame(nuevos_resultados)
        df_total = pd.concat([df_existentes, df_nuevos], ignore_index=True)

        # Eliminar duplicados (mantener el más reciente)
        df_total = df_total.sort_values('fecha_catalogacion', ascending=False)
        df_total = df_total.drop_duplicates(subset=['CodigoExterno', 'RutSucursal'], keep='first')

        # Guardar
        df_total.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")
        logging.info(f"Total de registros en archivo: {len(df_total)}")

    except Exception as e:
        logging.error(f"Error guardando resultados finales: {e}")
        raise


def main():
    """Función principal."""
    try:
        procesar_lote()
    except KeyboardInterrupt:
        logging.warning("\n\nProceso interrumpido por el usuario")
        logging.info("Los resultados parciales han sido guardados")
    except Exception as e:
        logging.exception(f"Error fatal: {e}")
        raise


if __name__ == "__main__":
    # Verificar que el archivo de entrada existe
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: No se encuentra el archivo de entrada: {INPUT_FILE}")
        print(f"Por favor verifica la ruta del archivo.")
        exit(1)

    # Verificar que el API está corriendo
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ Error: El API no está respondiendo correctamente")
            print("Por favor inicia el servidor con: python main.py")
            exit(1)
        print("✓ API está corriendo")
    except:
        print("\n❌ Error: No se pudo conectar al API")
        print("Por favor inicia el servidor con: python main.py")
        exit(1)

    main()
