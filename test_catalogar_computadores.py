"""
test_catalogar_computadores.py
Script de prueba para el endpoint POST /catalogar (Compra Ágil).

Lee 10 filas del CSV de muestra (Notebook, Desktop, AIO), llama a la API
y registra los TIEMPOS INTERNOS de cada fase del pipeline (devueltos por
el endpoint en metadata.tiempos).

El CSV de salida tiene una fila por fase, por iteración.
"""
import csv
import json
import logging
import time
from datetime import datetime

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000"
API_KEY = "1234"
TOKEN_BEARER = (
    "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI2Rk1NaXFwVWRLY3Ryb0IweXgwRWdKWS1xVDZIVDBEQXgyR3JvWFlja25JIn0.eyJleHAiOjE3NzI4NTAxNzUsImlhdCI6MTc3MjgyMTM3NSwianRpIjoiMThiNWI5ZWQtOWMyOC00MTgwLThmNjYtZTgwNzBkNWU5NGJlIiwiaXNzIjoiaHR0cHM6Ly9oZWltZGFsbC5tZXJjYWRvcHVibGljby5jbC9hdXRoL3JlYWxtcy9jaGlsZWNvbXByYXJlYWxtIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImY6YWQ3MGUxMjgtYzQzZi00NTBkLWE4ODctZjhhMGNlYzc3ZWE1OjIxODYyNjRfMTQwMTY0OSIsInR5cCI6IkJlYXJlciIsImF6cCI6Im1lcmNhZG9QdWJsaWNvQ2xpZW50Iiwic2lkIjoiODBhZDZmNjAtYmZkZi00YWY4LThkNmUtYjc4YjA1MDdiOGU5IiwiYWxsb3dlZC1vcmlnaW5zIjpbIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvbnN1bW9zZXJ2aWNpbyIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBlbWFpbCBwcm9maWxlIiwiY29kaWdvT3JnYW5pc21vIjoiMTQwMTY0OSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiY29kaWdvVXN1YXJpbyI6IjIxODYyNjQiLCJuYW1lIjoiRWR1YXJkbyBBbmRyw6lzIE1veWEgQnJpb25lcyIsInRpcG9Vc3VhcmlvIjoiUHJvdmVlZG9yIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMjE4NjI2NF8xNDAxNjQ5IiwiZ2l2ZW5fbmFtZSI6IkVkdWFyZG8gQW5kcsOpcyBNb3lhIEJyaW9uZXMifQ.UEC2_GZ6fA8jlJRsdSZ12SqyymWEZcSY_PiGo_JT1pms4Y3oCY0TxFa4o0g5CQ-sr-8wBrDRislJIurWH9J5ffDa4FT48wY-N5bte4wjL1LcDKEthdC6EMwEES-jgqqkZXtIKHH9JmrRdFHl7eb3prmJRVLHp8-NUv1OGdyer1ryDrL6n-Wd0fL8MPXQ97O8Y7ty7mTQW1r05tusWTESUNebpBW6nXUYa54TbPq1R2u2Ad3NwDAhwYWMtfq0l1b0kvJAxYMIC9j9-LZn4djsghjjsFUTb1bPWmR1rlrVot-QO2eZNmdl8GuZHairvGJrSsdK0jmaLedozB_KW4uqYw"
)

CSV_INPUT = "sample_OC_100_rows_computadores.csv"
TIPOS_PRODUCTO = ["Notebook", "Desktop", "All in One (AIO)"]
N_MUESTRAS = 10

CAMPOS_MANUALES = [
    "Pantalla (Pulgadas)",
    "Procesador",
    "Marca",
    "Tipo RAM",
    "RAM (GB)",
    "Tipo Almacenamiento",
    "Almacenamiento (GB)",
    "Hilos",
    "Nucleos",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"test_catalogar_{timestamp}.log"
csv_output = f"resultados_test_catalogar_{timestamp}.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Columnas del CSV de salida
COLUMNAS_CSV = [
    "iteracion",
    "tipo_producto",
    "codigo_cotizacion",
    "proveedor",
    "nivel",        # "interno" (pipeline) | "externo" (llamada HTTP)
    "nodo",         # nodo LangGraph o "http"
    "fase",         # sub-fase dentro del nodo
    "descripcion",
    "segundos",
    "exito",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def limpiar_texto(valor) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def agregar_fila(
    filas: list,
    iteracion: int,
    tipo_producto: str,
    codigo_cotizacion: str,
    proveedor: str,
    nivel: str,
    nodo: str,
    fase: str,
    descripcion: str,
    segundos: float,
    exito: bool = True,
):
    estado = "OK" if exito else "ERROR"
    logger.info(
        f"[Iter {iteracion:02d}] [{nivel:<8}] [{nodo:<30}] [{fase:<25}] [{estado}] "
        f"{descripcion[:60]} | {segundos:.3f}s"
    )
    filas.append({
        "iteracion": iteracion,
        "tipo_producto": tipo_producto,
        "codigo_cotizacion": codigo_cotizacion,
        "proveedor": proveedor,
        "nivel": nivel,
        "nodo": nodo,
        "fase": fase,
        "descripcion": descripcion,
        "segundos": round(segundos, 4),
        "exito": exito,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("PRUEBA /catalogar — Compra Ágil — Computadores")
    logger.info(f"API: {API_URL}   |   Salida: {csv_output}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Carga y filtrado del CSV
    # ------------------------------------------------------------------
    logger.info(f"\n[GLOBAL] Cargando CSV: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)
    logger.info(f"  Total filas CSV: {len(df)}")

    df_filtrado = df[df["Tipo Producto"].isin(TIPOS_PRODUCTO)].copy()
    logger.info(f"  Filas con tipos válidos: {len(df_filtrado)}")

    muestras = (
        df_filtrado
        .groupby("Tipo Producto", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), N_MUESTRAS), random_state=42))
        .sample(frac=1, random_state=42)
        .head(N_MUESTRAS)
        .reset_index(drop=True)
    )
    logger.info(f"  Muestras seleccionadas: {len(muestras)}")
    logger.info(f"  Distribución:\n{muestras['Tipo Producto'].value_counts().to_string()}\n")

    # ------------------------------------------------------------------
    # 2. Iterar sobre las muestras
    # ------------------------------------------------------------------
    filas_csv: list[dict] = []

    for idx, row in muestras.iterrows():
        iteracion = idx + 1
        t_iter_ini = time.time()

        tipo_producto   = limpiar_texto(row["Tipo Producto"])
        codigo_cot      = limpiar_texto(row["CodigoExterno"])
        rut_proveedor   = limpiar_texto(row["RutSucursal"])
        nombre_prov     = limpiar_texto(row["NombreProveedor"])
        esp_comprador   = limpiar_texto(row["EspecificacionComprador"])
        esp_proveedor   = limpiar_texto(row["EspecificacionProveedor"])
        producto_gen    = limpiar_texto(row["NombreroductoGenerico"])

        logger.info("-" * 70)
        logger.info(
            f"ITERACIÓN {iteracion}/{N_MUESTRAS} | {tipo_producto} | "
            f"COT: {codigo_cot} | {nombre_prov}"
        )

        payload = {
            "payload": {
                "Categoria": "Computadores",
                "DescripcionProductoComprador": esp_comprador,
                "DescripcionProductoProveedor": esp_proveedor,
                "productoname": producto_gen,
            },
            "codigo_cotizacion": codigo_cot,
            "rut_proveedor": rut_proveedor,
            "use_diccionarios": True,
            "campos_manuales": CAMPOS_MANUALES,
            "token_bearer": TOKEN_BEARER,
        }

        # ---- Llamada HTTP (tiempo externo total) ----
        t_http_ini = time.time()
        response = None
        exito_http = False

        try:
            logger.info(f"  → POST {API_URL}/catalogar ...")
            response = requests.post(
                f"{API_URL}/catalogar",
                headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=360,
            )
            t_http_fin = time.time()
            exito_http = response.status_code == 200

            agregar_fila(
                filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
                nivel="externo",
                nodo="http",
                fase="llamada_api",
                descripcion=f"POST /catalogar → HTTP {response.status_code}",
                segundos=t_http_fin - t_http_ini,
                exito=exito_http,
            )

        except requests.exceptions.Timeout:
            t_http_fin = time.time()
            agregar_fila(
                filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
                nivel="externo", nodo="http", fase="llamada_api",
                descripcion="TIMEOUT — la API no respondió en 360s",
                segundos=t_http_fin - t_http_ini,
                exito=False,
            )

        except Exception as exc:
            t_http_fin = time.time()
            agregar_fila(
                filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
                nivel="externo", nodo="http", fase="llamada_api",
                descripcion=f"Error de conexión: {exc}",
                segundos=t_http_fin - t_http_ini,
                exito=False,
            )

        # ---- Tiempos INTERNOS del pipeline (desde metadata.tiempos) ----
        if response is not None and exito_http:
            try:
                data = response.json()
                tiempos_internos = (data.get("metadata") or {}).get("tiempos", [])
                resultado = data.get("resultado") or {}
                errores = data.get("errores") or []
                warnings = data.get("warnings") or []

                logger.info(f"  ← Resultado: Tipo={resultado.get('Tipo','N/A')},"
                            f" Modelo={resultado.get('Modelo','N/A')}")
                if errores:
                    logger.warning(f"  ⚠ Errores: {errores}")
                if warnings:
                    logger.warning(f"  ⚠ Warnings: {warnings}")

                logger.info(f"  ← Fases internas recibidas: {len(tiempos_internos)}")

                for t in tiempos_internos:
                    agregar_fila(
                        filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
                        nivel="interno",
                        nodo=t.get("nodo", "?"),
                        fase=t.get("fase", "?"),
                        descripcion=t.get("descripcion", ""),
                        segundos=t.get("segundos", 0.0),
                        exito=True,
                    )

            except Exception as exc:
                logger.error(f"  Error parseando respuesta: {exc}")
                agregar_fila(
                    filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
                    nivel="externo", nodo="http", fase="parse_response",
                    descripcion=f"Error parseando JSON: {exc}",
                    segundos=0.0,
                    exito=False,
                )

        elif response is not None and not exito_http:
            body = response.text[:300] if response.text else "(vacío)"
            logger.error(f"  ← Error HTTP {response.status_code}: {body}")

        # ---- Tiempo total de la iteración (externo) ----
        agregar_fila(
            filas_csv, iteracion, tipo_producto, codigo_cot, nombre_prov,
            nivel="externo", nodo="http", fase="total_iteracion",
            descripcion=f"Tiempo total iteración {iteracion} (incluyendo red)",
            segundos=time.time() - t_iter_ini,
        )

    # ------------------------------------------------------------------
    # 3. Guardar CSV
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"Guardando {len(filas_csv)} filas en {csv_output} ...")

    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        writer.writeheader()
        writer.writerows(filas_csv)

    logger.info(f"Resultados: {csv_output}")
    logger.info(f"Log:        {log_file}")

    # Resumen por nodo (fases "total" internas)
    df_res = pd.DataFrame(filas_csv)
    df_int = df_res[(df_res["nivel"] == "interno") & (df_res["fase"] == "total")]
    if not df_int.empty:
        logger.info("\nResumen tiempos internos (fase 'total') por nodo:")
        resumen = (
            df_int.groupby("nodo")["segundos"]
            .agg(["mean", "min", "max", "count"])
            .round(3)
        )
        logger.info("\n" + resumen.to_string())

    ext_total = df_res[(df_res["nivel"] == "externo") & (df_res["fase"] == "total_iteracion")]
    if not ext_total.empty:
        logger.info(f"\nTiempo externo promedio por iteración: {ext_total['segundos'].mean():.2f}s")

    logger.info("=" * 70)
    logger.info("FIN DE PRUEBAS")


if __name__ == "__main__":
    main()
