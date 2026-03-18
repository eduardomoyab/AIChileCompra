"""
extractor/extraer_masivo.py

Extracción masiva de atributos usando el endpoint /catalogar.

- Lee campos y contextos desde attribute_descriptions.csv
- Lee los datos desde req_oferta_ganadora_202301_202602.csv
- Divide en N_WORKERS chunks y procesa cada uno en un proceso separado
- Cada proceso guarda sus resultados en extractor/results/chunk_XX.parquet
- Manejo de errores robusto: nunca detiene el proceso completo por un fallo individual
- Soporte de resume: si el archivo de salida existe, salta filas ya procesadas

Uso:
    python extractor/extraer_masivo.py
"""

import os
import sys
import json
import time
import logging
import math
import multiprocessing
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Dropbox upload
# ---------------------------------------------------------------------------

def _upload_dropbox(local_path: Path, chunk_id: int, log) -> str:
    """Sube el parquet a Dropbox y retorna la URL de descarga. Silencioso si falla."""
    token = os.getenv("DROPBOX_TOKEN")
    if not token:
        return ""
    dest = f"/extractor/chunk_{chunk_id:02d}.parquet"
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        resp = requests.post(
            "https://content.dropboxapi.com/2/files/upload",
            headers={
                "Authorization": f"Bearer {token}",
                "Dropbox-API-Arg": json.dumps({
                    "path": dest,
                    "mode": "overwrite",
                    "autorename": False,
                }),
                "Content-Type": "application/octet-stream",
            },
            data=data,
            timeout=120,
        )
        resp.raise_for_status()
        log.info(f"  [Dropbox] chunk_{chunk_id:02d} subido → {dest}")
        return dest
    except Exception as e:
        log.warning(f"  [Dropbox] Error subiendo chunk_{chunk_id:02d}: {e}")
        return ""

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
API_URL   = os.getenv("API_URL", "https://aichilecompra-production.up.railway.app")
API_KEY   = os.getenv("API_KEY", "1234")
CATEGORIA = os.getenv("CATEGORIA", "Computadores")

N_WORKERS  = int(os.getenv("N_WORKERS", "10"))
MAX_RETRIES = 3
TIMEOUT_S   = 360

SCRIPT_DIR    = Path(__file__).parent
CSV_ATTRS     = SCRIPT_DIR / "attribute_descriptions.csv"
CSV_DATOS     = SCRIPT_DIR / "req_oferta_ganadora_202301_202602.csv"
RESULTS_DIR   = SCRIPT_DIR / "results"
ID_COL        = "id_oferta_producto_aquiles"

# ---------------------------------------------------------------------------
# Construcción de campos_manuales desde attribute_descriptions.csv
# ---------------------------------------------------------------------------

def _val(v) -> str:
    return str(v).strip() if pd.notna(v) and str(v).strip() not in ("", "nan") else ""


def build_campos_manuales(csv_path: Path) -> list[dict]:
    """
    Lee attribute_descriptions.csv y construye la lista de campos_manuales.
    El contexto combina: descripcion + consideraciones + formato_ejemplo + rango_o_valores.
    """
    df = pd.read_csv(csv_path)
    campos = []
    for _, row in df.iterrows():
        partes = []
        desc   = _val(row.get("descripcion"))
        cons   = _val(row.get("consideraciones"))
        fmt    = _val(row.get("formato_ejemplo"))
        rang   = _val(row.get("rango_o_valores"))

        if desc:
            partes.append(desc + ".")
        if cons:
            partes.append(f"Consideraciones: {cons}.")
        if fmt:
            partes.append(f"Ejemplo de formato: {fmt}.")
        if rang and rang.lower() != "ver diccionario":
            partes.append(f"Valores posibles: {rang}.")

        contexto = " ".join(partes) if partes else desc or ""
        campos.append({"campo": _val(row["atributo"]), "contexto": contexto})

    return campos


# ---------------------------------------------------------------------------
# Llamada al endpoint
# ---------------------------------------------------------------------------

def llamar_api(
    codigo_cotizacion: str,
    rut_proveedor: str,
    payload: dict,
    campos_manuales: list[dict],
) -> dict:
    body = {
        "payload": payload,
        "codigo_cotizacion": str(codigo_cotizacion),
        "rut_proveedor": str(rut_proveedor),
        "use_diccionarios": True,
        "campos_manuales": campos_manuales,
        "diccionario_similarity_threshold": 0.85,
        "diccionario_llm_fallback": True,
    }
    for intento in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_URL}/catalogar",
                headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logging.warning(f"  Timeout intento {intento}/{MAX_RETRIES} — {codigo_cotizacion}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 429:
                wait = 2 ** intento
                logging.warning(f"  Rate limit (429) intento {intento}/{MAX_RETRIES} — esperando {wait}s")
                time.sleep(wait)
            elif status in (502, 503, 504):
                wait = 5 * intento
                logging.warning(f"  HTTP {status} intento {intento}/{MAX_RETRIES} — esperando {wait}s")
                time.sleep(wait)
            else:
                logging.error(f"  HTTP {status} en {codigo_cotizacion}: {e}")
                return {"error": str(e), "status_code": status}
        except Exception as e:
            logging.error(f"  Error inesperado intento {intento}/{MAX_RETRIES} — {e}")
            if intento == MAX_RETRIES:
                return {"error": str(e)}
        time.sleep(1)
    return {"error": f"Fallido tras {MAX_RETRIES} intentos"}


# ---------------------------------------------------------------------------
# Worker: procesa un chunk del DataFrame
# ---------------------------------------------------------------------------

def worker(args: tuple):
    chunk_id: int
    df_chunk: pd.DataFrame
    campos_manuales: list[dict]
    out_path: Path
    chunk_id, df_chunk, campos_manuales, out_path = args

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker-{chunk_id:02d}] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / f"chunk_{chunk_id:02d}.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger()

    # Resume: cargar IDs ya procesados
    procesados: set = set()
    if out_path.exists():
        try:
            df_prev = pd.read_parquet(out_path)
            procesados = set(df_prev[ID_COL].astype(str).tolist())
            log.info(f"Resume: {len(procesados)} filas ya procesadas en {out_path.name}")
        except Exception as e:
            log.warning(f"No se pudo leer archivo previo ({e}), empezando desde cero")

    pendientes = df_chunk[~df_chunk[ID_COL].astype(str).isin(procesados)]
    total      = len(df_chunk)
    log.info(f"Chunk {chunk_id}: {len(pendientes)}/{total} filas pendientes")

    filas: list[dict] = []

    for i, (_, row) in enumerate(pendientes.iterrows(), 1):
        id_unico   = str(row[ID_COL])
        cod_cot    = str(row["codigo_requerimiento"])
        rut        = str(row["rut_proveedor"])

        payload = {
            "Categoria": CATEGORIA,
            "DescripcionProductoComprador": str(row.get("descripcion_requerimiento", "") or ""),
            "DescripcionProductoProveedor": str(row.get("descripcion_producto", "") or ""),
            "productoname": str(row.get("nombre_requerimiento", "") or ""),
        }

        t0   = time.time()
        data = llamar_api(cod_cot, rut, payload, campos_manuales)
        seg  = round(time.time() - t0, 2)

        if "error" in data:
            log.warning(f"  [{i}/{len(pendientes)}] {id_unico} — ERROR: {data['error']} ({seg}s)")
            resultado = {"_error": data["error"]}
        else:
            resultado = data.get("resultado") or {}
            log.info(f"  [{i}/{len(pendientes)}] {id_unico} — OK ({seg}s)")

        filas.append({ID_COL: id_unico, "resultado": json.dumps(resultado, ensure_ascii=False)})

        # Guardar cada 50 filas para no perder progreso
        if len(filas) % 50 == 0:
            _flush(filas, out_path, procesados, log, chunk_id)
            procesados.update(f[ID_COL] for f in filas)
            filas.clear()

    if filas:
        _flush(filas, out_path, procesados, log, chunk_id)

    log.info(f"Chunk {chunk_id} finalizado.")


def _flush(filas: list[dict], out_path: Path, procesados: set, log, chunk_id: int = -1):
    """Append filas nuevas al parquet existente y sube a Dropbox."""
    df_new = pd.DataFrame(filas)
    if out_path.exists():
        df_prev = pd.read_parquet(out_path)
        df_out  = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(out_path, index=False)
    log.info(f"  Guardado: {len(df_out)} filas en {out_path.name}")
    if chunk_id >= 0:
        _upload_dropbox(out_path, chunk_id, log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [main] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Leyendo attribute_descriptions.csv...")
    campos_manuales = build_campos_manuales(CSV_ATTRS)
    log.info(f"  {len(campos_manuales)} campos: {[c['campo'] for c in campos_manuales]}")

    log.info("Leyendo datos...")
    df = pd.read_csv(CSV_DATOS, sep=";", on_bad_lines="skip", dtype=str)
    df = df.dropna(subset=[ID_COL, "codigo_requerimiento", "rut_proveedor"])
    log.info(f"  {len(df)} filas cargadas")

    # Dividir en N_WORKERS chunks
    chunk_size = math.ceil(len(df) / N_WORKERS)
    chunks = [df.iloc[i: i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]
    log.info(f"  Dividido en {len(chunks)} chunks de ~{chunk_size} filas c/u")

    args = [
        (i, chunk, campos_manuales, RESULTS_DIR / f"chunk_{i:02d}.parquet")
        for i, chunk in enumerate(chunks)
    ]

    log.info(f"Iniciando {N_WORKERS} procesos...")
    t0 = time.time()
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        pool.map(worker, args)

    elapsed = round(time.time() - t0, 1)
    log.info(f"Todo completado en {elapsed}s ({elapsed/3600:.1f}h)")

    # Merge final opcional
    parquets = sorted(RESULTS_DIR.glob("chunk_*.parquet"))
    if parquets:
        df_final = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        out_final = RESULTS_DIR / f"resultados_completos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df_final.to_parquet(out_final, index=False)
        log.info(f"Merge final: {len(df_final)} filas → {out_final.name}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
