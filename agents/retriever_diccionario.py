"""
retriever_diccionario.py

Diccionario genérico de normalización y complementos.
Lee desde dos CSV planos en diccionarios/:
  - attribute_dictionary.csv  (categoria, atributo, valor, fuente)
  - attribute_complement.csv  (categoria, atributo, valor, atributo_complementario, complemento)

API pública:
  warm_dictionaries()                                           → pre-carga en startup
  normalizar_con_diccionario(categoria, atributo, valor, ...)  → normaliza un valor
  get_atributos_en_diccionario(categoria)                       → atributos disponibles
  get_complementos(categoria, atributo, valor)                  → atributos derivados
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DICT_PATH = os.path.join(ROOT, "diccionarios", "attribute_dictionary.csv")
CSV_COMP_PATH = os.path.join(ROOT, "diccionarios", "attribute_complement.csv")

# ─── Caches ───────────────────────────────────────────────────────────────────
_dict_vs_cache: Dict[str, Any] = {}          # f"{categoria}::{atributo}" → FAISSStore
_dict_df: Optional[pd.DataFrame] = None      # attribute_dictionary.csv en memoria
_comp_df: Optional[pd.DataFrame] = None      # attribute_complement.csv en memoria


def _load_csvs() -> None:
    global _dict_df, _comp_df
    if _dict_df is None:
        if os.path.exists(CSV_DICT_PATH):
            _dict_df = pd.read_csv(CSV_DICT_PATH)
            logging.info(f"attribute_dictionary.csv cargado: {len(_dict_df)} filas")
        else:
            logging.warning(f"No se encontró {CSV_DICT_PATH}")
            _dict_df = pd.DataFrame(columns=["categoria", "atributo", "valor", "fuente"])
    if _comp_df is None:
        if os.path.exists(CSV_COMP_PATH):
            _comp_df = pd.read_csv(CSV_COMP_PATH)
            logging.info(f"attribute_complement.csv cargado: {len(_comp_df)} filas")
        else:
            logging.warning(f"No se encontró {CSV_COMP_PATH}")
            _comp_df = pd.DataFrame(columns=["categoria", "atributo", "valor", "atributo_complementario", "complemento"])


FAISS_DISK_DIR = os.path.join(ROOT, "cache", "faiss_dict")


def _cache_key(categoria: str, atributo: str) -> str:
    return f"{categoria}::{atributo}"


def _disk_path(categoria: str, atributo: str) -> str:
    """Directorio en disco para el FAISS de un par (categoria, atributo)."""
    safe = f"{categoria}__{atributo}".replace(" ", "_").replace("/", "-")
    return os.path.join(FAISS_DISK_DIR, safe)


def _build_vs_for(categoria: str, atributo: str) -> Optional[Any]:
    """Crea y cachea un FAISS para el par (categoria, atributo).
    Primero intenta cargar desde disco; si no existe, lo construye y lo persiste."""
    _load_csvs()

    from langchain_community.vectorstores import FAISS as FAISSStore
    from langchain_core.documents import Document as LCDocument
    from agents.get_vectorstore import _get_embeddings

    embeddings = _get_embeddings()

    disk_path = _disk_path(categoria, atributo)
    key = _cache_key(categoria, atributo)

    # Intentar cargar desde disco
    if os.path.isdir(disk_path):
        try:
            vs = FAISSStore.load_local(disk_path, embeddings, allow_dangerous_deserialization=True)
            _dict_vs_cache[key] = vs
            logging.info(f"  [FAISS] '{key}' cargado desde disco")
            return vs
        except Exception as e:
            logging.warning(f"  [FAISS] Error cargando '{key}' desde disco ({e}), reconstruyendo")

    mask = (_dict_df["categoria"] == categoria) & (_dict_df["atributo"] == atributo)
    valores = _dict_df[mask]["valor"].dropna().tolist()
    if not valores:
        return None

    docs = [
        LCDocument(page_content=v, metadata={"categoria": categoria, "atributo": atributo, "valor": v})
        for v in valores
    ]
    vs = FAISSStore.from_documents(docs, embeddings)

    # Persistir en disco para futuros workers/reinicios
    try:
        os.makedirs(disk_path, exist_ok=True)
        vs.save_local(disk_path)
        logging.info(f"  [FAISS] '{key}' guardado en disco ({len(valores)} valores)")
    except Exception as e:
        logging.warning(f"  [FAISS] No se pudo guardar '{key}' en disco: {e}")

    _dict_vs_cache[key] = vs
    return vs


def warm_dictionaries() -> None:
    """
    Pre-carga en cache un FAISS por cada (categoria, atributo) del attribute_dictionary.csv.
    Carga desde disco si ya existe (sin llamar a OpenAI); si no, construye y persiste.
    Se llama una vez en startup por worker.
    """
    _load_csvs()

    grupos = _dict_df.groupby(["categoria", "atributo"])
    total = 0
    for (categoria, atributo), _ in grupos:
        try:
            vs = _build_vs_for(categoria, atributo)
            if vs is not None:
                total += 1
        except Exception as e:
            logging.error(f"  ✗ Error cargando FAISS para '{categoria}::{atributo}': {e}")

    logging.info(f"warm_dictionaries() completado — {total} pares cargados")


def normalizar_con_diccionario(
    categoria: str,
    atributo: str,
    valor: str,
    k: int = 3,
    threshold: float = 0.85,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Busca el valor normalizado más similar en el diccionario para un (categoria, atributo).

    Returns:
        (valor_normalizado, score, candidatos)
    """
    key = _cache_key(categoria, atributo)
    vs = _dict_vs_cache.get(key)
    if vs is None:
        logging.warning(f"  ⚠ FAISS para '{key}' no está en cache — cargando on-demand")
        vs = _build_vs_for(categoria, atributo)

    if vs is None:
        logging.warning(f"  ✗ Sin FAISS para '{key}', campo NO normalizado")
        return valor, 0.0, []

    try:
        resultados = vs.similarity_search_with_relevance_scores(valor, k=k)
    except Exception as e:
        logging.warning(f"Error en similarity_search para '{key}': {e}")
        return valor, 0.0, []

    if not resultados:
        return valor, 0.0, []

    candidatos = [(doc.metadata.get("valor", doc.page_content), score) for doc, score in resultados]
    best_valor, best_score = candidatos[0]

    if best_score >= threshold:
        logging.info(f"  [Dict] '{key}': '{valor}' → '{best_valor}' (score={best_score:.3f})")
        return best_valor, best_score, candidatos
    else:
        logging.info(f"  [Dict] '{key}': '{valor}' sin match automático (best score={best_score:.3f})")
        return valor, best_score, candidatos


def get_atributos_en_diccionario(categoria: str) -> List[str]:
    """Devuelve los atributos disponibles en el diccionario para una categoria."""
    prefix = f"{categoria}::"
    return [k.split("::")[1] for k in _dict_vs_cache if k.startswith(prefix)]


def get_complementos(categoria: str, atributo: str, valor: str) -> Dict[str, str]:
    """
    Lookup de atributos derivados para un (categoria, atributo, valor).

    Ejemplo:
        get_complementos("Computadores", "procesador_principal", "Intel Core i7-1355U")
        → {"nucleos_procesador": "10", "hilos_procesador": "12"}

    Returns:
        Dict atributo_complementario → complemento (como str), vacío si no hay match.
    """
    _load_csvs()
    if _comp_df is None or _comp_df.empty:
        return {}

    mask = (
        (_comp_df["categoria"] == categoria) &
        (_comp_df["atributo"] == atributo) &
        (_comp_df["valor"] == valor)
    )
    rows = _comp_df[mask]
    if rows.empty:
        return {}

    return {
        str(row["atributo_complementario"]): str(int(row["complemento"]) if float(row["complemento"]) == int(float(row["complemento"])) else row["complemento"])
        for _, row in rows.iterrows()
    }
