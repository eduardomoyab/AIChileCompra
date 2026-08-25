"""
retriever_diccionario.py

Diccionario genérico de normalización y complementos.
Lee desde dos CSV planos en diccionarios/ Y desde la tabla DB diccionarios_valores:
  - attribute_dictionary.csv  (categoria, atributo, valor, fuente)
  - attribute_complement.csv  (categoria, atributo, valor, atributo_complementario, complemento)
  - public.diccionarios_valores  (mismo esquema: categoria, atributo, valor, fuente)

El FAISS de cada par (categoria, atributo) se construye con la UNIÓN de ambas fuentes.
Se persiste en disco; un manifest.txt guarda el count para detectar cambios en DB.

API pública:
  warm_dictionaries()                                           → pre-carga en startup
  normalizar_con_diccionario(categoria, atributo, valor, ...)  → normaliza un valor
  get_atributos_en_diccionario(categoria)                       → atributos disponibles
  get_complementos(categoria, atributo, valor)                  → atributos derivados
  agregar_valor_db(categoria, atributo, valor, fuente, forzar)  → agrega al diccionario online (con validación LLM)
  validar_valor_llm(categoria, atributo, valor)                 → (aprobado, razon) via LLM
  invalidar_cache_par(categoria, atributo)                      → fuerza reconstrucción del FAISS
"""

import os
import shutil
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Set

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DICT_PATH = os.path.join(ROOT, "diccionarios", "attribute_dictionary.csv")
CSV_COMP_PATH = os.path.join(ROOT, "diccionarios", "attribute_complement.csv")
FAISS_DISK_DIR = os.path.join(ROOT, "cache", "faiss_dict")

# ─── Caches ───────────────────────────────────────────────────────────────────
_dict_vs_cache: Dict[str, Any] = {}          # f"{categoria}::{atributo}" → FAISSStore
_dict_df: Optional[pd.DataFrame] = None      # attribute_dictionary.csv en memoria
_comp_df: Optional[pd.DataFrame] = None      # attribute_complement.csv en memoria

# ─── DB singleton ─────────────────────────────────────────────────────────────
_db_engine = None
_db_engine_lock = threading.Lock()

_ONLINE_DDL = """
CREATE TABLE IF NOT EXISTS public.diccionarios_valores (
    id        BIGSERIAL    PRIMARY KEY,
    categoria TEXT         NOT NULL,
    atributo  TEXT         NOT NULL,
    valor     TEXT         NOT NULL,
    fuente    TEXT         NOT NULL DEFAULT 'usuario',
    ts        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT diccionarios_valores_uq UNIQUE (categoria, atributo, valor)
)
"""


def _get_db_engine():
    global _db_engine
    if _db_engine is None:
        with _db_engine_lock:
            if _db_engine is None:
                db_url = os.getenv("DATABASE_URL", "")
                if not db_url:
                    return None
                from sqlalchemy import create_engine
                _db_engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
    return _db_engine


def _ensure_online_table() -> None:
    engine = _get_db_engine()
    if not engine:
        return
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(_ONLINE_DDL))
            conn.commit()
    except Exception as e:
        logging.warning(f"[dict_online] No se pudo crear tabla diccionarios_valores: {e}")


def _load_db_valores(categoria: str, atributo: str) -> List[str]:
    engine = _get_db_engine()
    if not engine:
        return []
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT valor FROM public.diccionarios_valores "
                "WHERE categoria = :cat AND atributo = :atr"
            ), {"cat": categoria, "atr": atributo}).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logging.warning(f"[dict_online] Error cargando DB para '{categoria}::{atributo}': {e}")
        return []


def _get_db_pairs() -> Set[Tuple[str, str]]:
    engine = _get_db_engine()
    if not engine:
        return set()
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT categoria, atributo FROM public.diccionarios_valores"
            )).fetchall()
        return {(r[0], r[1]) for r in rows}
    except Exception as e:
        logging.warning(f"[dict_online] Error obteniendo pares de DB: {e}")
        return set()


# ─── CSV loading ──────────────────────────────────────────────────────────────

def _read_csv_safe(path: str, fallback_columns: list) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logging.debug(f"CSV cargado con encoding={enc}: {path}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"Error leyendo {path} con encoding={enc}: {e}")
            break
    logging.warning(f"No se pudo leer {path} — devolviendo DataFrame vacío")
    return pd.DataFrame(columns=fallback_columns)


def _load_csvs() -> None:
    global _dict_df, _comp_df
    if _dict_df is None:
        if os.path.exists(CSV_DICT_PATH):
            _dict_df = _read_csv_safe(CSV_DICT_PATH, ["categoria", "atributo", "valor", "fuente"])
            logging.info(f"attribute_dictionary.csv cargado: {len(_dict_df)} filas")
        else:
            logging.warning(f"No se encontró {CSV_DICT_PATH}")
            _dict_df = pd.DataFrame(columns=["categoria", "atributo", "valor", "fuente"])
    if _comp_df is None:
        if os.path.exists(CSV_COMP_PATH):
            _comp_df = _read_csv_safe(CSV_COMP_PATH, ["categoria", "atributo", "valor", "atributo_complementario", "complemento"])
            logging.info(f"attribute_complement.csv cargado: {len(_comp_df)} filas")
        else:
            logging.warning(f"No se encontró {CSV_COMP_PATH}")
            _comp_df = pd.DataFrame(columns=["categoria", "atributo", "valor", "atributo_complementario", "complemento"])


# ─── FAISS disk helpers ───────────────────────────────────────────────────────

def _cache_key(categoria: str, atributo: str) -> str:
    return f"{categoria}::{atributo}"


def _disk_path(categoria: str, atributo: str) -> str:
    safe = f"{categoria}__{atributo}".replace(" ", "_").replace("/", "-")
    return os.path.join(FAISS_DISK_DIR, safe)


def _manifest_path(disk_path: str) -> str:
    return os.path.join(disk_path, "_manifest.txt")


def _read_manifest(disk_path: str) -> int:
    """Lee el count de valores almacenado en el manifest del FAISS en disco. -1 si no existe."""
    try:
        with open(_manifest_path(disk_path)) as f:
            return int(f.read().strip())
    except Exception:
        return -1


def _write_manifest(disk_path: str, count: int) -> None:
    try:
        with open(_manifest_path(disk_path), "w") as f:
            f.write(str(count))
    except Exception:
        pass


def _build_vs_for(categoria: str, atributo: str) -> Optional[Any]:
    """
    Crea y cachea un FAISS para el par (categoria, atributo).
    Combina valores del CSV local y del diccionario online (DB).
    Usa cache en disco si está vigente (mismo count de valores); si no, reconstruye.
    """
    _load_csvs()

    from langchain_community.vectorstores import FAISS as FAISSStore
    from langchain_core.documents import Document as LCDocument
    from agents.get_vectorstore import _get_embeddings

    embeddings = _get_embeddings()
    disk_path = _disk_path(categoria, atributo)
    key = _cache_key(categoria, atributo)

    # Calcular valores actuales (CSV + DB) para comparar con el manifest
    mask = (_dict_df["categoria"] == categoria) & (_dict_df["atributo"] == atributo)
    valores_csv = list(_dict_df[mask]["valor"].dropna().str.strip())
    valores_db  = _load_db_valores(categoria, atributo)
    current_count = len(valores_csv) + len(valores_db)

    # Intentar cargar desde disco si el manifest coincide
    if os.path.isdir(disk_path) and _read_manifest(disk_path) == current_count:
        try:
            vs = FAISSStore.load_local(disk_path, embeddings, allow_dangerous_deserialization=True)
            _dict_vs_cache[key] = vs
            logging.info(f"  [FAISS] '{key}' cargado desde disco ({current_count} valores)")
            return vs
        except Exception as e:
            logging.warning(f"  [FAISS] Error cargando '{key}' desde disco ({e}), reconstruyendo")

    # Combinar y deduplicar (case-insensitive; DB tiene precedencia en colisiones de case)
    seen_lower: Dict[str, str] = {}
    for v in sorted(valores_csv):
        seen_lower[v.lower()] = v
    for v in sorted(valores_db):
        seen_lower[v.lower()] = v
    valores = list(seen_lower.values())

    if not valores:
        return None

    docs = [
        LCDocument(page_content=v, metadata={"categoria": categoria, "atributo": atributo, "valor": v})
        for v in valores
    ]
    vs = FAISSStore.from_documents(docs, embeddings)

    try:
        os.makedirs(disk_path, exist_ok=True)
        vs.save_local(disk_path)
        _write_manifest(disk_path, current_count)
        logging.info(f"  [FAISS] '{key}' guardado en disco ({len(valores)} únicos, {len(valores_db)} de DB)")
    except Exception as e:
        logging.warning(f"  [FAISS] No se pudo guardar '{key}' en disco: {e}")

    _dict_vs_cache[key] = vs
    return vs


def invalidar_cache_par(categoria: str, atributo: str) -> None:
    """Elimina el FAISS en disco y en memoria para (categoria, atributo). Se reconstruirá en la próxima llamada."""
    key = _cache_key(categoria, atributo)
    _dict_vs_cache.pop(key, None)
    disk_path = _disk_path(categoria, atributo)
    if os.path.isdir(disk_path):
        try:
            shutil.rmtree(disk_path)
            logging.info(f"  [FAISS] cache eliminada para '{key}' (se reconstruirá con valores DB)")
        except Exception as e:
            logging.warning(f"  [FAISS] No se pudo eliminar cache de disco para '{key}': {e}")


_VALIDACION_PROMPT = """\
Eres un validador de diccionarios de normalización para licitaciones públicas en Chile.

Diccionario: categoría "{categoria}", atributo "{atributo}"
Propósito: normalizar valores de este atributo a sus formas canónicas estándar.

Muestra de valores ya existentes en el diccionario ({n_existentes} valores totales):
{muestra_existentes}

Valor propuesto para agregar: "{valor}"

¿Deberías agregar este valor al diccionario como entrada canónica?

Responde ÚNICAMENTE con uno de estos formatos (sin texto adicional):
APROBAR: <razón breve en español>
RECHAZAR: <razón breve en español>

Criterios para RECHAZAR:
- Es vago, aproximado o descriptivo (ej: "similar a X", "tipo Y", "aprox")
- Ya existe un valor equivalente o casi idéntico en la muestra
- No corresponde claramente a este atributo o categoría
- Es demasiado genérico, incompleto o parece un error tipográfico

Criterios para APROBAR:
- Es un valor real, específico y estandarizado (nombre de producto, modelo, estándar, etc.)
- No hay un equivalente ya en el diccionario (aporta algo nuevo)
- Pertenece claramente a este atributo y categoría\
"""


_SIMILARITY_PRECHECK_THRESHOLD = 0.92  # Score mínimo para considerar que ya existe un equivalente


def validar_valor_llm(
    categoria: str,
    atributo: str,
    valor: str,
    llm_provider: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Valida si un valor es una entrada canónica válida para (categoria, atributo).

    Flujo:
      1. Similitud contra el diccionario actual: si score >= 0.92, rechaza de inmediato
         (ya existe un equivalente — no necesita LLM).
      2. Si no hay match alto, el LLM decide si el valor es canónico válido.

    Retorna (aprobado: bool, razon: str).
    """
    _load_csvs()

    # ── Paso 1: pre-check de similitud ────────────────────────────────────────
    # Carga (o reconstruye) el FAISS del par para comparar el valor propuesto
    # contra las entradas ya existentes en CSV + DB.
    vs = _dict_vs_cache.get(_cache_key(categoria, atributo))
    if vs is None:
        vs = _build_vs_for(categoria, atributo)

    if vs is not None:
        try:
            resultados = vs.similarity_search_with_relevance_scores(valor, k=1)
            if resultados:
                mejor_doc, mejor_score = resultados[0]
                mejor_valor = mejor_doc.metadata.get("valor", mejor_doc.page_content)
                if mejor_score >= _SIMILARITY_PRECHECK_THRESHOLD:
                    return False, (
                        f"Ya existe un valor equivalente: '{mejor_valor}' "
                        f"(similitud={mejor_score:.3f} ≥ {_SIMILARITY_PRECHECK_THRESHOLD}). "
                        f"No es necesario agregar '{valor}'."
                    )
        except Exception as e:
            logging.warning(f"  [dict_llm] Error en pre-check de similitud: {e}")
    # ── Paso 2: validación LLM ────────────────────────────────────────────────

    # Obtener muestra de valores existentes (CSV + DB, hasta 20)
    mask = (_dict_df["categoria"] == categoria) & (_dict_df["atributo"] == atributo)
    valores_csv = _dict_df[mask]["valor"].dropna().tolist()
    valores_db  = _load_db_valores(categoria, atributo)
    todos = list(dict.fromkeys(valores_csv + valores_db))  # dedup preservando orden
    muestra = todos[:20]
    muestra_str = "\n".join(f"  - {v}" for v in muestra) if muestra else "  (diccionario vacío)"

    prompt = _VALIDACION_PROMPT.format(
        categoria=categoria,
        atributo=atributo,
        valor=valor,
        n_existentes=len(todos),
        muestra_existentes=muestra_str,
    )

    try:
        provider = llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        texto = resp.content.strip()
        logging.info(f"  [dict_llm] Validación '{categoria}::{atributo}' = '{valor}': {texto}")

        if texto.upper().startswith("APROBAR"):
            razon = texto.split(":", 1)[1].strip() if ":" in texto else texto
            return True, razon
        elif texto.upper().startswith("RECHAZAR"):
            razon = texto.split(":", 1)[1].strip() if ":" in texto else texto
            return False, razon
        else:
            # Respuesta inesperada — rechazar por precaución
            return False, f"Respuesta LLM no interpretable: {texto[:120]}"
    except Exception as e:
        logging.error(f"  [dict_llm] Error en validación LLM: {e}")
        # Si el LLM falla, rechazar por precaución (no agregar silenciosamente)
        return False, f"Error al consultar LLM: {str(e)}"


def agregar_valor_db(
    categoria: str,
    atributo: str,
    valor: str,
    fuente: str = "usuario",
    forzar: bool = False,
    llm_provider: Optional[str] = None,
) -> Tuple[bool, bool, str]:
    """
    Agrega un valor al diccionario online (tabla DB), con validación LLM previa.

    Args:
        forzar: Si True, omite la validación LLM e inserta directamente.
        llm_provider: Proveedor LLM para validación ('openai', 'gemini'). None = DEFAULT_LLM_PROVIDER.

    Returns:
        (insertado, llm_aprobado, razon)
        - insertado: True si se insertó en DB
        - llm_aprobado: True si el LLM aprobó (o forzar=True)
        - razon: Explicación del LLM (o motivo de rechazo)
    """
    valor = valor.strip()
    if not valor:
        return False, False, "Valor vacío"

    # Validación LLM (salvo forzar)
    if not forzar:
        aprobado, razon = validar_valor_llm(categoria, atributo, valor, llm_provider)
        if not aprobado:
            logging.info(f"  [dict_online] Rechazado por LLM: '{valor}' → {razon}")
            return False, False, razon
    else:
        aprobado, razon = True, "Forzado por usuario (sin validación LLM)"

    engine = _get_db_engine()
    if not engine:
        logging.error("[dict_online] Sin DATABASE_URL — no se puede agregar valor")
        return False, aprobado, "Sin DATABASE_URL"
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO public.diccionarios_valores (categoria, atributo, valor, fuente)
                VALUES (:cat, :atr, :val, :fuente)
                ON CONFLICT (categoria, atributo, valor) DO NOTHING
                RETURNING id
            """), {"cat": categoria, "atr": atributo, "val": valor, "fuente": fuente})
            insertado = result.fetchone() is not None
            conn.commit()
        if insertado:
            invalidar_cache_par(categoria, atributo)
            logging.info(f"  [dict_online] Agregado '{valor}' → '{categoria}::{atributo}'")
        else:
            logging.info(f"  [dict_online] Ya existía '{valor}' en '{categoria}::{atributo}'")
        return insertado, aprobado, razon
    except Exception as e:
        logging.error(f"  [dict_online] Error insertando en DB: {e}")
        return False, aprobado, f"Error DB: {str(e)}"


def listar_valores_db(categoria: Optional[str] = None, atributo: Optional[str] = None) -> List[Dict]:
    """Lista las entradas del diccionario online. Filtra por categoria y/o atributo si se proveen."""
    engine = _get_db_engine()
    if not engine:
        return []
    try:
        from sqlalchemy import text
        where_parts = []
        params: Dict = {}
        if categoria:
            where_parts.append("categoria = :cat")
            params["cat"] = categoria
        if atributo:
            where_parts.append("atributo = :atr")
            params["atr"] = atributo
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        with engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT id, categoria, atributo, valor, fuente, ts FROM public.diccionarios_valores {where} ORDER BY categoria, atributo, valor"
            ), params).fetchall()
        return [{"id": r[0], "categoria": r[1], "atributo": r[2], "valor": r[3], "fuente": r[4], "ts": str(r[5])} for r in rows]
    except Exception as e:
        logging.warning(f"[dict_online] Error listando valores: {e}")
        return []


# ─── API pública ──────────────────────────────────────────────────────────────

def warm_dictionaries() -> None:
    """
    Pre-carga en cache un FAISS por cada (categoria, atributo) del CSV + DB.
    Llama a _ensure_online_table() para crear la tabla si no existe.
    """
    _load_csvs()
    _ensure_online_table()

    # Pares desde CSV
    pares_csv: Set[Tuple[str, str]] = set(
        tuple(x) for x in _dict_df[["categoria", "atributo"]].drop_duplicates().values.tolist()
    )
    # Pares desde DB (pueden incluir categorías nuevas no en CSV)
    pares_db = _get_db_pairs()
    todos_pares = pares_csv | pares_db

    total = 0
    for (categoria, atributo) in sorted(todos_pares):
        try:
            vs = _build_vs_for(categoria, atributo)
            if vs is not None:
                total += 1
        except Exception as e:
            logging.error(f"  ✗ Error cargando FAISS para '{categoria}::{atributo}': {e}")

    logging.info(f"warm_dictionaries() completado — {total} pares cargados ({len(pares_db)} pares con valores en DB)")


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
