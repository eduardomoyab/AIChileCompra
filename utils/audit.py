"""
Auditoría de consumo por request: tabla public.auditoria_requests en PostgreSQL.
"""
import logging
import threading
from typing import Any, Dict

from sqlalchemy import create_engine, text

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS public.auditoria_requests (
    id                      BIGSERIAL    PRIMARY KEY,
    ts                      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    endpoint                TEXT         NOT NULL,
    categoria               TEXT,
    codigo_cotizacion       TEXT,
    rut_proveedor           TEXT,
    llm_provider            TEXT,
    con_adjuntos            BOOLEAN,
    num_atributos_extraidos INTEGER,
    tokens_llm_input        INTEGER,
    tokens_llm_output       INTEGER,
    tokens_llm_total        INTEGER,
    tokens_embedding_chars  INTEGER,
    duracion_segundos       NUMERIC(10,3),
    success                 BOOLEAN,
    num_errores             INTEGER,
    num_warnings            INTEGER,
    detalle                 JSONB
)
"""

_ALTER_DETALLE_DDL = """
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='auditoria_requests'
          AND column_name='detalle'
    ) THEN
        ALTER TABLE public.auditoria_requests ADD COLUMN detalle JSONB;
    END IF;
END $$;
"""

_INSERT_SQL = """
INSERT INTO public.auditoria_requests (
    endpoint, categoria, codigo_cotizacion, rut_proveedor, llm_provider,
    con_adjuntos, num_atributos_extraidos,
    tokens_llm_input, tokens_llm_output, tokens_llm_total,
    tokens_embedding_chars, duracion_segundos,
    success, num_errores, num_warnings, detalle
) VALUES (
    :endpoint, :categoria, :codigo_cotizacion, :rut_proveedor, :llm_provider,
    :con_adjuntos, :num_atributos_extraidos,
    :tokens_llm_input, :tokens_llm_output, :tokens_llm_total,
    :tokens_embedding_chars, :duracion_segundos,
    :success, :num_errores, :num_warnings, :detalle
)
"""

_engine = None
_engine_lock = threading.Lock()


def _get_engine(db_url: str):
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


def crear_tabla_auditoria(db_url: str) -> None:
    if not db_url:
        return
    try:
        engine = _get_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text(_AUDIT_DDL))
            conn.execute(text(_ALTER_DETALLE_DDL))
            conn.commit()
        logging.info("✓ Tabla public.auditoria_requests verificada/creada")
    except Exception as e:
        logging.warning(f"No se pudo crear tabla de auditoría: {e}")


def registrar_auditoria(db_url: str, data: Dict[str, Any]) -> None:
    if not db_url:
        return
    try:
        engine = _get_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text(_INSERT_SQL), data)
            conn.commit()
    except Exception as e:
        logging.warning(f"Error registrando auditoría: {e}")


def contar_atributos_extraidos(resultado: dict) -> int:
    """Cuenta campos con valor real (excluye ROWNUM y valores vacíos/no disponibles)."""
    VACIOS = {"no disponible", "no especificado", "false", "", "none"}
    return sum(
        1 for k, v in resultado.items()
        if k != "ROWNUM"
        and v is not None
        and str(v).strip().lower() not in VACIOS
    )
