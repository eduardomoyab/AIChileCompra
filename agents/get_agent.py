import os
import itertools
import threading
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
#from langchain_deepseek import ChatDeepSeek
# Cargar variables de entorno
load_dotenv()

# Round-robin entre OPENAI_API_KEY y OPENAI_API_KEY_2 (si existe)
_openai_keys_lock = threading.Lock()
_openai_keys_cycle: Optional[itertools.cycle] = None

# Cache de instancias LLM por (provider, temperature) para evitar recrear objetos por request
_llm_cache: dict = {}
_llm_cache_lock = threading.Lock()

# ── Contador de tokens por thread ──────────────────────────────────────────────
_thread_tokens = threading.local()


def _get_counter() -> dict:
    if not hasattr(_thread_tokens, "c"):
        _thread_tokens.c = {"input": 0, "output": 0, "embedding_chars": 0}
    return _thread_tokens.c


def reset_token_counter() -> None:
    _thread_tokens.c = {"input": 0, "output": 0, "embedding_chars": 0}


def get_token_counter() -> dict:
    return dict(_get_counter())


def add_embedding_chars(n: int) -> None:
    _get_counter()["embedding_chars"] += n


class _TokenCallback(BaseCallbackHandler):
    """Acumula tokens de cada llamada LLM en el contador del thread actual."""

    def on_llm_end(self, response, **kwargs):
        try:
            # Intento 1: llm_output estándar de OpenAI
            if response.llm_output:
                usage = (
                    response.llm_output.get("token_usage")
                    or response.llm_output.get("usage")
                    or {}
                )
                if usage:
                    c = _get_counter()
                    c["input"] += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                    c["output"] += usage.get("completion_tokens") or usage.get("output_tokens") or 0
                    return
            # Intento 2: usage_metadata en AIMessage (Gemini y OpenAI recientes)
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        c = _get_counter()
                        c["input"] += msg.usage_metadata.get("input_tokens", 0)
                        c["output"] += msg.usage_metadata.get("output_tokens", 0)
        except Exception:
            pass


_token_callback = _TokenCallback()

def _get_openai_key_roundrobin() -> str:
    global _openai_keys_cycle
    with _openai_keys_lock:
        if _openai_keys_cycle is None:
            # Soporta múltiples keys separadas por coma en OPENAI_API_KEY
            raw = os.getenv("OPENAI_API_KEY", "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            # Compatibilidad con variable separada OPENAI_API_KEY_2
            extra = (os.getenv("OPENAI_API_KEY_2") or "").strip()
            if extra and extra not in keys:
                keys.append(extra)
            if not keys:
                raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")
            _openai_keys_cycle = itertools.cycle(keys)
        return next(_openai_keys_cycle)


def _build_llm(model_provider: str, temperature: float):
    """Construye una instancia LLM nueva (uso interno, sin cache)."""
    if model_provider == "openai":
        api_key = _get_openai_key_roundrobin()
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

    elif model_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env")
        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)

    # elif model_provider == "deepseek":
    #     api_key = os.getenv("DEEPSEEK_API_KEY")
    #     if not api_key:
    #         raise ValueError("DEEPSEEK_API_KEY no está configurada en el archivo .env")
    #     model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    #     base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    #     return ChatDeepSeek(model=model_name, temperature=temperature,
    #                         deepseek_api_key=api_key, base_url=base_url)

    else:
        raise ValueError(
            f"Proveedor '{model_provider}' no válido. "
            f"Opciones válidas: 'openai', 'gemini', 'deepseek'"
        )


def get_llm(model_provider: str = "openai", temperature: float = 0.0):
    """
    Retorna una instancia del modelo LLM según el proveedor especificado.
    Las instancias se cachean por (provider, temperature) para reutilizar
    el cliente HTTP y evitar recrear objetos en cada request.

    Args:
        model_provider (str): Proveedor del modelo ('openai', 'gemini', 'deepseek')
        temperature (float): Temperatura para el modelo (0.0 - 1.0)

    Returns:
        Instancia del modelo LLM de LangChain

    Raises:
        ValueError: Si el proveedor no es válido o faltan credenciales
    """
    model_provider = model_provider.lower()
    key = (model_provider, temperature)
    with _llm_cache_lock:
        if key not in _llm_cache:
            _llm_cache[key] = _build_llm(model_provider, temperature)
        base_llm = _llm_cache[key]
    # with_config es barato: devuelve un RunnableBinding que reutiliza el cliente HTTP
    # cacheado pero añade el callback de tokens al invoke de este thread
    return base_llm.with_config(callbacks=[_token_callback])


