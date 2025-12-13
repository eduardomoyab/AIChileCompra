import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_deepseek import ChatDeepSeek
# Cargar variables de entorno
load_dotenv()


def get_llm(model_provider: str = "openai", temperature: float = 0.0):
    """
    Retorna una instancia del modelo LLM segán el proveedor especificado.

    Args:
        model_provider (str): Proveedor del modelo ('openai', 'gemini', 'deepseek')
        temperature (float): Temperatura para el modelo (0.0 - 1.0)

    Returns:
        Instancia del modelo LLM de LangChain

    Raises:
        ValueError: Si el proveedor no es válido o faltan credenciales
    """
    model_provider = model_provider.lower()

    if model_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    elif model_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env")

        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    # elif model_provider == "deepseek":
    #     api_key = os.getenv("DEEPSEEK_API_KEY")
    #     if not api_key:
    #         raise ValueError("DEEPSEEK_API_KEY no está configurada en el archivo .env")

    #     model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    #     base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    #     return ChatDeepSeek(
    #         model=model_name,
    #         temperature=temperature,
    #         deepseek_api_key=api_key,
    #         base_url=base_url
    #     )

    else:
        raise ValueError(
            f"Proveedor '{model_provider}' no válido. "
            f"Opciones válidas: 'openai', 'gemini', 'deepseek'"
        )


