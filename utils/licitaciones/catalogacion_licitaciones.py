"""
Módulo de catalogación para licitaciones usando un único LLM con RAG.

A diferencia del sistema de agentes de compra_agil, este módulo usa
un enfoque más directo con:
1. Vectorstore de adjuntos procesados
2. Un único LLM para extraer atributos
3. Opcionalmente, normalización con diccionarios
"""

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def get_llm(llm_provider: str = None):
    """
    Obtiene el LLM según el proveedor especificado.

    Args:
        llm_provider: 'openai', 'gemini', 'deepseek' o None (usa DEFAULT_LLM_PROVIDER)

    Returns:
        LLM instance
    """
    if llm_provider is None:
        llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

    if llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif llm_provider == "deepseek":
        # DeepSeek usa OpenAI-compatible API
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        raise ValueError(f"LLM provider no soportado: {llm_provider}")


def create_vectorstore_from_processed_files(archivos_procesados: List[Dict[str, str]]):
    """
    Crea un vectorstore a partir de archivos procesados.

    Args:
        archivos_procesados: Lista de dicts con 'output_path' y 'text'

    Returns:
        FAISS vectorstore
    """
    if not archivos_procesados:
        logging.warning("No hay archivos procesados para crear vectorstore")
        return None

    try:
        # Cargar documentos
        documents = []
        for archivo in archivos_procesados:
            loader = TextLoader(archivo['output_path'], encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)

        if not documents:
            return None

        # Split en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Crear vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        vectorstore = FAISS.from_documents(splits, embeddings)

        logging.info(f"Vectorstore creado con {len(splits)} chunks de {len(documents)} documentos")

        return vectorstore

    except Exception as e:
        logging.exception(f"Error creando vectorstore: {e}")
        return None


def format_docs(docs):
    """Formatea documentos para el contexto."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_campos_json(campos_manuales: List[str] = None) -> str:
    """Construye el bloque JSON esperado según los campos solicitados."""
    if campos_manuales:
        lines = [f'    "{campo}": "..."' for campo in campos_manuales]
    else:
        lines = [
            '    "marca": "..."',
            '    "modelo": "..."',
            '    "procesador": "..."',
            '    "ram": "..."',
            '    "almacenamiento": "..."',
            '    "pantalla": "..."',
            '    "sistema_operativo": "..."',
            '    "otros_atributos": {}',
        ]
    return "{{\n" + ",\n".join(lines) + "\n}}"


def catalogar_licitacion(
    payload: Dict[str, Any],
    codigo_licitacion: str,
    rut_proveedor: str,
    archivos_procesados: List[Dict[str, str]],
    use_diccionarios: bool = True,
    llm_provider: str = None,
    campos_manuales: List[str] = None
) -> Dict[str, Any]:
    """
    Cataloga un producto de licitación usando RAG + LLM.

    Args:
        payload: Info del producto
        codigo_licitacion: Código de la licitación
        rut_proveedor: RUT del proveedor
        archivos_procesados: Lista de archivos procesados
        use_diccionarios: Si usar normalización con diccionarios
        llm_provider: Proveedor de LLM

    Returns:
        dict: Resultado de la catalogación con atributos extraídos
    """
    try:
        # Crear vectorstore
        vectorstore = create_vectorstore_from_processed_files(archivos_procesados)

        if not vectorstore:
            logging.warning("No se pudo crear vectorstore, catalogando solo con descripciones")
            return catalogar_sin_adjuntos(payload, llm_provider, campos_manuales)

        # Crear retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Construir bloque JSON de salida según campos solicitados
        campos_json = _build_campos_json(campos_manuales)

        # Crear prompt
        template = """Eres un experto en catalogación de productos para licitaciones públicas.

Tu tarea es extraer atributos técnicos del producto basándote en las descripciones y documentos adjuntos.

PRODUCTO:
- Categoría: {categoria}
- Nombre: {productoname}
- Descripción Comprador: {desc_comprador}
- Descripción Proveedor: {desc_proveedor}

DOCUMENTOS ADJUNTOS:
{context}

INSTRUCCIONES:
1. Extrae los atributos técnicos solicitados que encuentres en los documentos
2. Prioriza información de los documentos adjuntos sobre las descripciones
3. Si no encuentras un atributo, usa "No disponible"
4. Sé específico y preciso con unidades y valores

Responde ÚNICAMENTE con un JSON válido con los atributos encontrados:
""" + campos_json + "\n"

        prompt = PromptTemplate.from_template(template)

        # Crear LLM
        llm = get_llm(llm_provider)

        # Crear chain
        chain = (
            {
                "context": retriever | format_docs,
                "categoria": RunnablePassthrough(),
                "productoname": RunnablePassthrough(),
                "desc_comprador": RunnablePassthrough(),
                "desc_proveedor": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invocar chain
        resultado = chain.invoke({
            "categoria": payload['Categoria'],
            "productoname": payload['productoname'],
            "desc_comprador": payload['DescripcionProductoComprador'],
            "desc_proveedor": payload['DescripcionProductoProveedor']
        })

        # Parsear JSON
        import json
        try:
            resultado_json = json.loads(resultado)
        except:
            # Si no es JSON válido, intentar extraerlo
            import re
            match = re.search(r'\{.*\}', resultado, re.DOTALL)
            if match:
                resultado_json = json.loads(match.group())
            else:
                resultado_json = {"raw_response": resultado}

        # Si usar diccionarios, aplicar normalización
        if use_diccionarios:
            logging.info("Aplicando normalización con diccionarios...")
            # TODO: Implementar normalización con diccionarios
            # Por ahora retornamos el resultado sin normalizar
            pass

        logging.info("Catalogación completada")

        return resultado_json

    except Exception as e:
        logging.exception(f"Error en catalogación: {e}")
        return {"error": str(e)}


def catalogar_sin_adjuntos(payload: Dict[str, Any], llm_provider: str = None,
                           campos_manuales: List[str] = None) -> Dict[str, Any]:
    """
    Cataloga usando solo las descripciones (fallback sin adjuntos).

    Args:
        payload: Info del producto
        llm_provider: Proveedor de LLM
        campos_manuales: Lista de campos a extraer

    Returns:
        dict: Resultado de la catalogación
    """
    try:
        campos_json = _build_campos_json(campos_manuales)

        template = """Eres un experto en catalogación de productos.

Extrae atributos técnicos del siguiente producto:

PRODUCTO:
- Categoría: {categoria}
- Nombre: {productoname}
- Descripción Comprador: {desc_comprador}
- Descripción Proveedor: {desc_proveedor}

Responde ÚNICAMENTE con un JSON válido con los atributos encontrados:
""" + campos_json + "\n"

        prompt = PromptTemplate.from_template(template)
        llm = get_llm(llm_provider)

        chain = prompt | llm | StrOutputParser()

        resultado = chain.invoke({
            "categoria": payload['Categoria'],
            "productoname": payload['productoname'],
            "desc_comprador": payload['DescripcionProductoComprador'],
            "desc_proveedor": payload['DescripcionProductoProveedor']
        })

        import json
        import re
        try:
            resultado_json = json.loads(resultado)
        except:
            match = re.search(r'\{.*\}', resultado, re.DOTALL)
            if match:
                resultado_json = json.loads(match.group())
            else:
                resultado_json = {"raw_response": resultado}

        return resultado_json

    except Exception as e:
        logging.exception(f"Error en catalogación sin adjuntos: {e}")
        return {"error": str(e)}
