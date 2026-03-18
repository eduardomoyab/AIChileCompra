import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Cargar variables de entorno PRIMERO
load_dotenv()

# Fix para conflicto de OpenMP con FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# CHROMA DESHABILITADO - Solo usamos FAISS ahora
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_faiss_from_files(
    file_paths: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> FAISS:
    """
    Crea un vector store FAISS en memoria desde archivos de texto.
    NO persiste nada en disco - solo para uso temporal.

    Args:
        file_paths (List[str]): Rutas a los archivos de texto
        metadatas (List[Dict], optional): Metadata para cada archivo
        embedding_model (str): Modelo de embeddings de OpenAI
        chunk_size (int, optional): Tamaño de los chunks. Si None, no hace chunking
        chunk_overlap (int, optional): Overlap entre chunks. Si None, usa chunk_size/10

    Returns:
        FAISS: Vector store en memoria listo para usar

    Example:
        >>> file_paths = ["processed/file1.txt", "processed/file2.txt"]
        >>> metadatas = [
        >>>     {"codigo_cotizacion": "12345", "source_file": "file1.txt"},
        >>>     {"codigo_cotizacion": "12346", "source_file": "file2.txt"}
        >>> ]
        >>> # Sin chunking
        >>> vectorstore = create_faiss_from_files(file_paths, metadatas)
        >>> # Con chunking pequeño (para campos manuales)
        >>> vectorstore = create_faiss_from_files(file_paths, metadatas, chunk_size=200, chunk_overlap=50)
    """
    # Validar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")

    # Inicializar embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key
    )

    # Leer archivos y crear documentos
    documents = []
    for i, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Combinar metadata proporcionada con metadata del archivo
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                metadata['source_file'] = os.path.basename(file_path)
                metadata['file_path'] = file_path

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    if not documents:
        raise ValueError("No se pudo leer ningún archivo")

    # Aplicar chunking si se especificó
    if chunk_size is not None:
        overlap = chunk_overlap if chunk_overlap is not None else max(1, chunk_size // 10)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        documents = text_splitter.split_documents(documents)
        print(f"Documentos divididos en {len(documents)} chunks de ~{chunk_size} caracteres")

    # Crear FAISS en memoria
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def create_faiss_from_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embedding_model: str = "text-embedding-3-small"
) -> FAISS:
    """
    Crea un vector store FAISS en memoria desde textos.
    NO persiste nada en disco - solo para uso temporal.

    Args:
        texts (List[str]): Lista de textos
        metadatas (List[Dict], optional): Metadata para cada texto
        embedding_model (str): Modelo de embeddings de OpenAI

    Returns:
        FAISS: Vector store en memoria listo para usar
    """
    # Validar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")

    # Inicializar embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key
    )

    # Crear documentos
    documents = []
    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    # Crear FAISS en memoria
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def search_in_faiss(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Busca en un vector store FAISS aplicando filtros de metadata.

    Args:
        vectorstore (FAISS): Vector store donde buscar
        query (str): Query de búsqueda
        k (int): Número de resultados
        filter (Dict, optional): Filtros de metadata

    Returns:
        List[Document]: Documentos encontrados
    """
    if not filter:
        return vectorstore.similarity_search(query, k=k)

    # FAISS no soporta filtros nativos, aplicamos manualmente
    # Obtenemos más resultados y luego filtramos
    all_results = vectorstore.similarity_search(query, k=k*10)

    filtered_results = []
    for doc in all_results:
        if _matches_filter(doc.metadata, filter):
            filtered_results.append(doc)
            if len(filtered_results) >= k:
                break

    return filtered_results


def _matches_filter(metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
    """
    Verifica si un documento coincide con los filtros de metadata.
    Soporta filtros con operadores $and, $eq similares a ChromaDB.
    """
    if not filter:
        return True

    # Soporte para formato ChromaDB: {"$and": [{"field": {"$eq": "value"}}]}
    if "$and" in filter:
        conditions = filter["$and"]
        for condition in conditions:
            for field, value_filter in condition.items():
                if isinstance(value_filter, dict) and "$eq" in value_filter:
                    if metadata.get(field) != value_filter["$eq"]:
                        return False
                elif metadata.get(field) != value_filter:
                    return False
        return True

    # Formato simple: {"field": "value"}
    for field, value in filter.items():
        if metadata.get(field) != value:
            return False

    return True


# ========== CHROMA DESHABILITADO ==========
# MIGRADO A FAISS - Chroma causaba overhead de telemetry y era lento
# El código ahora usa FAISS para todo (adjuntos Y diccionarios)
#
# class VectorStoreManager:
#     """
#     DEPRECATED: Esta clase usaba Chroma para diccionarios pre-cargados.
#     Ahora se usa FAISS para todo (ver retriever_diccionario_comp.py).
#     """
#     pass
