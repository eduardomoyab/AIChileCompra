import os
import sys
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.get_vectorstore import VectorStoreManager

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_retriever_adjuntos(
    codigo_cotizacion: str,
    rut_proveedor: str,
    k: int = 5,
    search_type: str = "similarity",
    collection_name: str = "adjuntos_procesados",
    persist_directory: str = "cache/Adjuntos"
):
    """
    Crea un retriever para buscar en los adjuntos procesados de una cotización específica.

    Este retriever filtra automáticamente por metadata para obtener solo los documentos
    relevantes a la cotización y proveedor especificados.

    Args:
        codigo_cotizacion (str): Código de la solicitud de cotización
        rut_proveedor (str): RUT del proveedor
        k (int): Número de documentos a retornar
        search_type (str): Tipo de búsqueda ('similarity' o 'mmr')
        collection_name (str): Nombre de la colección en ChromaDB
        persist_directory (str): Directorio donde está el vector store

    Returns:
        VectorStoreRetriever: Retriever configurado con los filtros de metadata

    Example:
        >>> retriever = create_retriever_adjuntos("12345678", "76123456-7", k=5)
        >>> results = retriever.get_relevant_documents("procesador intel")
        >>> for doc in results:
        >>>     print(doc.page_content)
    """
    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # Crear filtro de metadata con operador $and para ChromaDB
    metadata_filter = {
        "$and": [
            {"codigo_cotizacion": {"$eq": codigo_cotizacion}},
            {"rut_proveedor": {"$eq": rut_proveedor}}
        ]
    }

    # Obtener retriever con filtro aplicado
    retriever = manager.get_retriever(
        search_type=search_type,
        k=k,
        filter=metadata_filter
    )

    logging.info(f"Retriever creado para cotización {codigo_cotizacion} y proveedor {rut_proveedor}")
    return retriever


def search_adjuntos(
    query: str,
    codigo_cotizacion: str,
    rut_proveedor: str,
    k: int = 5,
    collection_name: str = "adjuntos_procesados",
    persist_directory: str = "cache/Adjuntos"
):
    """
    Busca directamente en los adjuntos procesados sin crear un retriever.

    Args:
        query (str): Consulta de búsqueda
        codigo_cotizacion (str): Código de la solicitud de cotización
        rut_proveedor (str): RUT del proveedor
        k (int): Número de documentos a retornar
        collection_name (str): Nombre de la colección
        persist_directory (str): Directorio del vector store

    Returns:
        List[Document]: Lista de documentos relevantes

    Example:
        >>> results = search_adjuntos("procesador intel", "12345678", "76123456-7", k=5)
        >>> for doc in results:
        >>>     print(f"Contenido: {doc.page_content}")
        >>>     print(f"Metadata: {doc.metadata}")
    """
    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # Crear filtro de metadata con operador $and para ChromaDB
    metadata_filter = {
        "$and": [
            {"codigo_cotizacion": {"$eq": codigo_cotizacion}},
            {"rut_proveedor": {"$eq": rut_proveedor}}
        ]
    }

    # Buscar con filtro
    results = manager.similarity_search(query, k=k, filter=metadata_filter)

    logging.info(f"Búsqueda en adjuntos: {len(results)} documentos encontrados")
    return results


def search_adjuntos_with_score(
    query: str,
    codigo_cotizacion: str,
    rut_proveedor: str,
    k: int = 5,
    collection_name: str = "adjuntos_procesados",
    persist_directory: str = "cache/Adjuntos"
):
    """
    Busca en adjuntos procesados y retorna documentos con scores de similitud.

    Args:
        query (str): Consulta de búsqueda
        codigo_cotizacion (str): Código de la solicitud de cotización
        rut_proveedor (str): RUT del proveedor
        k (int): Número de documentos a retornar
        collection_name (str): Nombre de la colección
        persist_directory (str): Directorio del vector store

    Returns:
        List[tuple]: Lista de tuplas (Document, score)

    Example:
        >>> results = search_adjuntos_with_score("RAM DDR4", "12345678", "76123456-7", k=3)
        >>> for doc, score in results:
        >>>     print(f"Score: {score}")
        >>>     print(f"Contenido: {doc.page_content}")
    """
    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # Crear filtro de metadata con operador $and para ChromaDB
    metadata_filter = {
        "$and": [
            {"codigo_cotizacion": {"$eq": codigo_cotizacion}},
            {"rut_proveedor": {"$eq": rut_proveedor}}
        ]
    }

    # Buscar con scores
    results = manager.similarity_search_with_score(query, k=k, filter=metadata_filter)

    logging.info(f"Búsqueda con scores: {len(results)} documentos encontrados")
    return results


# Script de prueba
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python retriever.py <codigo_cotizacion> <rut_proveedor> <query> [k]")
        print("\nExample:")
        print("  python retriever.py 12345678 76123456-7 'procesador intel' 5")
        sys.exit(1)

    codigo_cotizacion = sys.argv[1]
    rut_proveedor = sys.argv[2]
    query = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"\n=== Búsqueda en adjuntos procesados ===")
    print(f"Código cotización: {codigo_cotizacion}")
    print(f"RUT proveedor: {rut_proveedor}")
    print(f"Query: {query}")
    print(f"K: {k}\n")

    try:
        results = search_adjuntos_with_score(query, codigo_cotizacion, rut_proveedor, k=k)

        print(f"=== Resultados ({len(results)}) ===\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"Resultado {i} (Score: {score:.4f}):")
            print(f"  Archivo: {doc.metadata.get('source_file', 'N/A')}")
            print(f"  Código: {doc.metadata.get('codigo_cotizacion', 'N/A')}")
            print(f"  Proveedor: {doc.metadata.get('rut_proveedor', 'N/A')}")
            print(f"  Contenido: {doc.page_content[:200]}...")
            print()

    except Exception as e:
        logging.error(f"Error en la búsqueda: {e}")
        sys.exit(1)
