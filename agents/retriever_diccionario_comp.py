import os
import sys
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document

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

# Constantes
CACHE_PATH = "cache/Computadores"
COLLECTION_NAME = "diccionario_computadores"


def search_diccionario_balanced(
    query: str,
    k_total: int = 2,
    collection_name: str = COLLECTION_NAME,
    cache_path: str = CACHE_PATH
) -> List[Document]:
    """
    Busca en el diccionario de computadores asegurando que se obtenga exactamente
    1 resultado de cada origen (Features y Procesador).

    Esta función garantiza diversidad en los resultados al forzar que haya
    representación de ambos orígenes de manera obligatoria.

    Args:
        query (str): Consulta de búsqueda
        k_total (int): Número total de resultados (debe ser 2 para k=1 de cada origen)
        collection_name (str): Nombre de la colección
        cache_path (str): Ruta al cache

    Returns:
        List[Document]: Lista con exactamente k_total documentos,
                       balanceados entre orígenes (1 de cada uno si k_total=2)

    Example:
        >>> # Obtener 1 de Features y 1 de Procesador (total = 2)
        >>> results = search_diccionario_balanced("intel core i7", k_total=2)
        >>> assert len(results) == 2
        >>> origins = [doc.metadata['origen'] for doc in results]
        >>> assert 'Features' in origins and 'Procesador' in origins
    """
    if k_total % 2 != 0:
        logging.warning(f"k_total debe ser par para balancear orígenes. Ajustando de {k_total} a {k_total + 1}")
        k_total = k_total + 1

    k_per_origin = k_total // 2

    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=cache_path
    )

    # Buscar en cada origen por separado
    results_features = manager.similarity_search(
        query,
        k=k_per_origin,
        filter={"origen": "Features"}
    )

    results_procesador = manager.similarity_search(
        query,
        k=k_per_origin,
        filter={"origen": "Procesador"}
    )

    # Validar que se obtuvieron resultados de ambos orígenes
    if len(results_features) == 0:
        logging.warning("No se encontraron resultados en 'Features'")
    if len(results_procesador) == 0:
        logging.warning("No se encontraron resultados en 'Procesador'")

    # Combinar resultados intercalados (Features, Procesador, Features, Procesador, ...)
    combined_results = []
    for i in range(max(len(results_features), len(results_procesador))):
        if i < len(results_features):
            combined_results.append(results_features[i])
        if i < len(results_procesador):
            combined_results.append(results_procesador[i])

    logging.info(
        f"Búsqueda balanceada: {len(results_features)} Features + "
        f"{len(results_procesador)} Procesador = {len(combined_results)} total"
    )

    return combined_results


def search_diccionario_with_filter(
    query: str,
    origen: str,
    k: int = 2,
    collection_name: str = COLLECTION_NAME,
    cache_path: str = CACHE_PATH
) -> List[Document]:
    """
    Busca en el diccionario filtrando por un origen específico.

    Args:
        query (str): Consulta de búsqueda
        origen (str): Origen a filtrar ('Features' o 'Procesador')
        k (int): Número de resultados
        collection_name (str): Nombre de la colección
        cache_path (str): Ruta al cache

    Returns:
        List[Document]: Documentos filtrados por origen

    Example:
        >>> results = search_diccionario_with_filter("RAM DDR4", origen="Features", k=3)
        >>> assert all(doc.metadata['origen'] == 'Features' for doc in results)
    """
    if origen not in ['Features', 'Procesador']:
        raise ValueError("origen debe ser 'Features' o 'Procesador'")

    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=cache_path
    )

    # Buscar con filtro
    results = manager.similarity_search(
        query,
        k=k,
        filter={"origen": origen}
    )

    logging.info(f"Búsqueda en '{origen}': {len(results)} documentos encontrados")
    return results


def create_retriever_diccionario(
    origen: Optional[str] = None,
    k: int = 2,
    search_type: str = "similarity",
    collection_name: str = COLLECTION_NAME,
    cache_path: str = CACHE_PATH
):
    """
    Crea un retriever para el diccionario de computadores.

    Args:
        origen (str, optional): Filtrar por origen ('Features' o 'Procesador').
                               Si es None, busca en ambos.
        k (int): Número de documentos a retornar
        search_type (str): Tipo de búsqueda ('similarity' o 'mmr')
        collection_name (str): Nombre de la colección
        cache_path (str): Ruta al cache

    Returns:
        VectorStoreRetriever: Retriever configurado

    Example:
        >>> # Retriever solo para Features
        >>> retriever = create_retriever_diccionario(origen="Features", k=3)
        >>> results = retriever.get_relevant_documents("pantalla LED")
    """
    # Cargar vector store
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=cache_path
    )

    # Crear filtro si se especifica origen
    filter_dict = {"origen": origen} if origen else None

    # Obtener retriever
    retriever = manager.get_retriever(
        search_type=search_type,
        k=k,
        filter=filter_dict
    )

    logging.info(f"Retriever de diccionario creado (origen: {origen or 'todos'})")
    return retriever


def format_docs_for_llm(docs: List[Document]) -> str:
    """
    Formatea los documentos del diccionario para usarlos como contexto en el LLM.

    Args:
        docs (List[Document]): Lista de documentos del diccionario

    Returns:
        str: Texto formateado con información de cada documento

    Example:
        >>> docs = search_diccionario_balanced("intel i7", k_total=2)
        >>> context = format_docs_for_llm(docs)
        >>> # Usar context en el prompt del LLM
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        origen = doc.metadata.get('origen', 'N/A')
        archivo = doc.metadata.get('archivo', 'N/A')

        formatted_parts.append(
            f"[Documento {i} - Origen: {origen} - Archivo: {archivo}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_parts)


# Script de prueba
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retriever_diccionario_comp.py <query> [k_total]")
        print("\nExamples:")
        print("  # Búsqueda balanceada (1 de cada origen)")
        print("  python retriever_diccionario_comp.py 'intel core i7' 2")
        print()
        print("  # Búsqueda balanceada (2 de cada origen)")
        print("  python retriever_diccionario_comp.py 'RAM DDR4' 4")
        sys.exit(1)

    query = sys.argv[1]
    k_total = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"\n=== Búsqueda balanceada en diccionario de computadores ===")
    print(f"Query: {query}")
    print(f"K total: {k_total} ({k_total//2} de cada origen)\n")

    try:
        results = search_diccionario_balanced(query, k_total=k_total)

        print(f"=== Resultados ({len(results)}) ===\n")

        # Contar por origen
        origins_count = {}
        for doc in results:
            origen = doc.metadata.get('origen', 'N/A')
            origins_count[origen] = origins_count.get(origen, 0) + 1

        print(f"Distribución por origen:")
        for origen, count in origins_count.items():
            print(f"  - {origen}: {count}")
        print()

        # Mostrar resultados
        for i, doc in enumerate(results, 1):
            origen = doc.metadata.get('origen', 'N/A')
            archivo = doc.metadata.get('archivo', 'N/A')

            print(f"Resultado {i}:")
            print(f"  Origen: {origen}")
            print(f"  Archivo: {archivo}")

            if origen == "Procesador":
                print(f"  Columna: {doc.metadata.get('column', 'N/A')}")
                print(f"  Fila: {doc.metadata.get('row_number', 'N/A')}")

            print(f"  Contenido: {doc.page_content[:150]}...")
            print()

        # Mostrar contexto formateado
        print("\n=== Contexto formateado para LLM ===")
        context = format_docs_for_llm(results[:2])  # Solo primeros 2 para ejemplo
        print(context)

    except Exception as e:
        logging.error(f"Error en la búsqueda: {e}")
        sys.exit(1)
