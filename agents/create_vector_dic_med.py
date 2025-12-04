import os
import sys
import logging
from typing import List, Dict
from docx import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# Rutas
DICCIONARIOS_PATH = "diccionarios/Medicamentos"
CACHE_PATH = "cache/Medicamentos"

# Configuración de chunks desde .env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Mapeo de archivos a sus orígenes
FILE_MAPPING = {
    "Diccionario_FF.docx": "Forma Farmaceutica",
    "Diccionario_PA.docx": "Principio Activo"
}


def extract_text_from_docx(docx_path: str) -> str:
    """
    Extrae todo el texto de un archivo DOCX.

    Args:
        docx_path (str): Ruta al archivo DOCX

    Returns:
        str: Texto completo del documento
    """
    try:
        doc = Document(docx_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        logging.info(f"Extracted text from DOCX: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX {docx_path}: {e}")
        raise


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                          chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Divide el texto en chunks usando RecursiveCharacterTextSplitter.

    Args:
        text (str): Texto a dividir
        chunk_size (int): Tamaño de cada chunk
        chunk_overlap (int): Solapamiento entre chunks

    Returns:
        List[str]: Lista de chunks de texto
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        logging.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        raise


def create_medicamentos_vectorstore(
    diccionarios_path: str = DICCIONARIOS_PATH,
    cache_path: str = CACHE_PATH,
    collection_name: str = "diccionario_medicamentos",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> VectorStoreManager:
    """
    Crea un vector store con los diccionarios de medicamentos.

    Args:
        diccionarios_path (str): Ruta a la carpeta de diccionarios
        cache_path (str): Ruta donde guardar el vector store
        collection_name (str): Nombre de la colección
        chunk_size (int): Tamaño de cada chunk
        chunk_overlap (int): Solapamiento entre chunks

    Returns:
        VectorStoreManager: Gestor del vector store creado

    Example:
        >>> manager = create_medicamentos_vectorstore()
        >>> results = manager.similarity_search("paracetamol tabletas")
    """
    # Crear directorio de cache si no existe
    os.makedirs(cache_path, exist_ok=True)

    logging.info("=== Iniciando creación de vector store de medicamentos ===")
    logging.info(f"Configuración de chunks: size={chunk_size}, overlap={chunk_overlap}")

    # Inicializar vector store manager
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=cache_path
    )

    # Lista para almacenar todos los textos y metadatas
    all_texts = []
    all_metadatas = []

    # Procesar cada archivo DOCX
    for filename, origen in FILE_MAPPING.items():
        docx_path = os.path.join(diccionarios_path, filename)

        # Verificar que el archivo exista
        if not os.path.exists(docx_path):
            logging.warning(f"Archivo no encontrado: {docx_path}")
            continue

        logging.info(f"Procesando archivo: {filename} (origen: {origen})")

        # Extraer texto
        docx_text = extract_text_from_docx(docx_path)

        # Dividir en chunks
        chunks = split_text_into_chunks(docx_text, chunk_size, chunk_overlap)

        # Añadir cada chunk con su metadata
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "origen": origen,
                "tipo": "diccionario",
                "archivo": filename,
                "formato": "docx",
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        logging.info(f"  - {len(chunks)} chunks creados de {filename}")

    # Añadir todos los documentos al vector store en lotes
    logging.info(f"Añadiendo {len(all_texts)} documentos al vector store...")

    # ChromaDB tiene un límite de batch size, procesamos en lotes
    batch_size = 5000
    doc_ids = []

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]

        logging.info(f"Procesando lote {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}...")
        batch_ids = manager.add_documents(batch_texts, batch_metadatas)
        doc_ids.extend(batch_ids)

    logging.info(f"✓ Vector store creado exitosamente")
    logging.info(f"  - Total documentos: {len(doc_ids)}")

    # Contar documentos por origen
    ff_count = sum(1 for m in all_metadatas if m['origen'] == 'Forma Farmaceutica')
    pa_count = sum(1 for m in all_metadatas if m['origen'] == 'Principio Activo')

    logging.info(f"  - Documentos Forma Farmaceutica: {ff_count}")
    logging.info(f"  - Documentos Principio Activo: {pa_count}")
    logging.info(f"  - Ubicación: {cache_path}")
    logging.info(f"  - Colección: {collection_name}")

    return manager


def load_medicamentos_vectorstore(
    cache_path: str = CACHE_PATH,
    collection_name: str = "diccionario_medicamentos"
) -> VectorStoreManager:
    """
    Carga el vector store de diccionarios de medicamentos.

    Args:
        cache_path (str): Ruta donde está guardado el vector store
        collection_name (str): Nombre de la colección

    Returns:
        VectorStoreManager: Gestor del vector store cargado

    Example:
        >>> manager = load_medicamentos_vectorstore()
        >>> results = manager.similarity_search("ibuprofeno suspensión")
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No se encontró el vector store en {cache_path}. "
            f"Ejecuta create_medicamentos_vectorstore() primero."
        )

    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=cache_path
    )

    logging.info(f"✓ Vector store cargado desde {cache_path}")
    logging.info(f"  - Total documentos: {manager.get_collection_count()}")

    return manager


def search_diccionario(query: str,
                       k: int = 5,
                       filter_origen: str = None,
                       cache_path: str = CACHE_PATH,
                       collection_name: str = "diccionario_medicamentos"):
    """
    Función helper para buscar en el diccionario de medicamentos.

    Args:
        query (str): Consulta de búsqueda
        k (int): Número de resultados
        filter_origen (str, optional): Filtrar por origen ('Forma Farmaceutica' o 'Principio Activo')
        cache_path (str): Ruta al cache
        collection_name (str): Nombre de la colección

    Returns:
        List[Document]: Resultados de la búsqueda

    Example:
        >>> results = search_diccionario("paracetamol", k=3, filter_origen="Principio Activo")
        >>> for doc in results:
        >>>     print(doc.page_content)
    """
    manager = load_medicamentos_vectorstore(cache_path, collection_name)

    filter_dict = None
    if filter_origen:
        filter_dict = {"origen": filter_origen}

    results = manager.similarity_search(query, k=k, filter=filter_dict)
    return results


# Script principal
if __name__ == "__main__":
    import sys

    # Determinar acción
    action = sys.argv[1] if len(sys.argv) > 1 else "create"

    if action == "create":
        print("\n=== Creando vector store de diccionarios de medicamentos ===\n")

        try:
            # Obtener configuración de chunks desde .env o argumentos
            chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNK_SIZE
            chunk_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else CHUNK_OVERLAP

            print(f"Configuración:")
            print(f"  - Chunk size: {chunk_size}")
            print(f"  - Chunk overlap: {chunk_overlap}\n")

            manager = create_medicamentos_vectorstore(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            print("\n=== Vector store creado exitosamente ===")
            print(f"Total documentos en colección: {manager.get_collection_count()}")

            # Mostrar ejemplo de búsqueda
            print("\n=== Ejemplo de búsqueda ===")
            print("Query: 'paracetamol tabletas'")
            results = manager.similarity_search("paracetamol tabletas", k=3)

            for i, doc in enumerate(results, 1):
                print(f"\nResultado {i}:")
                print(f"  Origen: {doc.metadata.get('origen', 'N/A')}")
                print(f"  Archivo: {doc.metadata.get('archivo', 'N/A')}")
                print(f"  Chunk: {doc.metadata.get('chunk_index', 'N/A')}/{doc.metadata.get('total_chunks', 'N/A')}")
                print(f"  Contenido: {doc.page_content[:150]}...")

        except Exception as e:
            logging.error(f"Error creando vector store: {e}")
            sys.exit(1)

    elif action == "search":
        if len(sys.argv) < 3:
            print("Usage: python create_vector_dic_med.py search <query> [k] [origen]")
            print("\nExamples:")
            print("  python create_vector_dic_med.py search 'paracetamol' 5")
            print("  python create_vector_dic_med.py search 'ibuprofeno' 3 'Principio Activo'")
            sys.exit(1)

        query = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        filter_origen = sys.argv[4] if len(sys.argv) > 4 else None

        print(f"\n=== Búsqueda en diccionario ===")
        print(f"Query: {query}")
        print(f"K: {k}")
        if filter_origen:
            print(f"Filtro origen: {filter_origen}")

        try:
            results = search_diccionario(query, k=k, filter_origen=filter_origen)

            print(f"\n=== Resultados ({len(results)}) ===")
            for i, doc in enumerate(results, 1):
                print(f"\nResultado {i}:")
                print(f"  Origen: {doc.metadata.get('origen', 'N/A')}")
                print(f"  Archivo: {doc.metadata.get('archivo', 'N/A')}")
                print(f"  Chunk: {doc.metadata.get('chunk_index', 'N/A')}/{doc.metadata.get('total_chunks', 'N/A')}")
                print(f"  Contenido: {doc.page_content}")

        except Exception as e:
            logging.error(f"Error buscando: {e}")
            sys.exit(1)

    elif action == "info":
        try:
            manager = load_medicamentos_vectorstore()
            print(f"\n=== Información del vector store ===")
            print(f"Total documentos: {manager.get_collection_count()}")
            print(f"Ubicación: {CACHE_PATH}")
            print(f"Colección: diccionario_medicamentos")

            # Contar por origen
            results_ff = manager.similarity_search(".", k=10000, filter={"origen": "Forma Farmaceutica"})
            results_pa = manager.similarity_search(".", k=10000, filter={"origen": "Principio Activo"})

            print(f"\nDocumentos por origen:")
            print(f"  - Forma Farmaceutica: {len(results_ff)}")
            print(f"  - Principio Activo: {len(results_pa)}")

        except Exception as e:
            logging.error(f"Error obteniendo información: {e}")
            sys.exit(1)

    else:
        print("Acción no reconocida. Usa: create, search, o info")
        print("\nExamples:")
        print("  python create_vector_dic_med.py create [chunk_size] [chunk_overlap]")
        print("  python create_vector_dic_med.py search <query> [k] [origen]")
        print("  python create_vector_dic_med.py info")
        sys.exit(1)
