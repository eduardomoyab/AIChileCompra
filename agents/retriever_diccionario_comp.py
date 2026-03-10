import os
import sys
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
import pandas as pd
from docx import Document as DocxDocument

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.get_vectorstore import create_faiss_from_texts

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
DICCIONARIOS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diccionarios", "Computadores")
FEATURES_PATH = os.path.join(DICCIONARIOS_PATH, "diccionario_features.docx")
PROCESADOR_PATH = os.path.join(DICCIONARIOS_PATH, "diccionario_procesador.xlsx")

# Cache global para vector stores (se cargan una vez por sesión)
_vectorstore_cache = {}


def _load_features_dict() -> tuple[List[str], List[Dict]]:
    """
    Carga el diccionario de features desde .docx agrupando por sección.

    Cada sección (Tipo RAM, Sistema Operativo, Tipo Almacenamiento, Marca, etc.)
    se carga como un único documento con todos sus valores permitidos.
    Esto permite que el LLM vea TODOS los valores canónicos de cada campo
    en un solo retrieval, evitando normalizaciones incorrectas.
    """
    if not os.path.exists(FEATURES_PATH):
        logging.warning(f"No se encontró {FEATURES_PATH}")
        return [], []

    doc = DocxDocument(FEATURES_PATH)
    texts = []
    metadatas = []

    current_section = None
    current_lines: List[str] = []

    def _flush():
        if current_section and current_lines:
            section_text = f"{current_section}:\n" + "\n".join(current_lines)
            texts.append(section_text)
            metadatas.append({
                'origen': 'Features',
                'archivo': 'diccionario_features.docx',
                'seccion': current_section
            })

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line:
            continue
        if line.startswith('-'):
            # Valor dentro de la sección actual
            if current_section is not None:
                current_lines.append(line)
        else:
            # Nuevo encabezado de sección
            _flush()
            current_section = line
            current_lines = []

    _flush()  # Guardar la última sección

    logging.info(f"Cargado diccionario Features: {len(texts)} secciones — "
                 f"{[m['seccion'] for m in metadatas]}")
    return texts, metadatas


def _load_procesador_dict() -> tuple[List[str], List[Dict]]:
    """Carga el diccionario de procesadores desde .xlsx"""
    if not os.path.exists(PROCESADOR_PATH):
        logging.warning(f"No se encontró {PROCESADOR_PATH}")
        return [], []

    df = pd.read_excel(PROCESADOR_PATH)
    texts = []
    metadatas = []

    # Convertir cada fila a texto
    for idx, row in df.iterrows():
        procesador = row.get('Procesador', '')
        nucleos = row.get('Nucleos', '')
        hilos = row.get('Hilos', '')

        if procesador:
            text = f"Procesador: {procesador}, Núcleos: {nucleos}, Hilos: {hilos}"
            texts.append(text)
            metadatas.append({
                'origen': 'Procesador',
                'archivo': 'diccionario_procesador.xlsx',
                'row_number': int(idx),
                'procesador': procesador,
                'nucleos': int(nucleos) if pd.notna(nucleos) else None,
                'hilos': int(hilos) if pd.notna(hilos) else None
            })

    logging.info(f"Cargado diccionario Procesador: {len(texts)} procesadores")
    return texts, metadatas


def _get_or_create_vectorstore(origen: str = 'all'):
    """
    Obtiene o crea el vector store FAISS para el diccionario especificado.
    Usa cache para evitar recargar en cada consulta.

    Args:
        origen: 'Features', 'Procesador', o 'all' para ambos
    """
    if origen in _vectorstore_cache:
        return _vectorstore_cache[origen]

    texts_all = []
    metadatas_all = []

    # Cargar Features si se necesita
    if origen in ['Features', 'all']:
        texts_feat, meta_feat = _load_features_dict()
        texts_all.extend(texts_feat)
        metadatas_all.extend(meta_feat)

    # Cargar Procesador si se necesita
    if origen in ['Procesador', 'all']:
        texts_proc, meta_proc = _load_procesador_dict()
        texts_all.extend(texts_proc)
        metadatas_all.extend(meta_proc)

    if not texts_all:
        raise ValueError(f"No se pudieron cargar diccionarios para origen: {origen}")

    # Crear FAISS vector store en memoria
    vectorstore = create_faiss_from_texts(texts_all, metadatas_all)

    # Guardar en cache
    _vectorstore_cache[origen] = vectorstore
    logging.info(f"Vector store FAISS creado y cacheado para origen: {origen} ({len(texts_all)} documentos)")

    return vectorstore


def search_diccionario_balanced(
    query: str,
    k_total: int = 2,
    **kwargs  # Mantener compatibilidad con llamadas antiguas
) -> List[Document]:
    """
    Busca en el diccionario de computadores asegurando que se obtenga exactamente
    1 resultado de cada origen (Features y Procesador).

    Esta función garantiza diversidad en los resultados al forzar que haya
    representación de ambos orígenes de manera obligatoria.

    MIGRADO A FAISS - ya no usa Chroma.

    Args:
        query (str): Consulta de búsqueda
        k_total (int): Número total de resultados (debe ser 2 para k=1 de cada origen)

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
    # Features: recuperar hasta 4 secciones (Tipo RAM, SO, Tipo Almacenamiento, Marca, etc.)
    # Procesador: 3 docs para que el LLM tenga más candidatos y elija el más exacto
    k_features = 4
    k_procesador = 3

    # Cargar vector stores FAISS (uno para cada origen)
    vs_features = _get_or_create_vectorstore('Features')
    vs_procesador = _get_or_create_vectorstore('Procesador')

    # Buscar en cada origen por separado usando FAISS
    from agents.get_vectorstore import search_in_faiss

    results_features = search_in_faiss(
        vs_features,
        query,
        k=k_features,
        filter={"origen": "Features"}
    )

    results_procesador = search_in_faiss(
        vs_procesador,
        query,
        k=k_procesador,
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
        f"Búsqueda balanceada FAISS: {len(results_features)} Features + "
        f"{len(results_procesador)} Procesador = {len(combined_results)} total"
    )

    return combined_results


def search_diccionario_with_filter(
    query: str,
    origen: str,
    k: int = 2,
    **kwargs  # Mantener compatibilidad
) -> List[Document]:
    """
    Busca en el diccionario filtrando por un origen específico.

    MIGRADO A FAISS - ya no usa Chroma.

    Args:
        query (str): Consulta de búsqueda
        origen (str): Origen a filtrar ('Features' o 'Procesador')
        k (int): Número de resultados

    Returns:
        List[Document]: Documentos filtrados por origen

    Example:
        >>> results = search_diccionario_with_filter("RAM DDR4", origen="Features", k=3)
        >>> assert all(doc.metadata['origen'] == 'Features' for doc in results)
    """
    if origen not in ['Features', 'Procesador']:
        raise ValueError("origen debe ser 'Features' o 'Procesador'")

    # Cargar vector store FAISS para ese origen específico
    vectorstore = _get_or_create_vectorstore(origen)

    # Buscar con filtro usando FAISS
    from agents.get_vectorstore import search_in_faiss

    results = search_in_faiss(
        vectorstore,
        query,
        k=k,
        filter={"origen": origen}
    )

    logging.info(f"Búsqueda FAISS en '{origen}': {len(results)} documentos encontrados")
    return results


def create_retriever_diccionario(
    origen: Optional[str] = None,
    k: int = 2,
    **kwargs  # Mantener compatibilidad
):
    """
    DEPRECATED: Esta función ya no es necesaria con FAISS.
    Usa search_diccionario_balanced() o search_diccionario_with_filter() directamente.

    Se mantiene solo para compatibilidad con código antiguo.
    """
    logging.warning(
        "create_retriever_diccionario() está deprecado. "
        "Usa search_diccionario_balanced() o search_diccionario_with_filter() directamente."
    )

    # Retornar None - el código que lo use debe actualizarse
    return None


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
