import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Cargar variables de entorno
load_dotenv()


class VectorStoreManager:
    """Gestor de vector store con embeddings de Google"""

    def __init__(self,
                 collection_name: str = "default_collection",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "models/text-embedding-004"):
        """
        Inicializa el gestor de vector store.

        Args:
            collection_name (str): Nombre de la colección en el vector store
            persist_directory (str): Directorio donde persistir el vector store
            embedding_model (str): Modelo de embeddings de Google a usar
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        # Validar API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env")

        # Inicializar modelo de embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key
        )

        # Inicializar o cargar vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """Carga el vector store existente o crea uno nuevo"""
        persist_path = os.path.join(self.persist_directory, self.collection_name)

        if os.path.exists(persist_path):
            # Cargar vector store existente
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Crear nuevo vector store
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(self,
                     texts: List[str],
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Añade documentos al vector store.

        Args:
            texts (List[str]): Lista de textos a añadir
            metadatas (List[Dict], optional): Metadata para cada texto
            ids (List[str], optional): IDs personalizados para cada documento

        Returns:
            List[str]: IDs de los documentos añadidos

        Example:
            >>> manager = VectorStoreManager("my_collection")
            >>> texts = ["Texto 1", "Texto 2"]
            >>> metadatas = [
            >>>     {"codigo_cotizacion": "12345", "rut_proveedor": "76123456-7"},
            >>>     {"codigo_cotizacion": "12346", "rut_proveedor": "76123456-8"}
            >>> ]
            >>> ids = manager.add_documents(texts, metadatas)
        """
        # Crear documentos de LangChain
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        # Añadir al vector store
        doc_ids = self.vectorstore.add_documents(documents, ids=ids)

        return doc_ids

    def add_documents_from_files(self,
                                file_paths: List[str],
                                metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Añade documentos desde archivos de texto al vector store.

        Args:
            file_paths (List[str]): Rutas a los archivos de texto
            metadatas (List[Dict], optional): Metadata para cada archivo

        Returns:
            List[str]: IDs de los documentos añadidos

        Example:
            >>> manager = VectorStoreManager("processed_attachments")
            >>> file_paths = ["processed/file1.txt", "processed/file2.txt"]
            >>> metadatas = [
            >>>     {"codigo_cotizacion": "12345", "source_file": "file1.txt"},
            >>>     {"codigo_cotizacion": "12346", "source_file": "file2.txt"}
            >>> ]
            >>> ids = manager.add_documents_from_files(file_paths, metadatas)
        """
        texts = []
        final_metadatas = []

        for i, file_path in enumerate(file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)

                    # Combinar metadata proporcionada con metadata del archivo
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    metadata['source_file'] = os.path.basename(file_path)
                    metadata['file_path'] = file_path
                    final_metadatas.append(metadata)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

        if not texts:
            return []

        return self.add_documents(texts, final_metadatas)

    def similarity_search(self,
                         query: str,
                         k: int = 4,
                         filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Busca documentos similares a la consulta.

        Args:
            query (str): Texto de búsqueda
            k (int): Número de documentos a retornar
            filter (Dict, optional): Filtros de metadata

        Returns:
            List[Document]: Documentos más similares

        Example:
            >>> manager = VectorStoreManager("my_collection")
            >>> results = manager.similarity_search(
            >>>     "computadores portátiles",
            >>>     k=5,
            >>>     filter={"codigo_cotizacion": "12345"}
            >>> )
        """
        if filter:
            return self.vectorstore.similarity_search(query, k=k, filter=filter)
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self,
                                    query: str,
                                    k: int = 4,
                                    filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Busca documentos similares con scores de similitud.

        Args:
            query (str): Texto de búsqueda
            k (int): Número de documentos a retornar
            filter (Dict, optional): Filtros de metadata

        Returns:
            List[tuple]: Lista de tuplas (Document, score)
        """
        if filter:
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self,
                     search_type: str = "similarity",
                     k: int = 4,
                     filter: Optional[Dict[str, Any]] = None):
        """
        Obtiene un retriever para usar con chains de LangChain.

        Args:
            search_type (str): Tipo de búsqueda ('similarity', 'mmr')
            k (int): Número de documentos a retornar
            filter (Dict, optional): Filtros de metadata

        Returns:
            VectorStoreRetriever: Retriever configurado

        Example:
            >>> manager = VectorStoreManager("my_collection")
            >>> retriever = manager.get_retriever(search_type="mmr", k=5)
            >>> # Usar con un chain
            >>> from langchain.chains import RetrievalQA
            >>> qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def delete_collection(self):
        """Elimina toda la colección del vector store"""
        self.vectorstore.delete_collection()

    def get_collection_count(self) -> int:
        """Retorna el número de documentos en la colección"""
        return self.vectorstore._collection.count()


def create_vectorstore(collection_name: str,
                      texts: List[str],
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      persist_directory: str = "./chroma_db",
                      embedding_model: str = "models/text-embedding-004") -> VectorStoreManager:
    """
    Función helper para crear un vector store y añadir documentos en un solo paso.

    Args:
        collection_name (str): Nombre de la colección
        texts (List[str]): Textos a añadir
        metadatas (List[Dict], optional): Metadata para cada texto
        persist_directory (str): Directorio de persistencia
        embedding_model (str): Modelo de embeddings

    Returns:
        VectorStoreManager: Gestor del vector store creado

    Example:
        >>> texts = ["Documento 1", "Documento 2"]
        >>> metadatas = [{"type": "A"}, {"type": "B"}]
        >>> manager = create_vectorstore("my_collection", texts, metadatas)
    """
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )

    if texts:
        manager.add_documents(texts, metadatas)

    return manager


def load_vectorstore(collection_name: str,
                    persist_directory: str = "./chroma_db",
                    embedding_model: str = "models/text-embedding-004") -> VectorStoreManager:
    """
    Carga un vector store existente.

    Args:
        collection_name (str): Nombre de la colección
        persist_directory (str): Directorio de persistencia
        embedding_model (str): Modelo de embeddings

    Returns:
        VectorStoreManager: Gestor del vector store cargado

    Example:
        >>> manager = load_vectorstore("my_collection")
        >>> results = manager.similarity_search("búsqueda de ejemplo")
    """
    return VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )


