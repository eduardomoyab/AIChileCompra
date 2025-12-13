import os
import sys
import logging
import yaml
import json
import re
import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.get_agent import get_llm
from agents.get_vectorstore import create_faiss_from_files
# Ya no se necesita el retriever de diccionarios
# from agents.retriever_diccionario_comp import search_diccionario_balanced, format_docs_for_llm

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rutas de configuración
AGENTS_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_RAG_ADJ = os.path.join(AGENTS_PATH, "RAG_adjuntos_comp.yaml")
# Ya no se usan diccionarios ni normalización
# PATH_RAG_DIC = os.path.join(AGENTS_PATH, "RAG_diccionario_comp.yaml")
PATH_CONFIG = os.path.join(AGENTS_PATH, "config_comp.yaml")
# PATH_DICT_PROCESADOR = "diccionarios/Computadores/diccionario_procesador.xlsx"


# Modelos Pydantic para validación de respuestas
class ProductoExtraido(BaseModel):
    """Modelo para un producto extraído desde adjuntos"""
    ROWNUM: str = Field(..., description="Identificador único del producto")
    Tipo: str = Field(..., description="Tipo de producto: AIO, Laptop, Desktop, Otro")
    Part_Number: str = Field(alias="Part Number", description="Número de parte del fabricante")
    Modelo: str = Field(..., description="Modelo completo del producto")

    class Config:
        populate_by_name = True  # Permite usar tanto "Part_Number" como "Part Number"


class CatalogacionComputadores:
    """
    Clase para catalogar productos de computación (Desktops, Notebooks, All in One).

    El proceso incluye:
    1. Extracción de atributos (Tipo, Part Number, Modelo) usando RAG con adjuntos procesados

    Atributos extraídos:
    - Tipo: AIO, Laptop, Desktop, Otro
    - Part Number: Código del fabricante (P/N, SKU)
    - Modelo: Modelo completo del producto
    """

    def __init__(self,
                 path_rag_adj: str = PATH_RAG_ADJ,
                 path_config: str = PATH_CONFIG,
                 llm_provider: str = None):
        """
        Inicializa el catalogador de computadores.

        Args:
            path_rag_adj: Ruta al template de RAG para adjuntos
            path_config: Ruta al archivo de configuración
            llm_provider: Proveedor de LLM ('openai', 'gemini', 'deepseek').
                         Si es None, usa DEFAULT_LLM_PROVIDER del .env
        """
        self.path_rag_adj = path_rag_adj
        self.path_config = path_config

        # Cargar configuraciones
        self._load_configs()

        # Configurar LLM
        if llm_provider is None:
            llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

        self.llm_provider = llm_provider.lower()
        self.llm = None  # Solo un LLM para extracción de adjuntos

        # Ya no se usa diccionario de procesadores
        # self._load_processor_dict()

    def _load_configs(self):
        """Carga las configuraciones desde archivos YAML"""
        with open(self.path_rag_adj, 'r', encoding='utf-8') as f:
            self.config_rag_adj = yaml.safe_load(f)

        # Ya no se carga config de diccionarios
        # with open(self.path_rag_dic, 'r', encoding='utf-8') as f:
        #     self.config_rag_dic = yaml.safe_load(f)

        with open(self.path_config, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        logging.info("Configuraciones cargadas exitosamente")

    # Ya no se necesita cargar diccionario de procesadores
    # def _load_processor_dict(self):
    #     """Carga el diccionario de procesadores para mapeo de núcleos e hilos"""
    #     dict_path = os.path.join(os.path.dirname(AGENTS_PATH), PATH_DICT_PROCESADOR)
    #
    #     if os.path.exists(dict_path):
    #         df_proc = pd.read_excel(dict_path)
    #         self.dict_nucleos = df_proc.set_index('Procesador')['Nucleos'].to_dict()
    #         self.dict_hilos = df_proc.set_index('Procesador')['Hilos'].to_dict()
    #         logging.info(f"Diccionario de procesadores cargado: {len(df_proc)} procesadores")
    #     else:
    #         logging.warning(f"No se encontró el diccionario de procesadores en {dict_path}")
    #         self.dict_nucleos = {}
    #         self.dict_hilos = {}

    def _initialize_llms(self):
        """Inicializa el modelo LLM si aún no está creado"""
        if self.llm is None:
            # Leer temperatura desde .env
            temp = float(os.getenv('TEMPERATURE_ADJUNTOS', '0.7'))
            self.llm = get_llm(self.llm_provider, temperature=temp)
            logging.info(f"LLM inicializado: {self.llm_provider} (temp={temp})")

    def format_docs(self, docs):
        """Formatea documentos para el contexto RAG"""
        return "\n\n".join(doc.page_content for doc in docs)

    def augment_query(self, question: str) -> str:
        """Aumenta la query con términos técnicos relevantes"""
        texto_extra = (
            "\n\nAtributos: Tipo, Modelo, Part Number, P/N, SKU, "
            "Código de Producto, Laptop, Notebook, Desktop, AIO, All in One, "
            "Especificaciones"
        )
        return question + texto_extra

    def catalogar_producto(
        self,
        payload: Dict[str, Any],
        codigo_cotizacion: str,
        rut_proveedor: str,
        processed_path: str,
        use_diccionarios: bool = True,
        calcular_tokens: bool = False
    ) -> Dict[str, Any]:
        """
        Cataloga un producto individual de computación.

        Args:
            payload (dict): Información del producto con estructura:
                {
                    'ROWNUM': str,
                    'DescripcionProductoComprador': str,
                    'DescripcionProductoProveedor': str,
                    'productoname': str
                }
            codigo_cotizacion (str): Código de la solicitud de cotización
            rut_proveedor (str): RUT del proveedor
            processed_path (str): Ruta al directorio con archivos .txt procesados
            use_diccionarios (bool): Si usar normalización con diccionarios
            calcular_tokens (bool): Si calcular y guardar tokens usados

        Returns:
            dict: Resultado de la catalogación con todos los atributos

        Example:
            >>> catalogador = CatalogacionComputadores()
            >>> payload = {
            ...     'ROWNUM': '1',
            ...     'DescripcionProductoComprador': 'NOTEBOOK HP 8GB RAM',
            ...     'DescripcionProductoProveedor': 'Notebook HP Pavilion',
            ...     'productoname': 'Computadores'
            ... }
            >>> resultado = catalogador.catalogar_producto(
            ...     payload, "12345678", "76123456-7"
            ... )
        """
        self._initialize_llms()

        # PASO 1: RAG con adjuntos procesados
        logging.info(f"[PASO 1/1] Extrayendo atributos desde adjuntos (ROWNUM: {payload.get('ROWNUM')})")

        # Obtener archivos .txt procesados
        txt_files = [
            os.path.join(processed_path, f)
            for f in os.listdir(processed_path)
            if f.endswith('.txt') and f != 'skipped_files.txt'
        ]

        if not txt_files:
            logging.warning(f"No se encontraron archivos procesados en {processed_path}")
            return self._create_empty_result(payload.get('ROWNUM'))

        logging.info(f"Creando FAISS en memoria con {len(txt_files)} archivos...")

        # Crear metadata para archivos
        metadatas = [
            {
                "codigo_cotizacion": codigo_cotizacion,
                "rut_proveedor": rut_proveedor,
                "source_file": os.path.basename(f)
            }
            for f in txt_files
        ]

        # Crear FAISS en memoria
        try:
            vectorstore = create_faiss_from_files(txt_files, metadatas)
            logging.info(f"✓ FAISS creado con {len(txt_files)} documentos")
        except Exception as e:
            logging.error(f"Error creando FAISS: {e}")
            return self._create_empty_result(payload.get('ROWNUM'))

        # Crear chain RAG para adjuntos
        template_adj = self.config_rag_adj['rag_template']
        prompt_adj = PromptTemplate.from_template(template_adj)

        # Ejecutar retrieval
        json_str = str([payload])
        augmented_query = self.augment_query(json_str)
        k_adj = int(os.getenv('SEARCH_K_ADJUNTOS', '5'))
        retrieved_docs = vectorstore.similarity_search(augmented_query, k=k_adj)

        # Log de documentos recuperados
        logging.info(f"Documentos recuperados del RAG: {len(retrieved_docs)} documentos")
        for i, doc in enumerate(retrieved_docs, 1):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            logging.info(f"  Doc {i}: {content_preview}...")

        # Formatear contexto
        context = self.format_docs(retrieved_docs)
        logging.info(f"Contexto RAG (primeros 300 chars): {context[:300]}...")

        # Crear y ejecutar chain
        rag_chain_adj = (
            prompt_adj
            | self.llm
            | StrOutputParser()
        )

        # Ejecutar con contexto y pregunta
        response_adj = rag_chain_adj.invoke({"context": context, "question": json_str})

        # Parsear respuesta (viene como tupla)
        resultado_adj = self._parse_response_adjuntos(response_adj)

        if not resultado_adj:
            logging.warning(f"No se pudo parsear la respuesta de adjuntos para ROWNUM {payload.get('ROWNUM')}")
            return self._create_empty_result(payload.get('ROWNUM'))

        # PASO 2: Ya no se usa normalización con diccionarios para estos atributos simples
        # El resultado de adjuntos es suficiente para Tipo, Part Number y Modelo
        resultado_final = resultado_adj

        # if use_diccionarios:
        #     logging.info(f"[PASO 2/3] Normalizando con diccionarios (ROWNUM: {payload.get('ROWNUM')})")
        #
        #     # Buscar en diccionarios (1 de cada origen)
        #     k_dic = int(os.getenv('SEARCH_K_DICCIONARIOS', '2'))
        #     query_dic = f"{resultado_adj.get('Procesador', '')} {resultado_adj.get('Marca', '')}"
        #     docs_dic = search_diccionario_balanced(query_dic, k_total=k_dic)
        #     context_dic = format_docs_for_llm(docs_dic)
        #
        #     # Crear prompt para normalización
        #     template_dic = self.config_rag_dic['rag_template']
        #     prompt_dic = PromptTemplate.from_template(template_dic)
        #
        #     # Convertir resultado_adj a texto formateado
        #     texto_producto = self._format_producto_for_normalizacion(resultado_adj)
        #
        #     # Ejecutar normalización
        #     prompt_dic_filled = prompt_dic.format(context=context_dic, question=texto_producto)
        #     response_dic = self.llm_dic.invoke(prompt_dic_filled)
        #
        #     # Parsear respuesta del diccionario (viene como JSON)
        #     resultado_normalizado = self._parse_response_diccionario(response_dic.content if hasattr(response_dic, 'content') else response_dic)
        #
        #     if resultado_normalizado:
        #         resultado_final = resultado_normalizado
        #     else:
        #         logging.warning("No se pudo normalizar, usando resultado de adjuntos")
        #         resultado_final = resultado_adj
        # else:
        #     resultado_final = resultado_adj

        # PASO 3: Ya no se completan núcleos e hilos (se eliminó ese atributo)
        # logging.info(f"[PASO 3/3] Completando núcleos e hilos (ROWNUM: {payload.get('ROWNUM')})")
        # resultado_final = self._complete_processor_specs(resultado_final)

        logging.info(f"✓ Catalogación completada para ROWNUM {payload.get('ROWNUM')}")
        return resultado_final

    def catalogar_lote(
        self,
        payloads: List[Dict[str, Any]],
        codigo_cotizacion: str,
        rut_proveedor: str,
        use_diccionarios: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Cataloga un lote de productos.

        Args:
            payloads (list): Lista de payloads de productos
            codigo_cotizacion (str): Código de la solicitud de cotización
            rut_proveedor (str): RUT del proveedor
            use_diccionarios (bool): Si usar normalización con diccionarios

        Returns:
            list: Lista de resultados de catalogación

        Example:
            >>> catalogador = CatalogacionComputadores()
            >>> payloads = [
            ...     {'ROWNUM': '1', 'DescripcionProductoComprador': '...', ...},
            ...     {'ROWNUM': '2', 'DescripcionProductoComprador': '...', ...}
            ... ]
            >>> resultados = catalogador.catalogar_lote(
            ...     payloads, "12345678", "76123456-7"
            ... )
        """
        resultados = []

        for payload in payloads:
            try:
                resultado = self.catalogar_producto(
                    payload,
                    codigo_cotizacion,
                    rut_proveedor,
                    use_diccionarios=use_diccionarios
                )
                resultados.append(resultado)
            except Exception as e:
                logging.error(f"Error catalogando ROWNUM {payload.get('ROWNUM')}: {e}")
                resultados.append(self._create_empty_result(payload.get('ROWNUM')))

        return resultados

    def _parse_response_adjuntos(self, response: str) -> Optional[Dict[str, Any]]:
        """Parsea la respuesta del RAG de adjuntos (formato JSON) usando Pydantic"""
        try:
            # Limpiar respuesta - remover bloques de código markdown si existen
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]  # Remover ```json
            elif response.startswith('```'):
                response = response[3:]  # Remover ```
            if response.endswith('```'):
                response = response[:-3]  # Remover ``` final
            response = response.strip()

            # Log de la respuesta recibida para debugging
            logging.info(f"Respuesta LLM recibida (primeros 200 chars): {response[:200]}")

            # Parsear JSON
            data = json.loads(response)

            if not data or len(data) == 0:
                logging.warning("Respuesta JSON vacía")
                return None

            # Tomar el primer objeto
            primer_producto = data[0]

            # Validar con Pydantic
            producto_validado = ProductoExtraido(**primer_producto)

            # Convertir a diccionario con el nombre correcto del campo
            resultado = {
                'ROWNUM': producto_validado.ROWNUM,
                'Tipo': producto_validado.Tipo,
                'Part Number': producto_validado.Part_Number,
                'Modelo': producto_validado.Modelo
            }

            logging.info(f"Producto parseado exitosamente: {resultado}")
            return resultado

        except json.JSONDecodeError as e:
            logging.error(f"Error parseando JSON: {e}")
            logging.error(f"Respuesta recibida: {response}")
            return None
        except ValidationError as e:
            logging.error(f"Error de validación Pydantic: {e}")
            logging.error(f"Datos recibidos: {primer_producto if 'primer_producto' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logging.error(f"Error parseando respuesta de adjuntos: {e}")
            logging.error(f"Respuesta recibida: {response}")
            return None

    def _parse_response_diccionario(self, response: str) -> Optional[Dict[str, Any]]:
        """Parsea la respuesta del RAG de diccionarios (formato JSON)"""
        try:
            # Limpiar respuesta
            response = response.replace('```json', '').replace('```', '')
            response = response.replace('null', '"No disponible"')
            response = response.strip()

            # Parsear JSON
            data = json.loads(response)

            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            else:
                return None

        except Exception as e:
            logging.error(f"Error parseando respuesta de diccionario: {e}")
            return None

    def _format_producto_for_normalizacion(self, producto: Dict[str, Any]) -> str:
        """Formatea el producto para el prompt de normalización"""
        texto = f"Producto {producto.get('ROWNUM', 'N/A')}:\n"
        for key, value in producto.items():
            if key != 'ROWNUM':
                texto += f"  - {key}: {value}\n"
        return texto.strip()

    # Ya no se necesita completar núcleos e hilos
    # def _complete_processor_specs(self, resultado: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Completa núcleos e hilos basándose en el procesador usando diccionario.
    #
    #     Solo completa si los valores actuales son "No disponible".
    #     """
    #     procesador = resultado.get('Procesador', '')
    #
    #     # Solo completar si el procesador no es "No disponible"
    #     if procesador and procesador != 'No disponible':
    #         # Completar núcleos
    #         if resultado.get('Núcleos') == 'No disponible' or not resultado.get('Núcleos'):
    #             nucleos = self.dict_nucleos.get(procesador)
    #             if nucleos:
    #                 resultado['Núcleos'] = str(nucleos)
    #                 logging.info(f"  Completado Núcleos: {nucleos}")
    #
    #         # Completar hilos
    #         if resultado.get('Hilos Procesador') == 'No disponible' or not resultado.get('Hilos Procesador'):
    #             hilos = self.dict_hilos.get(procesador)
    #             if hilos:
    #                 resultado['Hilos Procesador'] = str(hilos)
    #                 logging.info(f"  Completado Hilos: {hilos}")
    #
    #     return resultado

    def _create_empty_result(self, rownum: str) -> Dict[str, Any]:
        """Crea un resultado vacío con todos los campos en 'No disponible'"""
        return {
            'ROWNUM': rownum,
            'Tipo': 'No disponible',
            'Part Number': 'No disponible',
            'Modelo': 'No disponible'
        }


# Función helper para uso directo
def catalogar_computador(
    payload: Dict[str, Any],
    codigo_cotizacion: str,
    rut_proveedor: str,
    llm_provider: str = None,
    use_diccionarios: bool = True
) -> Dict[str, Any]:
    """
    Función helper para catalogar un solo producto de computación.

    Args:
        payload: Información del producto
        codigo_cotizacion: Código de la solicitud de cotización
        rut_proveedor: RUT del proveedor
        llm_provider: Proveedor de LLM (None = usa DEFAULT_LLM_PROVIDER del .env)
        use_diccionarios: Si usar normalización con diccionarios

    Returns:
        dict: Resultado de la catalogación

    Example:
        >>> payload = {
        ...     'ROWNUM': '1',
        ...     'DescripcionProductoComprador': 'NOTEBOOK HP 8GB RAM',
        ...     'DescripcionProductoProveedor': 'Notebook HP Pavilion',
        ...     'productoname': 'Computadores'
        ... }
        >>> resultado = catalogar_computador(payload, "12345678", "76123456-7")
        >>> print(resultado['Marca'])  # HP
        >>> print(resultado['RAM (GB)'])  # 8
    """
    catalogador = CatalogacionComputadores(llm_provider=llm_provider)
    return catalogador.catalogar_producto(
        payload,
        codigo_cotizacion,
        rut_proveedor,
        use_diccionarios=use_diccionarios
    )

