import os
import sys
import logging
import time
import yaml
import json
import re
import unicodedata
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.get_agent import get_llm
from agents.get_vectorstore import create_faiss_from_files
from agents.retriever_diccionario_comp import search_diccionario_balanced, format_docs_for_llm

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
PATH_CONFIG = os.path.join(AGENTS_PATH, "config_comp.yaml")
PATH_DICT_PROCESADOR = "diccionarios/Computadores/diccionario_procesador.xlsx"


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
        self.llm = None  # LLM para extracción de adjuntos

        # Cargar diccionario de procesadores
        self._load_processor_dict()

    def _load_configs(self):
        """Carga las configuraciones desde archivos YAML"""
        with open(self.path_rag_adj, 'r', encoding='utf-8') as f:
            self.config_rag_adj = yaml.safe_load(f)

        logging.info("Configuraciones cargadas exitosamente")

    @staticmethod
    def _normalize_proc(name: str) -> str:
        """
        Normaliza un nombre de procesador para comparación flexible.
        Convierte a minúsculas, reemplaza guiones/barras por espacios,
        colapsa espacios y elimina diacríticos.
        """
        # Eliminar diacríticos (ñ → n, é → e, etc.)
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        name = name.lower().strip()
        # Eliminar todo carácter no alfanumérico (guiones, barras, espacios, etc.)
        name = re.sub(r'[^a-z0-9]', '', name)
        return name

    def _load_processor_dict(self):
        """Carga el diccionario de procesadores para mapeo de núcleos e hilos"""
        dict_path = os.path.join(os.path.dirname(AGENTS_PATH), PATH_DICT_PROCESADOR)

        if os.path.exists(dict_path):
            df_proc = pd.read_excel(dict_path)
            self.dict_nucleos = df_proc.set_index('Procesador')['Nucleos'].to_dict()
            self.dict_hilos = df_proc.set_index('Procesador')['Hilos'].to_dict()
            # Dicts normalizados para fallback sin distinción de guiones/mayúsculas
            self.dict_nucleos_norm = {
                self._normalize_proc(k): v for k, v in self.dict_nucleos.items()
            }
            self.dict_hilos_norm = {
                self._normalize_proc(k): v for k, v in self.dict_hilos.items()
            }
            logging.info(f"Diccionario de procesadores cargado: {len(df_proc)} procesadores")
        else:
            logging.warning(f"No se encontró el diccionario de procesadores en {dict_path}")
            self.dict_nucleos = {}
            self.dict_hilos = {}
            self.dict_nucleos_norm = {}
            self.dict_hilos_norm = {}

    def _initialize_llms(self):
        """Inicializa los modelos LLM si aún no están creados"""
        if self.llm is None:
            # Leer temperatura desde .env
            temp = float(os.getenv('TEMPERATURE_ADJUNTOS', '0.7'))
            self.llm = get_llm(self.llm_provider, temperature=temp)
            logging.info(f"LLM (adjuntos) inicializado: {self.llm_provider} (temp={temp})")

    def format_docs(self, docs):
        """Formatea documentos para el contexto RAG"""
        return "\n\n".join(doc.page_content for doc in docs)

    def augment_query(self, question: str) -> str:
        """Aumenta la query con términos técnicos relevantes para mejorar el retrieval"""
        texto_extra = (
            "\n\nAtributos: Tipo, Modelo, Part Number, P/N, SKU, Código de Producto, "
            "Laptop, Notebook, Desktop, AIO, All in One, "
            "Procesador, CPU, Intel, AMD, Apple, Core, Ryzen, "
            "RAM, Memoria, DDR4, DDR5, LPDDR, Memoria Unificada, "
            "Almacenamiento, SSD, HDD, NVMe, eMMC, Disco, "
            "Sistema Operativo, Windows, macOS, Linux, "
            "Pantalla, Display, Monitor, pulgadas, "
            "Marca, Fabricante, Especificaciones, Ficha Técnica"
        )
        return question + texto_extra

    def catalogar_producto(
        self,
        payload: Dict[str, Any],
        codigo_cotizacion: str,
        rut_proveedor: str,
        processed_path: str,
        use_diccionarios: bool = True,
        calcular_tokens: bool = False,
        tiempos: list = None,
        nodo_nombre: str = "nodo_3_rag_adjuntos",
        vectorstore=None,
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

        def _t(fase, desc, segundos):
            """Append timing a la lista compartida si existe."""
            if tiempos is not None:
                tiempos.append({
                    "nodo": nodo_nombre,
                    "fase": fase,
                    "descripcion": desc,
                    "segundos": round(segundos, 4),
                })

        # PASO 1: RAG con adjuntos procesados
        logging.info(f"[PASO 1/1] Extrayendo atributos desde adjuntos (ROWNUM: {payload.get('ROWNUM')})")

        if vectorstore is None:
            # Sub-paso: filtrar archivos admin
            t0 = time.time()
            all_txt_files = [
                os.path.join(processed_path, f)
                for f in os.listdir(processed_path)
                if f.endswith('.txt') and f != 'skipped_files.txt'
            ]
            skip_patterns_raw = os.getenv('SKIP_FILENAME_PATTERNS', 'anexo,formulario,propuesta,simulacion')
            skip_patterns = [p.strip().lower() for p in skip_patterns_raw.split(',') if p.strip()]
            tech_keywords_raw = os.getenv(
                'TECH_FILENAME_KEYWORDS',
                'procesador,cpu,ram,memoria,ssd,hdd,disco,pantalla,monitor,display,'
                'intel,amd,nvidia,gpu,ficha,especificacion,tecnica,tecnico,'
                'almacenamiento,bateria,ghz'
            )
            tech_keywords = [kw.strip().lower() for kw in tech_keywords_raw.split(',') if kw.strip()]

            def _is_admin_file(filename: str) -> bool:
                fname = filename.lower()
                has_admin = any(p in fname for p in skip_patterns)
                if not has_admin:
                    return False
                has_tech = any(kw in fname for kw in tech_keywords)
                return not has_tech

            txt_files_filtered = [f for f in all_txt_files if not _is_admin_file(os.path.basename(f))]
            if txt_files_filtered:
                excluded = len(all_txt_files) - len(txt_files_filtered)
                if excluded > 0:
                    logging.info(f"Filtrado admin: {len(all_txt_files)} → {len(txt_files_filtered)} archivos ({excluded} excluidos)")
                txt_files = txt_files_filtered
            else:
                logging.warning("SKIP_FILENAME_PATTERNS excluyó todos los archivos — usando todos sin filtro")
                txt_files = all_txt_files
            _t("filtrar_archivos",
               f"Filtrar archivos admin ({len(all_txt_files)} total → {len(txt_files)} válidos)",
               time.time() - t0)

            if not txt_files:
                logging.warning(f"No se encontraron archivos procesados en {processed_path}")
                return self._create_empty_result(payload.get('ROWNUM'))

            # Sub-paso: crear FAISS
            t0 = time.time()
            logging.info(f"Creando FAISS en memoria con {len(txt_files)} archivos...")
            metadatas = [
                {
                    "codigo_cotizacion": codigo_cotizacion,
                    "rut_proveedor": rut_proveedor,
                    "source_file": os.path.basename(f)
                }
                for f in txt_files
            ]
            try:
                chunk_size = int(os.getenv('ADJUNTOS_CHUNK_SIZE', '500'))
                chunk_overlap = int(os.getenv('ADJUNTOS_CHUNK_OVERLAP', '100'))
                logging.info(f"Creando FAISS con chunks de {chunk_size} caracteres (overlap: {chunk_overlap})...")
                vectorstore = create_faiss_from_files(
                    txt_files, metadatas,
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                logging.info(f"✓ FAISS creado con chunking para extracción de atributos fijos")
            except Exception as e:
                logging.error(f"Error creando FAISS: {e}")
                return self._create_empty_result(payload.get('ROWNUM'))
            _t("crear_faiss",
               f"Crear FAISS adjuntos ({len(txt_files)} archivos, chunk={chunk_size})",
               time.time() - t0)
        else:
            logging.info("✓ Reutilizando FAISS pre-construido (nodo_3)")

        # Sub-paso: retrieval (similarity search)
        t0 = time.time()
        template_adj = self.config_rag_adj['rag_template']
        prompt_adj = PromptTemplate.from_template(template_adj)
        json_str = str([payload])
        augmented_query = self.augment_query(json_str)
        k_adj = int(os.getenv('SEARCH_K_ADJUNTOS', '5'))
        retrieved_docs = vectorstore.similarity_search(augmented_query, k=k_adj)
        logging.info(f"Documentos recuperados del RAG: {len(retrieved_docs)} documentos")
        for i, doc in enumerate(retrieved_docs, 1):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            logging.info(f"  Doc {i}: {content_preview}...")
        context = self.format_docs(retrieved_docs)
        logging.info(f"Contexto RAG (primeros 300 chars): {context[:300]}...")
        _t("retrieval",
           f"Similarity search FAISS (k={k_adj}, docs recuperados={len(retrieved_docs)})",
           time.time() - t0)

        # Sub-paso: llamada LLM
        t0 = time.time()
        rag_chain_adj = (
            prompt_adj
            | self.llm
            | StrOutputParser()
        )
        response_adj = rag_chain_adj.invoke({"context": context, "question": json_str})
        _t("llm_invoke",
           f"Llamada LLM ({self.llm_provider}) para extracción Tipo/PartNumber/Modelo",
           time.time() - t0)

        # Sub-paso: parseo de respuesta
        t0 = time.time()
        resultado_adj = self._parse_response_adjuntos(response_adj)
        _t("parse_respuesta",
           "Parseo JSON respuesta LLM + validación Pydantic",
           time.time() - t0)

        if not resultado_adj:
            logging.warning(f"No se pudo parsear la respuesta de adjuntos para ROWNUM {payload.get('ROWNUM')}")
            return self._create_empty_result(payload.get('ROWNUM'))

        logging.info(f"[INFO] PASO 2 y 3 deshabilitados en catalogar_producto - usar aplicar_diccionarios() después")
        resultado_final = resultado_adj

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

    def _complete_processor_specs(self, resultado: Dict[str, Any]) -> Dict[str, Any]:
        """
        Completa núcleos e hilos basándose en el procesador usando diccionario.

        Solo completa si los valores actuales son "No disponible", "No especificado" o vacíos.
        Maneja tanto "Nucleos" (CAMPOS_MANUALES) como "nucleos" (API normalizada).
        """
        # Buscar procesador (case-insensitive)
        procesador = resultado.get('Procesador') or resultado.get('procesador', '')

        # Solo completar si el procesador no es "No disponible"
        if procesador and procesador not in ['No disponible', 'No especificado']:
            proc_norm = self._normalize_proc(procesador)

            # Nucleos: siempre sobreescribir con el diccionario si el procesador está en él
            nucleos = self.dict_nucleos.get(procesador) or self.dict_nucleos_norm.get(proc_norm)
            if nucleos:
                for key in ('Nucleos', 'nucleos'):
                    if key in resultado:
                        prev = resultado[key]
                        resultado[key] = str(nucleos)
                        if str(prev) != str(nucleos):
                            logging.info(f"  ✓ Nucleos corregido: {prev} → {nucleos} (diccionario)")
                        else:
                            logging.info(f"  ✓ Nucleos confirmado: {nucleos} (diccionario)")
            else:
                logging.info(f"  ⚠ Procesador no encontrado en diccionario de núcleos: '{procesador}'")

            # Hilos: siempre sobreescribir con el diccionario si el procesador está en él
            hilos = self.dict_hilos.get(procesador) or self.dict_hilos_norm.get(proc_norm)
            if hilos:
                for key in ('Hilos', 'hilos'):
                    if key in resultado:
                        prev = resultado[key]
                        resultado[key] = str(hilos)
                        if str(prev) != str(hilos):
                            logging.info(f"  ✓ Hilos corregido: {prev} → {hilos} (diccionario)")
                        else:
                            logging.info(f"  ✓ Hilos confirmado: {hilos} (diccionario)")

        return resultado

    def aplicar_diccionarios(
        self,
        resultado_adjuntos: Dict[str, Any],
        tiempos: list = None,
        nodo_nombre: str = "nodo_5_rag_diccionarios",
        similarity_threshold: float = 0.85,
        llm_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Normaliza campos usando similitud de embeddings por campo + LLM fallback opcional.

        Para cada campo presente en diccionario_features.xlsx:
          1. Busca el valor más similar en el FAISS de ese campo (k=3)
          2. Si score >= threshold: usa el valor normalizado automáticamente
          3. Si score < threshold y llm_fallback=True: pregunta al LLM con top-3 candidatos
          4. Si LLM no confirma ninguno: deja el valor original

        Tras normalizar, completa Nucleos/Hilos si el procesador está en el diccionario.

        Args:
            resultado_adjuntos: Resultado previo (adjuntos + campos manuales mezclados)
            similarity_threshold: Score mínimo para aceptar match automático (0-1)
            llm_fallback: Si usar LLM cuando el score no alcanza el threshold
        """
        self._initialize_llms()

        def _t(fase, desc, segundos):
            if tiempos is not None:
                tiempos.append({
                    "nodo": nodo_nombre,
                    "fase": fase,
                    "descripcion": desc,
                    "segundos": round(segundos, 4),
                })

        from agents.retriever_diccionario_comp import (
            normalizar_con_features,
            get_campos_en_features,
            get_nucleos_hilos,
        )

        NO_DISP = {"no disponible", "no especificado", ""}
        campos_en_features = get_campos_en_features()
        resultado_final = dict(resultado_adjuntos)

        logging.info(
            f"[PASO 2/3] Normalizando con features por campo "
            f"(threshold={similarity_threshold}, llm_fallback={llm_fallback}, "
            f"ROWNUM: {resultado_adjuntos.get('ROWNUM')})"
        )

        t0_total = time.time()
        campos_normalizados = 0

        # Recolectar campos con valor válido para normalizar
        campos_a_normalizar = {
            campo: resultado_final[campo]
            for campo in campos_en_features
            if resultado_final.get(campo) and str(resultado_final[campo]).lower().strip() not in NO_DISP
        }

        def _worker(campo, valor_actual):
            """Normaliza un campo de forma independiente (ejecutado en paralelo)."""
            t0 = time.time()
            valor_norm, score, candidatos = normalizar_con_features(
                campo, str(valor_actual), k=3, threshold=similarity_threshold
            )
            if score >= similarity_threshold:
                return campo, valor_norm, [(f"sim_{campo}", f"Similitud automática '{campo}' (score={score:.3f})", time.time() - t0)]
            elif llm_fallback and candidatos:
                t0_llm = time.time()
                valor_llm = self._llm_resolver_match(campo, valor_actual, candidatos)
                elapsed_llm = time.time() - t0_llm
                timing = [(f"llm_fallback_{campo}", f"LLM fallback '{campo}' (best_score={score:.3f})", elapsed_llm)]
                if valor_llm:
                    logging.info(f"  [LLM fallback] '{campo}': '{valor_actual}' → '{valor_llm}'")
                    return campo, valor_llm, timing
                else:
                    logging.info(f"  [LLM fallback] '{campo}': sin match para '{valor_actual}' (best_score={score:.3f})")
                    return campo, None, timing  # registrar el intento aunque no haya match
            return campo, None, []

        max_workers = min(len(campos_a_normalizar), 5) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, campo, valor): campo
                for campo, valor in campos_a_normalizar.items()
            }
            for future in as_completed(futures):
                campo, nuevo_valor, timings = future.result()
                for fase, desc, seg in timings:
                    _t(fase, desc, seg)
                if nuevo_valor is not None:
                    resultado_final[campo] = nuevo_valor
                    campos_normalizados += 1

        _t("normalizacion_total",
           f"Normalización features por campo ({campos_normalizados} campos normalizados)",
           time.time() - t0_total)
        logging.info(f"  ✓ Normalización completada: {campos_normalizados}/{len(campos_en_features)} campos")

        # PASO 3: Completar Nucleos/Hilos desde diccionario de procesadores
        logging.info(f"[PASO 3/3] Completando núcleos e hilos (ROWNUM: {resultado_adjuntos.get('ROWNUM')})")
        t0 = time.time()
        resultado_final = self._complete_processor_specs(resultado_final)
        _t("lookup_nucleos_hilos",
           "Completar Nucleos/Hilos desde diccionario Excel (lookup directo)",
           time.time() - t0)

        logging.info(f"✓ Diccionarios aplicados para ROWNUM {resultado_adjuntos.get('ROWNUM')}")
        return resultado_final

    def _llm_resolver_match(
        self,
        campo: str,
        valor_extraido: str,
        candidatos: list,
    ) -> Optional[str]:
        """
        Usa el LLM para decidir si alguno de los candidatos es el mismo concepto
        que el valor extraído. Retorna el candidato exacto o None si ninguno matchea.
        """
        if not candidatos:
            return None

        lineas = "\n".join(
            f"{i+1}. {v} (similitud: {s:.2f})"
            for i, (v, s) in enumerate(candidatos)
        )
        prompt = (
            f"Se extrajo el valor \"{valor_extraido}\" para el campo \"{campo}\".\n\n"
            f"Candidatos del diccionario (por similitud descendente):\n{lineas}\n\n"
            f"¿Alguno de estos candidatos es el mismo producto/versión base que \"{valor_extraido}\"?\n"
            f"Criterio: acepta si el candidato es la versión estándar del mismo producto "
            f"(ej: 'Windows 11 Home Single Language' → 'Microsoft Windows 11 Home', "
            f"'Win 11 Pro' → 'Microsoft Windows 11 Pro', 'macOS Sonoma' → 'macOS').\n"
            f"Responde ÚNICAMENTE con el texto exacto del candidato si hay match, "
            f"o con la palabra \"ninguno\" si ninguno corresponde."
        )
        max_intentos = 4
        for intento in range(max_intentos):
            try:
                response = self.llm.invoke(prompt)
                respuesta = response.content.strip()
                valores_validos = [v for v, _ in candidatos]
                if respuesta in valores_validos:
                    return respuesta
                # Tolerar respuesta en minúsculas
                for v in valores_validos:
                    if v.lower() == respuesta.lower():
                        return v
                return None
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = 2 ** intento  # 1s, 2s, 4s, 8s
                    logging.warning(f"  LLM 429 para '{campo}' — reintento {intento+1}/{max_intentos} en {wait}s")
                    time.sleep(wait)
                else:
                    logging.warning(f"  LLM fallback error para '{campo}': {e}")
                    return None
        logging.warning(f"  LLM fallback '{campo}': agotados {max_intentos} reintentos por rate limit")
        return None

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

