import os
import sys
import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.get_agent import get_llm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Normalizador:
    """
    Normaliza atributos extraídos usando diccionarios de embeddings + LLM fallback.

    Usa attribute_dictionary.csv para normalizar valores canónicos y
    attribute_complement.csv para completar atributos derivados.
    """

    def __init__(self, llm_provider: str = None):
        if llm_provider is None:
            llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
        self.llm_provider = llm_provider.lower()
        self.llm = None

    def _initialize_llms(self):
        if self.llm is None:
            temp = float(os.getenv('TEMPERATURE_ADJUNTOS', '0.7'))
            self.llm = get_llm(self.llm_provider, temperature=temp)
            logging.info(f"LLM inicializado: {self.llm_provider} (temp={temp})")

    def _complete_complementary_fields(self, resultado: Dict[str, Any], categoria: str) -> Dict[str, Any]:
        """
        Para cada atributo en resultado que tenga entradas en attribute_complement.csv,
        escribe los atributos derivados en resultado (siempre sobreescribe si hay match).
        """
        from agents.retriever_diccionario import get_complementos
        NO_DISP = {"no disponible", "no especificado", ""}

        for atributo, valor in list(resultado.items()):
            if not valor or str(valor).strip().lower() in NO_DISP:
                continue
            complementos = get_complementos(categoria, atributo, str(valor))
            for comp_atributo, comp_valor in complementos.items():
                prev = resultado.get(comp_atributo)
                resultado[comp_atributo] = comp_valor
                if str(prev) != comp_valor:
                    logging.info(f"  ✓ {comp_atributo} desde '{atributo}'='{valor}': {prev} → {comp_valor}")
                else:
                    logging.info(f"  ✓ {comp_atributo} confirmado: {comp_valor}")

        return resultado

    def aplicar_diccionarios(
        self,
        resultado_adjuntos: Dict[str, Any],
        categoria: str = "",
        tiempos: list = None,
        nodo_nombre: str = "nodo_5_rag_diccionarios",
        similarity_threshold: float = 0.85,
        llm_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Normaliza campos usando similitud de embeddings por campo + LLM fallback opcional.

        Para cada atributo presente en attribute_dictionary.csv para la categoria:
          1. Busca el valor más similar en el FAISS de ese campo (k=3)
          2. Si score >= threshold: usa el valor normalizado automáticamente
          3. Si score < threshold y llm_fallback=True: pregunta al LLM con top-3 candidatos
          4. Si LLM no confirma ninguno: deja el valor original

        Tras normalizar, completa atributos derivados desde attribute_complement.csv.
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

        from agents.retriever_diccionario import (
            normalizar_con_diccionario,
            get_atributos_en_diccionario,
        )

        NO_DISP = {"no disponible", "no especificado", ""}
        atributos_en_dict = get_atributos_en_diccionario(categoria)
        resultado_final = dict(resultado_adjuntos)

        logging.info(
            f"[PASO 2/3] Normalizando con diccionario por campo "
            f"(categoria='{categoria}', threshold={similarity_threshold}, llm_fallback={llm_fallback}, "
            f"ROWNUM: {resultado_adjuntos.get('ROWNUM')})"
        )

        t0_total = time.time()
        campos_normalizados = 0

        campos_a_normalizar = {
            campo: resultado_final[campo]
            for campo in atributos_en_dict
            if resultado_final.get(campo) and str(resultado_final[campo]).lower().strip() not in NO_DISP
        }

        def _worker(campo, valor_actual):
            t0 = time.time()
            valor_norm, score, candidatos = normalizar_con_diccionario(
                categoria, campo, str(valor_actual), k=3, threshold=similarity_threshold
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
                    return campo, None, timing
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
        logging.info(f"  ✓ Normalización completada: {campos_normalizados}/{len(atributos_en_dict)} atributos")

        # PASO 3: Completar atributos derivados desde attribute_complement.csv
        logging.info(f"[PASO 3/3] Completando atributos derivados (ROWNUM: {resultado_adjuntos.get('ROWNUM')})")
        t0 = time.time()
        resultado_final = self._complete_complementary_fields(resultado_final, categoria)
        _t("lookup_complementos",
           f"Completar atributos derivados desde attribute_complement.csv (categoria='{categoria}')",
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
                for v in valores_validos:
                    if v.lower() == respuesta.lower():
                        return v
                return None
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = 2 ** intento
                    logging.warning(f"  LLM 429 para '{campo}' — reintento {intento+1}/{max_intentos} en {wait}s")
                    time.sleep(wait)
                else:
                    logging.warning(f"  LLM fallback error para '{campo}': {e}")
                    return None
        logging.warning(f"  LLM fallback '{campo}': agotados {max_intentos} reintentos por rate limit")
        return None
