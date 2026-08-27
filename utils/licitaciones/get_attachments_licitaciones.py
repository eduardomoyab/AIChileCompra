"""
Descargador de adjuntos para licitaciones.

A diferencia de compra ágil que requiere autenticación con Selenium,
las licitaciones permiten descargar anexos técnicos y económicos
directamente con requests siguiendo el flujo de navegación del portal.
"""

import requests
import os
import logging
import re
from urllib.parse import unquote
from html import unescape


class LicitacionAttachmentDownloader:
    """
    Descargador de adjuntos de licitaciones sin necesidad de autenticación.

    Descarga anexos técnicos y económicos siguiendo el flujo:
    1. Página principal de la licitación
    2. OpeningFrame
    3. OpeningHeader
    4. SupplySummary
    5. Búsqueda por RUT
    6. Descarga de anexos
    """

    def __init__(self, codigo_licitacion: str, rut_proveedor: str, output_folder: str):
        """
        Inicializa el descargador.

        Args:
            codigo_licitacion: Código de la licitación (ej: "1234-567-LE24")
            rut_proveedor: RUT del proveedor
            output_folder: Carpeta donde guardar los adjuntos
        """
        self.codigo_licitacion = codigo_licitacion
        self.rut_proveedor = rut_proveedor
        self.output_folder = output_folder
        self._last_error: str = ""  # paso exacto donde falló la descarga

        # Crear carpetas de salida
        self.tech_folder = os.path.join(output_folder, "tech")
        self.econ_folder = os.path.join(output_folder, "econ")
        self.admin_folder = os.path.join(output_folder, "admin")
        os.makedirs(self.tech_folder, exist_ok=True)
        os.makedirs(self.econ_folder, exist_ok=True)
        os.makedirs(self.admin_folder, exist_ok=True)

        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        logging.info(f"Inicializado descargador para licitación {codigo_licitacion}, RUT {rut_proveedor}")

    def extraer_campos_hidden(self, html):
        """Extrae campos hidden necesarios para el POST."""
        campos = {}
        for name in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTTARGET", "__EVENTARGUMENT"]:
            pattern = rf'id="{name}"[^>]*value="([^"]+)"'
            match = re.search(pattern, html)
            campos[name] = match.group(1) if match else ""
        return campos

    def detectar_extension(self, bin_data):
        """Detecta la extensión del archivo por magic bytes."""
        if bin_data.startswith(b'%PDF'):
            return 'pdf'
        elif bin_data.startswith(b'PK'):
            return 'zip'
        elif bin_data.startswith(b'\x89PNG'):
            return 'png'
        elif bin_data.startswith(b'\xff\xd8'):
            return 'jpg'
        elif bin_data.startswith(b'Rar!'):
            return 'rar'
        elif bin_data.startswith(b'\xd0\xcf\x11\xe0'):
            return 'xls'
        elif bin_data.startswith(b'\x50\x4b\x03\x04') and b'[Content_Types].xml' in bin_data:
            return 'xlsx'
        elif b',' in bin_data[:100] and b'\n' in bin_data[:100]:
            return 'csv'
        else:
            return 'bin'

    def obtener_anexos_rut(self):
        """
        Obtiene los ENC (tokens) de anexos técnicos y económicos para el RUT.

        Returns:
            tuple: (enc_tecnicos_list, enc_economicos_list) o (None, None) si falla
        """
        _TIMEOUT = 30

        try:
            # Paso 1: Página principal
            url_main = f"https://www.mercadopublico.cl/Procurement/Modules/RFB/DetailsAcquisition.aspx?idLicitacion={self.codigo_licitacion}"
            response = self.session.get(url_main, headers=self.headers, timeout=_TIMEOUT)
            if response.status_code != 200:
                self._last_error = f"paso1_pagina_principal: HTTP {response.status_code}"
                logging.error(f"Error en página principal: {response.status_code}")
                return None, None

            # Paso 2: ENC de OpeningFrame
            match = re.search(r'OpeningFrame\.aspx\?enc=([^"&]+)', response.text)
            if not match:
                snippet = response.text[:300].replace('\n', ' ')
                self._last_error = f"paso2_enc_opening_frame: no encontrado. HTML inicio: {snippet}"
                logging.error(f"No se encontró enc de OpeningFrame. HTML inicio: {snippet}")
                return None, None
            enc_frame = unquote(match.group(1))

            # Paso 3: HTML de OpeningFrame
            url_frame = "https://www.mercadopublico.cl/Procurement/Modules/RFB/StepsProcessAward/OpeningFrame.aspx"
            response_frame = self.session.get(url_frame, params={"enc": enc_frame}, headers=self.headers, timeout=_TIMEOUT)
            html_frame = response_frame.text

            # Paso 4: ENC de OpeningHeader
            match_header = re.search(r'OpeningHeader\.aspx\?enc=([^"\s]+)', html_frame)
            if not match_header:
                snippet = html_frame[:300].replace('\n', ' ')
                self._last_error = f"paso4_enc_opening_header: no encontrado. HTML: {snippet}"
                logging.error(f"No se encontró enc de OpeningHeader. HTML: {snippet}")
                return None, None
            enc_header = unquote(match_header.group(1))

            # Paso 5: HTML de OpeningHeader
            url_header = "https://www.mercadopublico.cl/Procurement/Modules/RFB/StepsProcessAward/OpeningHeader.aspx"
            self.headers["Referer"] = url_frame
            response_header = self.session.get(url_header, params={"enc": enc_header}, headers=self.headers, timeout=_TIMEOUT)
            html_header = response_header.text

            # Paso 6: ENC de SupplySummary
            match_summary = re.search(r'SupplySummary\.aspx\?enc=([^"\';]+)', html_header)
            if not match_summary:
                snippet = html_header[:300].replace('\n', ' ')
                self._last_error = f"paso6_enc_supply_summary: no encontrado. HTML: {snippet}"
                logging.error(f"No se encontró enc de SupplySummary. HTML: {snippet}")
                return None, None
            enc_summary = unquote(match_summary.group(1))

            # Paso 7: GET para obtener VIEWSTATE
            url_summary = "https://www.mercadopublico.cl/Procurement/Modules/RFB/StepsProcessAward/SupplySummary.aspx"
            params_summary = {"enc": enc_summary}
            response_get = self.session.get(url_summary, params=params_summary, headers=self.headers, timeout=_TIMEOUT)
            html_get = response_get.text
            campos = self.extraer_campos_hidden(html_get)
            if not campos.get("__VIEWSTATE"):
                snippet = html_get[:300].replace('\n', ' ')
                self._last_error = f"paso7_viewstate: no encontrado en SupplySummary. HTML: {snippet}"
                logging.error(f"VIEWSTATE vacío en SupplySummary. HTML: {snippet}")
                return None, None

            # Paso 8: POST con RUT
            data = {
                "__VIEWSTATE": campos["__VIEWSTATE"],
                "__VIEWSTATEGENERATOR": campos["__VIEWSTATEGENERATOR"],
                "__EVENTTARGET": campos["__EVENTTARGET"],
                "__EVENTARGUMENT": campos["__EVENTARGUMENT"],
                "txtBuscarPorRut": self.rut_proveedor,
                "btnBuscarPorRut": "Buscar",
                "WucPagerGridShidIndex": "0",
                "WucPagerGridShidCountRows": "5",
                "WucPagerGridShidMaxRowCount": "10",
                "WucPagerGridShidSort": ""
            }

            response_post = self.session.post(url_summary, params=params_summary, data=data, headers=self.headers, timeout=_TIMEOUT)
            if response_post.status_code != 200:
                self._last_error = f"paso8_post_rut: HTTP {response_post.status_code}"
                logging.error(f"Error en POST con RUT: {response_post.status_code}")
                return None, None

            html_result = response_post.text

            # Paso 9: Buscar el RUT en filas y extraer los enc
            html_unescaped = unescape(html_result)

            bloques = re.findall(r'<tr[^>]*class="[^"]*?"[^>]*>.*?</tr>', html_unescaped, re.DOTALL | re.IGNORECASE)
            enc_tecnicos = []
            enc_economicos = []
            enc_administrativos = []
            rut_encontrado = False

            for bloque in bloques:
                if self.rut_proveedor in bloque:
                    rut_encontrado = True
                    tecnicos = re.findall(
                        r'title="Anexos Técnicos".*?openPopUp\(\'.*?enc=([^\'"]+)',
                        bloque, re.DOTALL
                    )
                    enc_tecnicos.extend(tecnicos)

                    economicos = re.findall(
                        r'title="Anexos econ[oó]micos".*?openPopUp\(\'.*?enc=([^\'"]+)',
                        bloque, re.DOTALL
                    )
                    enc_economicos.extend(economicos)

                    administrativos = re.findall(
                        r'title="Anexos Administrativos".*?openPopUp\(\'.*?enc=([^\'"]+)',
                        bloque, re.DOTALL
                    )
                    enc_administrativos.extend(administrativos)
                    break

            if not rut_encontrado:
                n_bloques = len(bloques)
                muestra = [b[:80].replace('\n', ' ') for b in bloques[:3]]
                self._last_error = (
                    f"paso9_rut_no_encontrado: RUT '{self.rut_proveedor}' no está en ninguno de "
                    f"{n_bloques} bloques <tr>. Muestra: {muestra}"
                )
                logging.warning(self._last_error)
            elif not enc_tecnicos and not enc_economicos and not enc_administrativos:
                self._last_error = (
                    f"paso9_sin_enc: RUT encontrado en tabla pero sin links de anexos técnicos, económicos ni administrativos"
                )
                logging.warning(self._last_error)

            logging.info(
                f"Encontrados {len(enc_tecnicos)} técnicos, "
                f"{len(enc_economicos)} económicos, "
                f"{len(enc_administrativos)} administrativos"
            )

            return enc_tecnicos or None, enc_economicos or None, enc_administrativos or None

        except Exception as e:
            self._last_error = f"excepcion: {type(e).__name__}: {e}"
            logging.exception(f"Error obteniendo anexos: {e}")
            return None, None

    def descargar_anexo(self, viewstate, viewstategen, ctl_id, nombre_archivo, folder, full_url):
        """Descarga un anexo individual."""
        try:
            data = {
                "__VIEWSTATE": viewstate,
                "__VIEWSTATEGENERATOR": viewstategen,
                f"DWNL$grdId${ctl_id}$search.x": "6",
                f"DWNL$grdId${ctl_id}$search.y": "14",
                "DWNL$ctl10": ""
            }
            r = self.session.post(full_url, headers=self.headers, data=data, timeout=30)
            content = r.content
            ext = self.detectar_extension(content)

            # Rechazar respuestas HTML (portal devuelve página de error o sesión vencida)
            if ext == 'bin' and (content.lstrip()[:15].lower().startswith(b'<!doctype') or
                                  content.lstrip()[:6].lower().startswith(b'<html')):
                snippet = content[:500].decode('utf-8', errors='replace').replace('\n', ' ').strip()
                logging.warning(f"Respuesta HTML inesperada para {nombre_archivo} — omitiendo. Primeros 500 chars: {snippet}")
                return False

            # Usar extensión detectada si el nombre no trae una, o si no coincide con el contenido real
            detected_name = nombre_archivo
            if '.' not in nombre_archivo:
                detected_name = f"{nombre_archivo}.{ext}"
            else:
                declared_ext = os.path.splitext(nombre_archivo)[1].lstrip('.').lower()
                if declared_ext != ext and ext != 'bin':
                    logging.warning(f"Extensión declarada .{declared_ext} no coincide con contenido detectado .{ext} para {nombre_archivo}")

            # Limpiar y hacer nombre único
            safe_name = detected_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            original_safe_name = safe_name
            count = 1
            full_path = os.path.join(folder, safe_name)
            while os.path.exists(full_path):
                safe_name = f"{os.path.splitext(original_safe_name)[0]}_{count}{os.path.splitext(original_safe_name)[1]}"
                full_path = os.path.join(folder, safe_name)
                count += 1

            with open(full_path, "wb") as f:
                f.write(content)

            logging.info(f"Descargado: {nombre_archivo} -> {safe_name}")
            return True

        except Exception as e:
            logging.exception(f"Error descargando {nombre_archivo}: {e}")
            return False

    def download_attachments_from_enc(self, enc_value, folder):
        """Descarga todos los anexos de un ENC dado."""
        from bs4 import BeautifulSoup

        try:
            base_url = "https://www.mercadopublico.cl/BID/Modules/POPUPS/ViewBidAttachment.aspx"
            full_url = f"{base_url}?enc={enc_value}"

            self.headers["Referer"] = full_url

            # GET inicial usando la sesión establecida (mantiene cookies de navegación)
            r = self.session.get(full_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            viewstate = soup.find("input", {"id": "__VIEWSTATE"})["value"]
            viewstategen = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})["value"]

            # Encontrar anexos
            anexos = {}
            for input_tag in soup.find_all("input", {"type": "image", "name": True}):
                if "search" in input_tag['name']:
                    ctl_id = input_tag['name'].split('$')[2]
                    span_id = f"DWNL_grdId_{ctl_id}_File"
                    span_tag = soup.find("span", {"id": span_id})
                    if span_tag:
                        nombre_archivo = span_tag.text.strip()
                        anexos[ctl_id] = nombre_archivo

            # Descargar cada anexo — contar solo los exitosos
            count = 0
            for ctl_id, nombre_archivo in anexos.items():
                if self.descargar_anexo(viewstate, viewstategen, ctl_id, nombre_archivo, folder, full_url):
                    count += 1

            return count

        except Exception as e:
            logging.exception(f"Error descargando anexos de enc {enc_value}: {e}")
            return 0

    def download_all(self):
        """
        Descarga todos los anexos (técnicos y económicos).

        Returns:
            bool: True si se descargó al menos un anexo
        """
        try:
            # Obtener ENC de anexos
            encs_tech, encs_econ, encs_admin = self.obtener_anexos_rut()

            if not encs_tech and not encs_econ and not encs_admin:
                # _last_error ya fue seteado en obtener_anexos_rut
                return False

            total_descargados = 0

            # Descargar técnicos
            if encs_tech:
                logging.info(f"Descargando {len(encs_tech)} grupos de anexos técnicos...")
                for enc in encs_tech:
                    count = self.download_attachments_from_enc(enc, self.tech_folder)
                    total_descargados += count

            # Descargar económicos
            if encs_econ:
                logging.info(f"Descargando {len(encs_econ)} grupos de anexos económicos...")
                for enc in encs_econ:
                    count = self.download_attachments_from_enc(enc, self.econ_folder)
                    total_descargados += count

            # Descargar administrativos
            if encs_admin:
                logging.info(f"Descargando {len(encs_admin)} grupos de anexos administrativos...")
                for enc in encs_admin:
                    count = self.download_attachments_from_enc(enc, self.admin_folder)
                    total_descargados += count

            if total_descargados == 0 and not self._last_error:
                self._last_error = "enc encontrados pero 0 archivos descargados exitosamente"

            logging.info(f"Total descargados: {total_descargados} archivos")

            return total_descargados > 0

        except Exception as e:
            self._last_error = f"excepcion_download_all: {type(e).__name__}: {e}"
            logging.exception(f"Error en download_all: {e}")
            return False
