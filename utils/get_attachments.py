import os
import time
import logging
import requests
import shutil
import zipfile
import rarfile
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configuración
OUTPUT_BASE_PATH = os.getenv("ATTACHMENTS_OUTPUT_PATH", "attachments")

# Claves públicas embebidas en el JS bundle de buscador.mercadopublico.cl
# (no requieren login ni credenciales)
BASE_BUSCADOR = "https://api.buscador.mercadopublico.cl"
BASE_ADJUNTO = "https://adjunto.mercadopublico.cl/adjunto-compra-agil"
BUSCADOR_API_KEY = "e93089e4-437c-4723-b343-4fa20045e3bc"
USER_KEY = "41186b85826e80d1a0d445a6ce67d1a3"


class BuscadorAttachmentDownloader:
    """Descarga adjuntos de Compra Ágil usando la API pública del Buscador.
    No requiere login, Selenium ni credenciales."""

    def __init__(self):
        self._headers_buscador = {"x-api-key": BUSCADOR_API_KEY}
        self._headers_adjunto = {"user_key": USER_KEY}
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_with_retries(self, url: str, headers: dict = None, params: dict = None,
                          max_retries: int = 3, wait_seconds: int = 2, stream: bool = False):
        """Realiza peticiones GET con reintentos ante errores transitorios."""
        for intento in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params,
                                        stream=stream, timeout=10)
                if response.status_code == 200:
                    return response
                logging.warning(f"Intento {intento}: status {response.status_code} en {url}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Intento {intento}: error de red: {e}")
            time.sleep(wait_seconds)
        raise Exception(f"No se obtuvo respuesta 200 tras {max_retries} intentos: {url}")

    def _extract_and_flatten(self, file_path: str, target_dir: str):
        """Extrae archivos ZIP/RAR y aplana la estructura de directorios."""
        temp_extract_dir = os.path.join(target_dir, "__temp_extract__")
        os.makedirs(temp_extract_dir, exist_ok=True)

        try:
            if file_path.lower().endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(temp_extract_dir)
            elif file_path.lower().endswith(".rar"):
                with rarfile.RarFile(file_path, "r") as rf:
                    rf.extractall(temp_extract_dir)

            for root, _, files in os.walk(temp_extract_dir):
                if "_MACOSX" in root:
                    continue
                for file in files:
                    origen = os.path.join(root, file)
                    destino = os.path.join(target_dir, file)
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(destino):
                        destino = os.path.join(target_dir, f"{base}_{count}{ext}")
                        count += 1
                    shutil.move(origen, destino)
        finally:
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            os.remove(file_path)

    def _sanitize_filename(self, filename: str) -> str:
        """Elimina caracteres inválidos del nombre de archivo."""
        invalid_chars = [" ", ":", "/", "\\", "|", "?", "*", "<", ">", '"', "'",
                         ";", ",", "&", "%", "$", "#"]
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    # ------------------------------------------------------------------
    # Flujo principal
    # ------------------------------------------------------------------

    def _get_winner_cotizacion_id(self, codigo_cotizacion: str, rut_proveedor: str = None) -> Optional[int]:
        """
        Consulta la ficha de la cotización en el Buscador y retorna el
        id_cotizacion del proveedor seleccionado (ganador).
        """
        url = f"{BASE_BUSCADOR}/compra-agil?action=ficha&code={codigo_cotizacion}"
        resp = self._get_with_retries(url, headers=self._headers_buscador)
        ficha = resp.json().get("payload") or {}

        proveedores = ficha.get("proveedores_cotizando", [])
        if not proveedores:
            logging.warning(f"No hay proveedores en la ficha de {codigo_cotizacion}")
            return None

        # 1. Buscar por proveedor_seleccionado (acepta 1, True, "1")
        ganador = next(
            (p for p in proveedores if p.get("proveedor_seleccionado") in (1, True, "1")),
            None
        )

        # 2. Fallback: buscar por rut_proveedor
        if ganador is None and rut_proveedor:
            rut_normalizado = rut_proveedor.replace(".", "").replace("-", "").lower()
            ganador = next(
                (p for p in proveedores
                 if str(p.get("rut_proveedor", "")).replace(".", "").replace("-", "").lower() == rut_normalizado),
                None
            )
            if ganador:
                logging.info(f"Proveedor encontrado por RUT: {rut_proveedor}")

        # 3. Último recurso: primer proveedor si solo hay uno
        if ganador is None and len(proveedores) == 1:
            ganador = proveedores[0]
            logging.info(f"Un solo proveedor en la ficha — usando directamente")

        if ganador is None:
            logging.warning(
                f"No se encontró proveedor ganador para {codigo_cotizacion}. "
                f"Proveedores: {[p.get('rut_proveedor') for p in proveedores]}"
            )
            return None

        logging.info(
            f"Proveedor ganador: {ganador.get('razon_social')} "
            f"(RUT: {ganador.get('rut_proveedor')}, "
            f"id_cotizacion: {ganador.get('id_cotizacion')})"
        )
        return ganador["id_cotizacion"]

    def _list_files(self, id_cotizacion: int) -> list:
        """
        Retorna la lista de archivos adjuntos para la cotización del ganador.
        Cada elemento tiene al menos {id, nombreArchivo}.
        """
        url = (f"{BASE_ADJUNTO}/v1/adjuntos-compra-agil"
               f"/cotizacion/listar/{id_cotizacion}")
        resp = self._get_with_retries(url, headers=self._headers_adjunto)
        payload = resp.json().get("payload") or {}
        return payload.get("files", [])

    def download_attachments(self, codigo_cotizacion: str, rut_proveedor: str) -> dict:
        """
        Descarga los adjuntos del proveedor ganador de una Compra Ágil.

        Args:
            codigo_cotizacion: Código de la cotización (ej. "2927-350-COT25").
            rut_proveedor: RUT del proveedor — se usa solo para la ruta de salida.

        Returns:
            dict con claves: success, files_downloaded, output_path, total_files, error.
        """
        output_path = os.path.join(OUTPUT_BASE_PATH,
                                   str(codigo_cotizacion),
                                   str(rut_proveedor))
        os.makedirs(output_path, exist_ok=True)
        files_downloaded = []

        try:
            # Paso 1: identificar id_cotizacion del proveedor ganador
            id_cot = self._get_winner_cotizacion_id(codigo_cotizacion, rut_proveedor)
            if id_cot is None:
                return {
                    "success": False,
                    "error": "No se encontró proveedor seleccionado en la ficha",
                    "files_downloaded": [],
                    "output_path": output_path,
                }

            # Paso 2: listar adjuntos
            files = self._list_files(id_cot)
            if not files:
                logging.info(f"Sin adjuntos para cotización {codigo_cotizacion}")
                return {
                    "success": True,
                    "files_downloaded": [],
                    "output_path": output_path,
                    "total_files": 0,
                }

            # Paso 3: descargar cada adjunto
            for file_info in files:
                file_id = file_info["id"]
                filename = self._sanitize_filename(file_info["nombreArchivo"])
                file_path = os.path.join(output_path, filename)

                if os.path.exists(file_path):
                    logging.info(f"Ya existe, omitiendo: {filename}")
                    files_downloaded.append(filename)
                    continue

                url_dl = (f"{BASE_ADJUNTO}/v1/adjuntos-compra-agil"
                          f"/descargar/{file_id}")
                resp_dl = self._get_with_retries(
                    url_dl, headers=self._headers_adjunto, stream=True
                )

                with open(file_path, "wb") as fout:
                    for chunk in resp_dl.iter_content(chunk_size=8192):
                        fout.write(chunk)

                logging.info(f"Descargado: {filename} ({os.path.getsize(file_path)} bytes)")
                files_downloaded.append(filename)

                if filename.lower().endswith((".zip", ".rar")):
                    logging.info(f"Extrayendo: {filename}")
                    self._extract_and_flatten(file_path, output_path)

            return {
                "success": True,
                "files_downloaded": files_downloaded,
                "output_path": output_path,
                "total_files": len(files_downloaded),
            }

        except Exception as e:
            logging.exception(f"Error descargando adjuntos: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_downloaded": files_downloaded,
                "output_path": output_path,
            }

    # Mantener compatibilidad — no hay sesión que cerrar
    def close(self):
        pass


# ---------------------------------------------------------------------------
# Flujo autenticado (token Bearer manual, sin Selenium)
# ---------------------------------------------------------------------------

BASE_SERVICIOS = "https://servicios-compra-agil.mercadopublico.cl"


class TokenAttachmentDownloader:
    """Descarga adjuntos usando las APIs internas de Compra Ágil con un
    token Bearer proporcionado manualmente. No usa Selenium."""

    def __init__(self, token_bearer: str):
        self.token_bearer = token_bearer
        self._headers = {
            "Authorization": f"Bearer {token_bearer}",
            "Content-Type": "application/json",
        }

    def _get_with_retries(self, url: str, headers: dict = None, params: dict = None,
                          max_retries: int = 3, wait_seconds: int = 2, stream: bool = False):
        for intento in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params,
                                        stream=stream, timeout=10)
                if response.status_code == 200:
                    return response
                logging.warning(f"Intento {intento}: status {response.status_code} en {url}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Intento {intento}: error de red: {e}")
            time.sleep(wait_seconds)
        raise Exception(f"No se obtuvo respuesta 200 tras {max_retries} intentos: {url}")

    def _extract_and_flatten(self, file_path: str, target_dir: str):
        temp_extract_dir = os.path.join(target_dir, "__temp_extract__")
        os.makedirs(temp_extract_dir, exist_ok=True)
        try:
            if file_path.lower().endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(temp_extract_dir)
            elif file_path.lower().endswith(".rar"):
                with rarfile.RarFile(file_path, "r") as rf:
                    rf.extractall(temp_extract_dir)
            for root, _, files in os.walk(temp_extract_dir):
                if "_MACOSX" in root:
                    continue
                for file in files:
                    origen = os.path.join(root, file)
                    destino = os.path.join(target_dir, file)
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(destino):
                        destino = os.path.join(target_dir, f"{base}_{count}{ext}")
                        count += 1
                    shutil.move(origen, destino)
        finally:
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            os.remove(file_path)

    def _sanitize_filename(self, filename: str) -> str:
        invalid_chars = [" ", ":", "/", "\\", "|", "?", "*", "<", ">", '"', "'",
                         ";", ",", "&", "%", "$", "#"]
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    def download_attachments(self, codigo_cotizacion: str, rut_proveedor: str) -> dict:
        """Descarga adjuntos usando el token Bearer proporcionado."""
        output_path = os.path.join(OUTPUT_BASE_PATH,
                                   str(codigo_cotizacion),
                                   str(rut_proveedor))
        os.makedirs(output_path, exist_ok=True)
        files_downloaded = []

        try:
            # Paso 1: obtener oferta seleccionada
            url_cot = f"{BASE_SERVICIOS}/v1/compra-agil/solicitud/{codigo_cotizacion}"
            resp = self._get_with_retries(url_cot, headers=self._headers,
                                          params={"size": 500, "page": 0})
            ofertas = (resp.json().get("payload") or {}).get("ofertas", [])
            oferta_id = next(
                (o["id"] for o in ofertas if o.get("esOfertaSeleccionada") == 1),
                None
            )
            if oferta_id is None:
                return {
                    "success": False,
                    "error": "No se encontró oferta seleccionada en la cotización",
                    "files_downloaded": [],
                    "output_path": output_path,
                }

            # Paso 2: listar adjuntos
            url_adj = f"{BASE_SERVICIOS}/v1/compra-agil/solicitud/cotizacion/{oferta_id}"
            resp2 = self._get_with_retries(url_adj, headers=self._headers)
            adjuntos = (resp2.json().get("payload") or {}).get("documentosAdjuntos", [])

            if not adjuntos:
                return {
                    "success": True,
                    "files_downloaded": [],
                    "output_path": output_path,
                    "total_files": 0,
                }

            # Paso 3: descargar cada adjunto
            for adjunto in adjuntos:
                adjunto_id = adjunto["id"]
                filename = self._sanitize_filename(adjunto["filename"])
                file_path = os.path.join(output_path, filename)

                if os.path.exists(file_path):
                    logging.info(f"Ya existe, omitiendo: {filename}")
                    files_downloaded.append(filename)
                    continue

                url_dl = (f"{BASE_SERVICIOS}/v1/compra-agil/proveedor"
                          f"/cotizacion/descargarAdjunto/{adjunto_id}")
                resp_dl = self._get_with_retries(url_dl, headers=self._headers, stream=True)

                with open(file_path, "wb") as fout:
                    shutil.copyfileobj(resp_dl.raw, fout)

                logging.info(f"Descargado: {filename} ({os.path.getsize(file_path)} bytes)")
                files_downloaded.append(filename)

                if filename.lower().endswith((".zip", ".rar")):
                    logging.info(f"Extrayendo: {filename}")
                    self._extract_and_flatten(file_path, output_path)

            return {
                "success": True,
                "files_downloaded": files_downloaded,
                "output_path": output_path,
                "total_files": len(files_downloaded),
            }

        except Exception as e:
            logging.exception(f"Error descargando adjuntos con token: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_downloaded": files_downloaded,
                "output_path": output_path,
            }

    def close(self):
        pass


class LicitacionDownloaderAdapter:
    """
    Adapter que envuelve LicitacionAttachmentDownloader para que sea compatible
    con la interfaz de BuscadorAttachmentDownloader/TokenAttachmentDownloader.

    Descarga los adjuntos de la licitación (tech + econ) y los aplana en una
    carpeta plana, de modo que process_attachments_simple pueda encontrarlos.
    """

    def download_attachments(self, codigo_licitacion: str, rut_proveedor: str) -> dict:
        from utils.licitaciones.get_attachments_licitaciones import LicitacionAttachmentDownloader

        output_base = os.getenv("ATTACHMENTS_OUTPUT_PATH", "attachments")
        output_folder = os.path.join(output_base, codigo_licitacion, rut_proveedor)
        processed_dir = os.path.join(output_base, codigo_licitacion, "processed")

        cache_ttl = int(os.getenv("ATTACHMENTS_CACHE_TTL_SECONDS", "0"))

        # Misma lógica de caché que compra ágil
        if os.path.exists(output_folder):
            existing = [f for f in os.listdir(output_folder)
                        if os.path.isfile(os.path.join(output_folder, f))]
            if existing:
                if cache_ttl <= 0:
                    shutil.rmtree(output_folder, ignore_errors=True)
                    if os.path.exists(processed_dir):
                        shutil.rmtree(processed_dir, ignore_errors=True)
                    logging.info(f"  Cache TTL=0 — directorio limpiado: {output_folder}")
                else:
                    newest_mtime = max(
                        os.path.getmtime(os.path.join(output_folder, f)) for f in existing
                    )
                    age = time.time() - newest_mtime
                    if age > cache_ttl:
                        shutil.rmtree(output_folder, ignore_errors=True)
                        if os.path.exists(processed_dir):
                            shutil.rmtree(processed_dir, ignore_errors=True)
                        logging.info(f"  Cache expirado ({age:.0f}s > TTL {cache_ttl}s) — directorio limpiado")
                    else:
                        logging.info(f"✓ Archivos ya descargados: {len(existing)} archivos en caché")
                        return {
                            "success": True,
                            "files_downloaded": existing,
                            "output_path": output_folder,
                            "total_files": len(existing),
                        }

        os.makedirs(output_folder, exist_ok=True)

        dl = LicitacionAttachmentDownloader(
            codigo_licitacion=codigo_licitacion,
            rut_proveedor=rut_proveedor,
            output_folder=output_folder,
        )
        download_error: Optional[str] = None
        try:
            success = dl.download_all()
            if not success:
                download_error = "No se encontraron o descargaron anexos técnicos ni económicos en la licitación"
        except Exception as e:
            success = False
            download_error = f"{type(e).__name__}: {str(e)}"
            logging.exception(f"Excepción en dl.download_all(): {e}")

        # Aplanar: mover archivos de tech/ y econ/ al nivel raíz
        files_downloaded = []
        for subfolder in ["tech", "econ"]:
            sub_path = os.path.join(output_folder, subfolder)
            if not os.path.exists(sub_path):
                continue
            for fname in os.listdir(sub_path):
                src = os.path.join(sub_path, fname)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(output_folder, fname)
                base, ext = os.path.splitext(fname)
                count = 1
                while os.path.exists(dst):
                    dst = os.path.join(output_folder, f"{base}_{count}{ext}")
                    count += 1
                shutil.move(src, dst)
                files_downloaded.append(os.path.basename(dst))
            shutil.rmtree(sub_path, ignore_errors=True)

        result: dict = {
            "success": success,
            "files_downloaded": files_downloaded,
            "output_path": output_folder,
            "total_files": len(files_downloaded),
        }
        if not success:
            result["error"] = download_error or "Error desconocido en descarga"
        return result

    def close(self):
        pass


def download_attachments_simple(codigo_cotizacion: str, rut_proveedor: str,
                                headless: bool = True,
                                downloader: Optional[object] = None) -> dict:
    """
    Descarga adjuntos de una Compra Ágil.

    - Si se pasa un downloader (ej. TokenAttachmentDownloader), lo usa directamente.
    - Si no, usa la API pública del Buscador (sin login).

    Args:
        codigo_cotizacion: Código de la cotización (ej. "2927-350-COT25").
        rut_proveedor: RUT del proveedor (usado para la ruta de salida).
        headless: Ignorado — mantenido para compatibilidad.
        downloader: Instancia de TokenAttachmentDownloader u otro downloader compatible.

    Returns:
        dict con claves: success, files_downloaded, output_path, total_files, error.
    """
    if downloader is not None and hasattr(downloader, "download_attachments"):
        return downloader.download_attachments(codigo_cotizacion, rut_proveedor)

    dl = BuscadorAttachmentDownloader()
    return dl.download_attachments(codigo_cotizacion, rut_proveedor)
