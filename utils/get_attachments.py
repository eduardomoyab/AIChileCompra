import os
import time
import logging
import requests
import shutil
import zipfile
import rarfile
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
GECKO_DRIVER_PATH = os.getenv("GECKO_DRIVER_PATH", "C:/SeleniumDrivers/geckodriver.exe")
OUTPUT_BASE_PATH = os.getenv("ATTACHMENTS_OUTPUT_PATH", "attachments")

# Constantes de espera
MAX_WAIT_TIME = 60
WAIT_LOGIN_EXTRA = 5
CLICK_RETRY_COUNT = 5
SLEEP_BEFORE_CLICK = 3
LOGIN_BUTTON_DELAY = 5


class MercadoPublicoAttachmentDownloader:
    """Clase para gestionar la descarga de adjuntos desde Mercado Público"""

    def __init__(self, headless: bool = True):
        """
        Inicializa el downloader de adjuntos.

        Args:
            headless (bool): Si se ejecuta el navegador en modo headless
        """
        self.headless = headless
        self.driver = None
        self.token_bearer = None
        self.setup_logging()

    def setup_logging(self):
        """Configura el logging básico"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_driver(self):
        """Configura y retorna el driver de Firefox"""
        firefox_options = Options()
        if self.headless:
            firefox_options.add_argument("--headless")

        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
        firefox_options.set_preference(
            "browser.helperApps.neverAsk.saveToDisk",
            "application/pdf, application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, "
            "application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, image/jpeg, "
            "image/png, image/gif, image/bmp, image/tiff, application/zip, application/octet-stream"
        )
        firefox_options.set_preference("pdfjs.disabled", True)
        firefox_options.set_preference("browser.helperApps.alwaysAsk.force", False)

        service = Service(executable_path=GECKO_DRIVER_PATH)
        return webdriver.Firefox(service=service, options=firefox_options)

    def login_mercado_publico(self):
        """Realiza el login en Mercado Público y obtiene el token de acceso"""
        if self.driver is None:
            self.driver = self.setup_driver()

        self.driver.get("https://www.mercadopublico.cl/Home")
        logging.info("Visiting Mercado Publico home page")

        # Click en botón de acceso
        for attempt in range(CLICK_RETRY_COUNT):
            try:
                access_button = WebDriverWait(self.driver, MAX_WAIT_TIME).until(
                    EC.element_to_be_clickable((By.XPATH, '/html/body/header/nav/div[2]/div[1]/div/button'))
                )
                self.driver.execute_script("arguments[0].scrollIntoView(); arguments[0].click();", access_button)
                time.sleep(3)
                if self._check_clave_unica_button():
                    break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to click access button: {e}")

        # Click en botón de Clave Única
        for attempt in range(CLICK_RETRY_COUNT):
            try:
                time.sleep(SLEEP_BEFORE_CLICK)
                clave_unica_button = WebDriverWait(self.driver, MAX_WAIT_TIME).until(
                    EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[2]/div/div/section/div/div[1]/div[2]/div/div/div/div/div/div/ul/a/img'))
                )
                self.driver.execute_script("arguments[0].scrollIntoView(); arguments[0].click();", clave_unica_button)
                time.sleep(3)
                if self._check_login_page():
                    break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to click clave unica button: {e}")

        # Ingresar credenciales
        WebDriverWait(self.driver, MAX_WAIT_TIME).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="uname"]'))
        )

        username = os.getenv("CU_USER")
        password = os.getenv("CU_PASSWORD")

        if not username or not password:
            raise ValueError("CU_USER y CU_PASSWORD deben estar configurados en .env")

        self.driver.find_element(By.XPATH, '//*[@id="uname"]').send_keys(username)
        self.driver.find_element(By.XPATH, '//*[@id="pword"]').send_keys(password)
        time.sleep(LOGIN_BUTTON_DELAY)

        login_button = self.driver.find_element(By.XPATH, '//*[@id="login-submit"]')
        if login_button.is_enabled():
            login_button.click()
            logging.info("Login submitted")

        # Esperar post-login y seleccionar usuario
        WebDriverWait(self.driver, MAX_WAIT_TIME).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/header/nav/div[2]/div[1]/div/div[2]/div/div/div/div[2]/div/table/tbody/tr/td/div[2]/label/span'))
        )
        time.sleep(WAIT_LOGIN_EXTRA)

        # Seleccionar usuario y obtener token
        select_user = self.driver.find_element(By.XPATH, '//*[@id="rdbOrg1984080"]')
        select_user.click()
        time.sleep(SLEEP_BEFORE_CLICK)

        access_btn = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn.btn-pri"))
        )
        access_btn.click()
        time.sleep(WAIT_LOGIN_EXTRA)

        # Extraer token
        cookies = self.driver.get_cookies()
        for cookie in cookies:
            if cookie['name'] == 'access_token_ccr':
                self.token_bearer = cookie['value']
                logging.info("Token obtained successfully")
                break

    def _check_clave_unica_button(self) -> bool:
        """Verifica si el botón de Clave Única está presente"""
        try:
            self.driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/div/section/div/div[1]/div[2]/div/div/div/div/div/div/ul/a/img')
            return True
        except:
            return False

    def _check_login_page(self) -> bool:
        """Verifica si la página de login se ha cargado"""
        try:
            self.driver.find_element(By.XPATH, '//*[@id="uname"]')
            return True
        except:
            return False

    def _get_with_retries(self, url: str, headers: dict = None, params: dict = None,
                          max_retries: int = 5, wait_seconds: int = 2, stream: bool = False):
        """Realiza peticiones GET con reintentos"""
        for intento in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params, stream=stream)
                if response.status_code == 200:
                    return response
                else:
                    logging.warning(f"Attempt {intento}: Status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {intento}: Request error: {e}")
            time.sleep(wait_seconds)
        raise Exception(f"Failed to get 200 response after {max_retries} attempts")

    def _extract_and_flatten(self, file_path: str, target_dir: str):
        """Extrae archivos ZIP/RAR y aplana la estructura"""
        temp_extract_dir = os.path.join(target_dir, "__temp_extract__")
        os.makedirs(temp_extract_dir, exist_ok=True)

        try:
            if file_path.lower().endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(temp_extract_dir)
            elif file_path.lower().endswith('.rar'):
                with rarfile.RarFile(file_path, 'r') as rf:
                    rf.extractall(temp_extract_dir)

            # Mover archivos al directorio destino
            for root, _, files in os.walk(temp_extract_dir):
                if "_MACOSX" in root:
                    continue
                for file in files:
                    origen = os.path.join(root, file)
                    destino = os.path.join(target_dir, file)

                    # Evitar duplicados
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
        """Limpia el nombre del archivo de caracteres inválidos"""
        invalid_chars = [" ", ":", "/", "\\", "|", "?", "*", "<", ">", '"', "'", ";", ",", "&", "%", "$", "#"]
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    def download_attachments(self, codigo_cotizacion: str, rut_proveedor: str,
                           auto_login: bool = True) -> dict:
        """
        Descarga los adjuntos de una cotización específica.

        Args:
            codigo_cotizacion (str): Código de la solicitud de cotización
            rut_proveedor (str): RUT del proveedor
            auto_login (bool): Si debe hacer login automáticamente si no hay token

        Returns:
            dict: Información sobre la descarga con las siguientes claves:
                - success (bool): Si la descarga fue exitosa
                - files_downloaded (list): Lista de archivos descargados
                - output_path (str): Ruta donde se guardaron los archivos
                - error (str, optional): Mensaje de error si falló
        """
        # Verificar token
        if not self.token_bearer and auto_login:
            logging.info("No token found, performing login...")
            self.login_mercado_publico()

        if not self.token_bearer:
            return {
                "success": False,
                "error": "No token available. Please login first.",
                "files_downloaded": [],
                "output_path": None
            }

        # Crear carpeta de salida
        output_path = os.path.join(OUTPUT_BASE_PATH, str(codigo_cotizacion), str(rut_proveedor))
        os.makedirs(output_path, exist_ok=True)

        files_downloaded = []

        try:
            # URLs de la API
            url_cotizacion = f'https://servicios-compra-agil.mercadopublico.cl/v1/compra-agil/solicitud/{codigo_cotizacion}'

            headers = {
                'Authorization': f'Bearer {self.token_bearer}',
                'Content-Type': 'application/json'
            }

            params = {'size': 500, 'page': 0}

            # Obtener información de la cotización
            response_cotizacion = self._get_with_retries(url_cotizacion, headers=headers, params=params)
            data = response_cotizacion.json()
            ofertas = data['payload']['ofertas']

            # Buscar la oferta seleccionada
            oferta_id = None
            for oferta in ofertas:
                if oferta['esOfertaSeleccionada'] == 1:
                    oferta_id = oferta['id']
                    break

            if not oferta_id:
                return {
                    "success": False,
                    "error": "No selected offer found",
                    "files_downloaded": [],
                    "output_path": output_path
                }

            # Obtener adjuntos
            url_attachments = f'https://servicios-compra-agil.mercadopublico.cl/v1/compra-agil/solicitud/cotizacion/{oferta_id}'
            response_attachments = self._get_with_retries(url_attachments, headers=headers)

            data_attachments = response_attachments.json()
            adjuntos = data_attachments['payload']['documentosAdjuntos']

            # Descargar cada adjunto
            for adjunto in adjuntos:
                adjunto_id = adjunto['id']
                filename = self._sanitize_filename(adjunto['filename'])

                url_download = f'https://servicios-compra-agil.mercadopublico.cl/v1/compra-agil/proveedor/cotizacion/descargarAdjunto/{adjunto_id}'

                file_path = os.path.join(output_path, filename)

                # No descargar si ya existe
                if os.path.exists(file_path):
                    logging.info(f"File already exists: {filename}")
                    files_downloaded.append(filename)
                    continue

                response_download = self._get_with_retries(url_download, headers=headers, stream=True)

                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(response_download.raw, f)

                logging.info(f"Downloaded: {filename}")
                files_downloaded.append(filename)

                # Extraer si es ZIP o RAR
                if filename.lower().endswith(('.zip', '.rar')):
                    logging.info(f"Extracting: {filename}")
                    self._extract_and_flatten(file_path, output_path)

            return {
                "success": True,
                "files_downloaded": files_downloaded,
                "output_path": output_path,
                "total_files": len(files_downloaded)
            }

        except Exception as e:
            logging.exception(f"Error downloading attachments: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_downloaded": files_downloaded,
                "output_path": output_path
            }

    def close(self):
        """Cierra el driver del navegador"""
        if self.driver:
            self.driver.quit()
            logging.info("Driver closed")


def download_attachments_simple(codigo_cotizacion: str, rut_proveedor: str,
                                headless: bool = True,
                                downloader: Optional[object] = None) -> dict:
    """
    Función simple para descargar adjuntos sin necesidad de gestionar la clase.

    Args:
        codigo_cotizacion (str): Código de la solicitud de cotización
        rut_proveedor (str): RUT del proveedor
        headless (bool): Si ejecutar el navegador en modo headless (solo si downloader es None)
        downloader (Optional[MercadoPublicoAttachmentDownloader]): Downloader existente con sesión activa.
                                                                    Si se proporciona, se reutiliza la sesión.

    Returns:
        dict: Resultado de la descarga

    Example:
        >>> # Con sesión nueva (modo antiguo)
        >>> result = download_attachments_simple("12345678", "76123456-7")
        >>>
        >>> # Con sesión existente (recomendado)
        >>> global_downloader = MercadoPublicoAttachmentDownloader(headless=True)
        >>> global_downloader.login_mercado_publico()
        >>> result = download_attachments_simple("12345678", "76123456-7", downloader=global_downloader)
    """
    # Si se proporciona un downloader, usarlo directamente
    if downloader is not None:
        return downloader.download_attachments(codigo_cotizacion, rut_proveedor, auto_login=False)

    # Modo antiguo: crear downloader temporal
    downloader = MercadoPublicoAttachmentDownloader(headless=headless)
    try:
        result = downloader.download_attachments(codigo_cotizacion, rut_proveedor)
        return result
    finally:
        downloader.close()


