import os
import io
import logging
import unicodedata
import gc
import numpy as np
from tqdm import tqdm
from docx import Document
from openpyxl import load_workbook
from bs4 import BeautifulSoup
from typing import Optional, List
import fitz  # PyMuPDF
import easyocr
from PIL import Image

# Configuración por defecto
MAX_PAGES = 15
TEXT_THRESHOLD = 100
MAX_FILENAME_LENGTH = 100

# Reader EasyOCR compartido entre todos los AttachmentProcessor del proceso
# Se inicializa una vez al importar el módulo (una vez por worker de gunicorn)
try:
    import torch
    _gpu = torch.cuda.is_available()
except ImportError:
    _gpu = False

try:
    _easyocr_reader = easyocr.Reader(['es', 'en'], gpu=_gpu)
    logging.info(f"EasyOCR inicializado al cargar módulo (gpu={_gpu})")
except Exception as _e:
    _easyocr_reader = None
    logging.warning(f"EasyOCR no disponible: {_e}")


class AttachmentProcessor:
    """Procesa archivos adjuntos extrayendo texto de diversos formatos.
    PDFs e imágenes: PyMuPDF (texto nativo) + EasyOCR (OCR para páginas escaneadas).
    """

    def __init__(self, attachments_path: str, output_path: Optional[str] = None,
                 blacklist: Optional[List[str]] = None, use_gpu: bool = False):
        """
        Inicializa el procesador de adjuntos.

        Args:
            attachments_path (str): Ruta a la carpeta con los adjuntos descargados
            output_path (str, optional): Ruta donde guardar los archivos procesados
            blacklist (list, optional): Lista de palabras para filtrar archivos
            use_gpu (bool): Si usar GPU para EasyOCR
        """
        self.attachments_path = os.path.abspath(attachments_path)

        if output_path is None:
            parent_dir = os.path.dirname(self.attachments_path)
            self.output_path = os.path.join(parent_dir, 'processed')
        else:
            self.output_path = os.path.abspath(output_path)

        self.blacklist = blacklist if blacklist else []

        # Reutilizar el reader global inicializado al cargar el módulo
        self.reader = _easyocr_reader

        # Archivos omitidos
        self.skipped_files = set()
        self.skip_files_path = os.path.join(self.output_path, 'skipped_files.txt')

        # Crear directorio de salida
        os.makedirs(self.output_path, exist_ok=True)

        # Cargar archivos previamente omitidos
        self._load_skipped_files()

    def _load_skipped_files(self):
        """Carga la lista de archivos previamente omitidos"""
        if os.path.exists(self.skip_files_path):
            with open(self.skip_files_path, 'r', encoding='utf-8') as file:
                self.skipped_files = set(line.strip() for line in file)

    def _add_to_skipped_files(self, file_name: str):
        """Agrega un archivo a la lista de omitidos"""
        if file_name not in self.skipped_files:
            with open(self.skip_files_path, 'a', encoding='utf-8') as file:
                file.write(f"{file_name}\n")
            self.skipped_files.add(file_name)

    def _normalize_filename(self, filename: str) -> str:
        """Normaliza el nombre del archivo removiendo caracteres especiales"""
        normalized = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
        normalized = normalized.replace(' ', '_')

        if len(normalized) > MAX_FILENAME_LENGTH:
            normalized = normalized[:MAX_FILENAME_LENGTH]
            logging.warning(f"Filename truncated to {MAX_FILENAME_LENGTH} chars: {normalized}")

        return normalized

    def _contains_blacklisted_word(self, text: str) -> bool:
        """Verifica si el texto contiene palabras de la blacklist"""
        if not self.blacklist or not text:
            return False
        return any(word.lower() in text.lower() for word in self.blacklist)

    def _ocr_image(self, img: Image.Image) -> str:
        """Aplica EasyOCR a una imagen PIL y retorna el texto."""
        if not self.reader:
            return ''
        try:
            result = self.reader.readtext(np.array(img), detail=0)
            return '\n'.join(result)
        except Exception as e:
            logging.warning(f"EasyOCR error: {e}")
            return ''

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extrae texto de un PDF página por página con PyMuPDF.
        - Páginas con texto nativo suficiente → extracción directa.
        - Páginas escaneadas (poco texto)     → render a imagen + EasyOCR.
        """
        try:
            doc = fitz.open(pdf_path)
            pages_text = []

            for i, page in enumerate(doc):
                if i >= MAX_PAGES:
                    break
                native = page.get_text() or ''

                if len(native.strip()) >= TEXT_THRESHOLD:
                    pages_text.append(native)
                else:
                    # Página escaneada: render a 200 DPI y OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = self._ocr_image(img)
                    pages_text.append(ocr_text)

            doc.close()
            combined = '\n\n'.join(t for t in pages_text if t.strip())
            return combined if combined.strip() else None

        except Exception as e:
            logging.error(f"Error reading PDF {pdf_path}: {e}")
            return None

    def _extract_text_with_ocr(self, image_path: str) -> Optional[str]:
        """Extrae texto de imágenes usando EasyOCR"""
        try:
            img = Image.open(image_path)
            text = self._ocr_image(img)
            return text if text.strip() else None
        except Exception as e:
            logging.error(f"OCR error for {image_path}: {e}")
            return None

    def _extract_text_from_docx(self, docx_path: str) -> Optional[str]:
        """Extrae texto de archivos DOCX (texto nativo sin OCR)"""
        try:
            doc = Document(docx_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text if text.strip() else None
        except Exception as e:
            logging.error(f"Error extracting text from DOCX {docx_path}: {e}")
            return None

    def _extract_text_from_html(self, html_path: str) -> Optional[str]:
        """Extrae texto de archivos HTML"""
        try:
            with open(html_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator='\n')
            return text if text.strip() else None
        except Exception as e:
            logging.error(f"Error extracting text from HTML {html_path}: {e}")
            return None

    def _extract_text_from_xlsx(self, xlsx_path: str) -> Optional[str]:
        """Extrae texto de archivos Excel"""
        try:
            workbook = load_workbook(xlsx_path, data_only=True)
            text = ""
            for sheet in workbook.sheetnames:
                worksheet = workbook[sheet]
                for row in worksheet.iter_rows(values_only=True):
                    row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                    text += row_text + "\n"
            return text if text.strip() else None
        except Exception as e:
            logging.error(f"Error extracting text from XLSX {xlsx_path}: {e}")
            return None

    def _extract_text_from_file(self, file_path: str) -> Optional[str]:
        """Extrae texto según el tipo de archivo"""
        ext = file_path.lower()

        if ext.endswith('.pdf'):
            return self._extract_text_from_pdf(file_path)
        elif ext.endswith(('.jpg', '.jpeg', '.png')):
            return self._extract_text_with_ocr(file_path)
        elif ext.endswith('.docx'):
            return self._extract_text_from_docx(file_path)
        elif ext.endswith('.html'):
            return self._extract_text_from_html(file_path)
        elif ext.endswith('.xlsx'):
            return self._extract_text_from_xlsx(file_path)

        return None

    def _save_text_file(self, txt_path: str, codigo_cotizacion: str,
                       rut_proveedor: str, original_filename: str, content: str):
        """Guarda el contenido extraído en un archivo de texto"""
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"Codigo Cotizacion: {codigo_cotizacion}\n")
            txt_file.write(f"Rut Proveedor: {rut_proveedor}\n")
            txt_file.write(f"Nombre Original: {original_filename}\n\n")
            txt_file.write("Contenido:\n")
            txt_file.write(content)
        logging.info(f"Text file saved: {txt_path}")

    def process_attachments(self) -> dict:
        """
        Procesa todos los archivos en la carpeta de adjuntos.

        Returns:
            dict: Resultado del procesamiento con:
                - success (bool): Si el proceso fue exitoso
                - processed_files (int): Cantidad de archivos procesados
                - skipped_files (int): Cantidad de archivos omitidos
                - output_path (str): Ruta donde se guardaron los archivos procesados
                - error (str, optional): Mensaje de error si falló
        """
        if not os.path.exists(self.attachments_path):
            return {
                "success": False,
                "error": f"Path does not exist: {self.attachments_path}",
                "processed_files": 0,
                "skipped_files": 0,
                "output_path": self.output_path
            }

        processed_count = 0
        skipped_count = 0
        errors = []

        try:
            files_to_process = []

            for file in os.listdir(self.attachments_path):
                file_path = os.path.join(self.attachments_path, file)

                if not file.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.docx', '.html', '.xlsx')):
                    continue

                base_name = os.path.splitext(file)[0]
                normalized_name = self._normalize_filename(base_name)
                txt_path = os.path.join(self.output_path, f"{normalized_name}.txt")

                if os.path.exists(txt_path):
                    logging.info(f"Already processed: {file}")
                    continue

                if file in self.skipped_files:
                    logging.info(f"Previously skipped: {file}")
                    skipped_count += 1
                    continue

                files_to_process.append((file, file_path, txt_path))

            for file, file_path, txt_path in tqdm(files_to_process, desc="Processing attachments"):
                try:
                    text_content = self._extract_text_from_file(file_path)

                    if self.blacklist and text_content and self._contains_blacklisted_word(text_content):
                        logging.info(f"Skipped (blacklist): {file}")
                        self._add_to_skipped_files(file)
                        skipped_count += 1
                        continue

                    if text_content is None or len(text_content.strip()) == 0:
                        logging.warning(f"No text extracted: {file}")
                        self._add_to_skipped_files(file)
                        skipped_count += 1
                        continue

                    # Formato esperado: attachments/{codigo_cotizacion}/{rut_proveedor}/archivo.ext
                    path_parts = self.attachments_path.split(os.sep)
                    if len(path_parts) >= 2:
                        rut_proveedor = path_parts[-1]
                        codigo_cotizacion = path_parts[-2]
                    else:
                        rut_proveedor = "unknown"
                        codigo_cotizacion = "unknown"

                    self._save_text_file(txt_path, codigo_cotizacion, rut_proveedor, file, text_content)
                    processed_count += 1

                    gc.collect()

                except Exception as e:
                    error_msg = f"Error processing {file}: {e}"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    skipped_count += 1

            return {
                "success": True,
                "processed_files": processed_count,
                "skipped_files": skipped_count,
                "output_path": self.output_path,
                "errors": errors if errors else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processed_files": processed_count,
                "skipped_files": skipped_count,
                "output_path": self.output_path
            }


def process_attachments_simple(attachments_path: str, output_path: Optional[str] = None,
                               blacklist: Optional[List[str]] = None, use_gpu: bool = False) -> dict:
    """
    Función simple para procesar adjuntos sin necesidad de gestionar la clase.

    Args:
        attachments_path (str): Ruta a la carpeta con los adjuntos descargados
        output_path (str, optional): Ruta donde guardar archivos procesados
        blacklist (list, optional): Lista de palabras para filtrar
        use_gpu (bool): Ignorado (mantenido por compatibilidad)

    Returns:
        dict: Resultado del procesamiento

    Example:
        >>> result = process_attachments_simple("attachments/12345678/76123456-7")
        >>> if result['success']:
        >>>     print(f"Processed {result['processed_files']} files")
        >>>     print(f"Output: {result['output_path']}")
    """
    processor = AttachmentProcessor(attachments_path, output_path, blacklist, use_gpu)
    return processor.process_attachments()
