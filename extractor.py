# extractor.py
import pdfplumber
from typing import Tuple
import io

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF bytes using pdfplumber.
    Returns tuple: (extracted_text, error_message)
    If extraction fails, extracted_text will be '' and error_message will be set.
    """
    try:
        text_chunks = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if len(pdf.pages) == 0:
                return "", "PDF has no pages"
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                    text_chunks.append(page_text)
                except Exception:
                    # some pages may fail; continue
                    continue
        full_text = "\n\n".join(text_chunks).strip()
        if not full_text:
            return "", "No text extracted (might be scanned image PDF)"
        return full_text, ""
    except Exception as e:
        return "", f"Failed to read PDF: {e}"
