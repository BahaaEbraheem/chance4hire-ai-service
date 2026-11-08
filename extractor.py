"""PDF text extraction for the Chance4Hire AI service.

Digital (text-layer) PDFs are extracted with **PyMuPDF**, chosen on evidence:
on a Word-exported Arabic CV it recovered 12/13 skills versus 8/13 for
pdfplumber, because it preserves the *logical* order of Arabic multi-word phrases
instead of the reversed visual order that breaks them. The extracted text is fed
through the same ``normalize()`` step as the analyzer (NFKC + Arabic folding), so
PDF presentation-form glyphs collapse back to canonical letters downstream.

Scanned PDFs (image-only, no text layer) are *detected and flagged* rather than
silently returning garbage — OCR (Tesseract + Arabic) is documented future work.
"""
import fitz  # PyMuPDF


def extract_pdf(data: bytes) -> dict:
    """Extract text from a PDF byte string.

    Returns ``{text, pages, char_count, is_scanned}``. ``is_scanned`` is True when
    a page carries images but yields no text layer — the caller should then route
    the file to OCR (not yet implemented) instead of trusting the empty text.
    """
    text_parts = []
    has_image = False
    pages = 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = doc.page_count
        for page in doc:
            text_parts.append(page.get_text("text"))
            if page.get_images():
                has_image = True

    text = "\n".join(text_parts).strip()
    return {
        "text": text,
        "pages": pages,
        "char_count": len(text),
        "is_scanned": len(text) < 10 and has_image,
    }
