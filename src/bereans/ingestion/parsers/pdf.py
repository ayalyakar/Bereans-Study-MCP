"""PDF parser using PyMuPDF."""
from pathlib import Path
import pymupdf
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("pdf")
class PDFParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            doc = pymupdf.open(stream=source, filetype="pdf")
        else:
            doc = pymupdf.open(source)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        full_text = "\n\n".join(pages)
        metadata = {"page_count": len(doc)}
        title = doc.metadata.get("title") or Path(source_path or str(source)).stem
        doc.close()
        return ParseResult(text=full_text, metadata=metadata, title=title)
