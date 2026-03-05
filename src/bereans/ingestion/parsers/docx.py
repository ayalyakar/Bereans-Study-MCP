"""DOCX parser using python-docx."""
from pathlib import Path
from docx import Document
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("docx")
class DOCXParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        doc = Document(source)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        full_text = "\n\n".join(paragraphs)
        title = doc.core_properties.title or Path(source_path or str(source)).stem
        metadata = {}
        if doc.core_properties.author:
            metadata["author"] = doc.core_properties.author
        return ParseResult(text=full_text, metadata=metadata, title=title)
