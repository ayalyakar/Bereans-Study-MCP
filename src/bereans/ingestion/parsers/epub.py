"""EPUB parser using ebooklib + BeautifulSoup."""
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("epub")
class EPUBParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        book = epub.read_epub(source, options={"ignore_ncx": True})
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "lxml")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                chapters.append(text)
        full_text = "\n\n".join(chapters)
        title = book.get_metadata("DC", "title")
        title_str = title[0][0] if title else Path(source_path or str(source)).stem
        metadata = {}
        authors = book.get_metadata("DC", "creator")
        if authors:
            metadata["author"] = authors[0][0]
        return ParseResult(text=full_text, metadata=metadata, title=title_str)
