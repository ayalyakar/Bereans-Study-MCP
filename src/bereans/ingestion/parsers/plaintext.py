"""Plain text parser."""
from pathlib import Path
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("txt")
class PlainTextParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        title = Path(source_path or source).stem if isinstance(source, str) else None
        return ParseResult(text=text, title=title)
