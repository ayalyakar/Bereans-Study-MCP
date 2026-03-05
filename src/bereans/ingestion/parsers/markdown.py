"""Markdown parser — preserves structure, extracts title from first heading."""
import re
from pathlib import Path
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("md")
class MarkdownParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        title = None
        first_heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if first_heading:
            title = first_heading.group(1).strip()
        return ParseResult(text=text, title=title or Path(source_path or source).stem)
