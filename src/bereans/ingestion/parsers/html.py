"""HTML parser using BeautifulSoup."""
from pathlib import Path
from bs4 import BeautifulSoup
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("html")
@register("htm")
class HTMLParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            html = source.decode("utf-8", errors="replace")
        elif Path(source).exists():
            html = Path(source).read_text(encoding="utf-8", errors="replace")
        else:
            html = source
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        text = soup.get_text(separator="\n", strip=True)
        return ParseResult(
            text=text,
            title=title or Path(source_path).stem if source_path else title,
        )
