"""Parser registry — maps file formats to parser classes."""

from dataclasses import dataclass, field


@dataclass
class ParseResult:
    text: str
    metadata: dict = field(default_factory=dict)
    title: str | None = None


class BaseParser:
    """All parsers implement this interface."""

    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        raise NotImplementedError


_REGISTRY: dict[str, type[BaseParser]] = {}


def register(fmt: str):
    """Decorator to register a parser for a file format."""
    def decorator(cls):
        _REGISTRY[fmt] = cls
        return cls
    return decorator


def get_parser(fmt: str) -> BaseParser:
    fmt = fmt.lower().lstrip(".")
    if fmt not in _REGISTRY:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[fmt]()


def supported_formats() -> list[str]:
    return sorted(_REGISTRY.keys())


# Import all parser modules so they register themselves
from bereans.ingestion.parsers import plaintext  # noqa: E402, F401
from bereans.ingestion.parsers import markdown  # noqa: E402, F401
from bereans.ingestion.parsers import pdf  # noqa: E402, F401
from bereans.ingestion.parsers import epub  # noqa: E402, F401
from bereans.ingestion.parsers import docx  # noqa: E402, F401
from bereans.ingestion.parsers import html  # noqa: E402, F401
from bereans.ingestion.parsers import csv_json  # noqa: E402, F401
