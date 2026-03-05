"""CSV and JSON parsers."""
import json as json_lib
from pathlib import Path
import pandas as pd
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("csv")
class CSVParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            from io import BytesIO
            df = pd.read_csv(BytesIO(source))
        else:
            df = pd.read_csv(source)
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            rows.append(row_text)
        full_text = "\n".join(rows)
        metadata = {"columns": list(df.columns), "row_count": len(df)}
        title = Path(source_path or str(source)).stem if isinstance(source, str) else "csv_data"
        return ParseResult(text=full_text, metadata=metadata, title=title)


@register("json")
class JSONParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            data = json_lib.loads(source.decode("utf-8"))
        elif Path(source).exists():
            data = json_lib.loads(Path(source).read_text(encoding="utf-8"))
        else:
            data = json_lib.loads(source)
        text = self._flatten(data)
        title = Path(source_path or str(source)).stem if isinstance(source, str) else "json_data"
        return ParseResult(text=text, title=title)

    def _flatten(self, data, prefix: str = "") -> str:
        lines = []
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(self._flatten(value, path))
                else:
                    lines.append(f"{path}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(self._flatten(item, f"{prefix}[{i}]"))
        else:
            lines.append(f"{prefix}: {data}" if prefix else str(data))
        return "\n".join(lines)
