"""Tests for file parsers."""
import json
import pytest
from bereans.ingestion.parsers import get_parser, supported_formats, ParseResult


def test_supported_formats():
    fmts = supported_formats()
    assert "txt" in fmts
    assert "md" in fmts
    assert "pdf" in fmts
    assert "html" in fmts
    assert "csv" in fmts
    assert "json" in fmts


def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        get_parser("xyz")


def test_plaintext_parser(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world, this is a test.", encoding="utf-8")
    parser = get_parser("txt")
    result = parser.parse(str(f), source_path=str(f))
    assert "Hello world" in result.text
    assert result.title == "test"


def test_markdown_parser(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# My Title\n\nSome paragraph text.", encoding="utf-8")
    parser = get_parser("md")
    result = parser.parse(str(f), source_path=str(f))
    assert result.title == "My Title"
    assert "Some paragraph text" in result.text


def test_html_parser(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>Page Title</title></head>"
        "<body><p>Content here</p><script>bad()</script></body></html>",
        encoding="utf-8",
    )
    parser = get_parser("html")
    result = parser.parse(str(f), source_path=str(f))
    assert result.title == "Page Title"
    assert "Content here" in result.text
    assert "bad()" not in result.text


def test_csv_parser(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    parser = get_parser("csv")
    result = parser.parse(str(f), source_path=str(f))
    assert "Alice" in result.text
    assert "Bob" in result.text
    assert result.metadata["row_count"] == 2


def test_json_parser(tmp_path):
    f = tmp_path / "data.json"
    data = {"name": "Genesis", "chapters": 50, "author": "Moses"}
    f.write_text(json.dumps(data), encoding="utf-8")
    parser = get_parser("json")
    result = parser.parse(str(f), source_path=str(f))
    assert "Genesis" in result.text
    assert "Moses" in result.text
