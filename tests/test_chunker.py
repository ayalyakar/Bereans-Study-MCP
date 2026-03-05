"""Tests for smart text chunker."""

import pytest
from bereans.ingestion.chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=50, chunk_overlap=10)


def test_short_text_single_chunk(chunker):
    text = "This is a short text."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(text)


def test_long_text_multiple_chunks(chunker):
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["content"]) > 0
        assert chunk["start_char"] < chunk["end_char"]


def test_chunks_have_overlap(chunker):
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    if len(chunks) >= 2:
        assert chunks[1]["start_char"] < chunks[0]["end_char"]


def test_chunk_indices_sequential(chunker):
    sentences = [f"Sentence {i}." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_empty_text(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_respects_paragraph_boundaries():
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
