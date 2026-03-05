"""Smart text chunker that respects sentence and paragraph boundaries."""

import re
import tiktoken


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [{
                "content": text,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(text),
                "metadata": {},
            }]

        # Find all possible split points (paragraph and sentence boundaries)
        split_points = self._find_split_points(text)
        return self._build_chunks(text, split_points)

    def _find_split_points(self, text: str) -> list[int]:
        """Find character positions where splits can occur (paragraph/sentence boundaries)."""
        points = set()

        # Paragraph boundaries (double newlines)
        for m in re.finditer(r"\n\s*\n", text):
            points.add(m.end())

        # Sentence boundaries
        for m in re.finditer(r"(?<=[.!?])\s+", text):
            points.add(m.end())

        # Sort and return
        result = sorted(points)
        return result

    def _build_chunks(self, text: str, split_points: list[int]) -> list[dict]:
        """Build chunks with overlap using split points."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk: take everything remaining
                chunks.append({
                    "content": text[start:],
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": len(text),
                    "metadata": {},
                })
                break

            # Find the best split point at or before 'end'
            best_split = end
            for sp in reversed(split_points):
                if sp <= end and sp > start:
                    best_split = sp
                    break

            chunk_content = text[start:best_split].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": best_split,
                    "metadata": {},
                })

            # Move start forward with overlap
            start = best_split - self.chunk_overlap
            if start < 0:
                start = 0
            # Make sure we actually advance to avoid infinite loop
            if start <= chunks[-1]["start_char"] if chunks else False:
                start = best_split

        return chunks

    def _split_into_segments(self, text: str) -> list[str]:
        """Split text into segments respecting paragraph and sentence boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                segments.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                segments.extend(s.strip() for s in sentences if s.strip())
        return segments

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))
