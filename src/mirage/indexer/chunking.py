from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass
class Chunk:
    content: str
    position: int
    structure: dict[str, Any]


class Chunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_text(self, text: str, structure: dict[str, Any]) -> list[Chunk]:
        if not text.strip():
            return []

        # If text is short enough, return as single chunk
        if self._count_tokens(text) <= self.chunk_size:
            return [Chunk(content=text.strip(), position=0, structure=structure)]

        paragraphs = self._split_into_paragraphs(text)
        chunks: list[Chunk] = []
        current_content: list[str] = []
        current_tokens = 0
        position = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if any
                if current_content:
                    chunks.append(Chunk(
                        content="\n\n".join(current_content),
                        position=position,
                        structure=structure,
                    ))
                    position += 1
                    current_content = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = para.replace(". ", ".|").split("|")
                for sentence in sentences:
                    sent_tokens = self._count_tokens(sentence)
                    if current_tokens + sent_tokens > self.chunk_size and current_content:
                        chunks.append(Chunk(
                            content=" ".join(current_content),
                            position=position,
                            structure=structure,
                        ))
                        position += 1
                        # Keep overlap
                        overlap_content = current_content[-1] if current_content else ""
                        current_content = [overlap_content] if overlap_content else []
                        current_tokens = self._count_tokens(overlap_content) if overlap_content else 0

                    current_content.append(sentence)
                    current_tokens += sent_tokens

            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                chunks.append(Chunk(
                    content="\n\n".join(current_content),
                    position=position,
                    structure=structure,
                ))
                position += 1

                # Keep some overlap
                overlap_text = current_content[-1] if current_content else ""
                overlap_tokens = self._count_tokens(overlap_text)
                if overlap_tokens <= self.overlap:
                    current_content = [overlap_text, para]
                    current_tokens = overlap_tokens + para_tokens
                else:
                    current_content = [para]
                    current_tokens = para_tokens
            else:
                current_content.append(para)
                current_tokens += para_tokens

        # Save remaining content
        if current_content:
            chunks.append(Chunk(
                content="\n\n".join(current_content) if len(current_content) > 1 else current_content[0],
                position=position,
                structure=structure,
            ))

        return chunks
