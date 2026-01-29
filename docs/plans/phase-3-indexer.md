# miRAGe Phase 3: Indexer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать систему индексации документов: парсеры для разных форматов, chunking, worker.

**Prerequisite:** Phase 1 (Foundation) завершена. Существуют: `src/mirage/shared/config.py`, `src/mirage/shared/db.py`, `src/mirage/shared/embedding.py`

**Deliverable:** Рабочий indexer с поддержкой PDF, EPUB, Markdown. Все тесты проходят.

---

## Task 4.1: Markdown Parser

**Files:**
- Create: `src/mirage/indexer/parsers/__init__.py`
- Create: `src/mirage/indexer/parsers/markdown.py`
- Create: `tests/indexer/__init__.py`
- Create: `tests/indexer/parsers/__init__.py`
- Create: `tests/indexer/parsers/test_markdown.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_markdown.py`:
```python
from mirage.indexer.parsers.markdown import MarkdownParser


def test_parse_markdown_with_headings():
    content = """# Book Title

## Chapter 1

This is the first chapter content.
It has multiple paragraphs.

Second paragraph here.

## Chapter 2

### Section 2.1

Content of section 2.1.
"""
    parser = MarkdownParser()
    result = parser.parse(content)

    assert result["title"] == "Book Title"
    assert len(result["sections"]) > 0


def test_parse_markdown_extracts_structure():
    content = """# My Book

## Introduction

Welcome to the book.

## Main Content

### Part 1

First part content.
"""
    parser = MarkdownParser()
    result = parser.parse(content)

    sections = result["sections"]
    assert any(s["heading"] == "Introduction" for s in sections)
    assert any(s["heading"] == "Part 1" for s in sections)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_markdown.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/markdown.py`:
```python
import re
from dataclasses import dataclass


@dataclass
class Section:
    heading: str
    level: int
    content: str
    parent_headings: list[str]


class MarkdownParser:
    def parse(self, content: str) -> dict:
        lines = content.split("\n")
        title = ""
        sections: list[dict] = []
        current_section: dict | None = None
        heading_stack: list[tuple[int, str]] = []
        content_lines: list[str] = []

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section
                if current_section is not None:
                    current_section["content"] = "\n".join(content_lines).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                if level == 1 and not title:
                    title = heading_text

                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                parent_headings = [h[1] for h in heading_stack]
                heading_stack.append((level, heading_text))

                current_section = {
                    "heading": heading_text,
                    "level": level,
                    "parent_headings": parent_headings,
                }
                content_lines = []
            else:
                content_lines.append(line)

        # Save last section
        if current_section is not None:
            current_section["content"] = "\n".join(content_lines).strip()
            if current_section["content"]:
                sections.append(current_section)

        return {
            "title": title,
            "sections": sections,
        }
```

Create `src/mirage/indexer/parsers/__init__.py`: empty file.
Create `tests/indexer/__init__.py`: empty file.
Create `tests/indexer/parsers/__init__.py`: empty file.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_markdown.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add markdown parser"
```

---

## Task 4.2: PDF Parser

**Files:**
- Create: `src/mirage/indexer/parsers/pdf.py`
- Create: `tests/indexer/parsers/test_pdf.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_pdf.py`:
```python
import pytest
from pathlib import Path
from mirage.indexer.parsers.pdf import PDFParser


@pytest.fixture
def sample_pdf(tmp_path):
    # Create a minimal valid PDF for testing
    # This is a very basic PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(pdf_content)
    return pdf_path


def test_pdf_parser_extracts_text(sample_pdf):
    parser = PDFParser()
    result = parser.parse(str(sample_pdf))

    assert "pages" in result
    assert isinstance(result["pages"], list)


def test_pdf_parser_handles_missing_file():
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.pdf")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_pdf.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/pdf.py`:
```python
from pathlib import Path

import fitz  # PyMuPDF


class PDFParser:
    def parse(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(file_path)

        # Try to get TOC
        toc = doc.get_toc()

        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "content": text.strip(),
            })

        # Extract title from metadata or first heading
        title = doc.metadata.get("title", "")
        if not title and toc:
            title = toc[0][1]  # First TOC entry

        doc.close()

        return {
            "title": title,
            "toc": [{"level": t[0], "title": t[1], "page": t[2]} for t in toc],
            "pages": pages,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_pdf.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add PDF parser"
```

---

## Task 4.3: EPUB Parser

**Files:**
- Create: `src/mirage/indexer/parsers/epub.py`
- Create: `tests/indexer/parsers/test_epub.py`

**Step 1: Write the failing test**

`tests/indexer/parsers/test_epub.py`:
```python
import pytest
from mirage.indexer.parsers.epub import EPUBParser


def test_epub_parser_handles_missing_file():
    parser = EPUBParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.epub")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/parsers/test_epub.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/parsers/epub.py`:
```python
import re
from pathlib import Path

import ebooklib
from ebooklib import epub


class EPUBParser:
    def parse(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")

        book = epub.read_epub(file_path)

        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else ""

        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="ignore")
                # Strip HTML tags for plain text
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"\s+", " ", text).strip()

                if text:
                    chapters.append({
                        "id": item.get_id(),
                        "name": item.get_name(),
                        "content": text,
                    })

        # Get TOC
        toc = []
        for nav_item in book.toc:
            if isinstance(nav_item, epub.Link):
                toc.append({
                    "title": nav_item.title,
                    "href": nav_item.href,
                })
            elif isinstance(nav_item, tuple):
                section, links = nav_item
                toc.append({
                    "title": section.title if hasattr(section, "title") else str(section),
                    "children": [{"title": l.title, "href": l.href} for l in links if isinstance(l, epub.Link)],
                })

        return {
            "title": title,
            "toc": toc,
            "chapters": chapters,
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/parsers/test_epub.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add EPUB parser"
```

---

## Task 4.4: Chunking Module

**Files:**
- Create: `src/mirage/indexer/chunking.py`
- Create: `tests/indexer/test_chunking.py`

**Step 1: Write the failing test**

`tests/indexer/test_chunking.py`:
```python
from mirage.indexer.chunking import Chunker, Chunk


def test_chunker_splits_long_text():
    chunker = Chunker(chunk_size=100, overlap=20)
    text = "This is a test. " * 50  # Long text

    chunks = chunker.chunk_text(text, structure={"chapter": "Test"})

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.structure["chapter"] == "Test" for c in chunks)


def test_chunker_preserves_short_text():
    chunker = Chunker(chunk_size=1000, overlap=100)
    text = "Short text."

    chunks = chunker.chunk_text(text, structure={})

    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunker_handles_paragraphs():
    chunker = Chunker(chunk_size=100, overlap=20)
    text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""

    chunks = chunker.chunk_text(text, structure={})

    assert len(chunks) >= 1
    # Check that chunks maintain paragraph boundaries where possible
    for chunk in chunks:
        assert chunk.content.strip()


def test_chunk_positions_are_sequential():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = "Word " * 100

    chunks = chunker.chunk_text(text, structure={})

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_chunking.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/chunking.py`:
```python
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_chunking.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add semantic chunking module"
```

---

## Task 4.5: Indexer Worker

**Files:**
- Create: `src/mirage/indexer/worker.py`
- Create: `tests/indexer/test_worker.py`

**Step 1: Write the failing test**

`tests/indexer/test_worker.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mirage.indexer.worker import IndexerWorker
from mirage.shared.config import Settings


@pytest.fixture
def settings():
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        api_key="test-key",
        ollama_url="http://localhost:11434",
        documents_path="/tmp/docs",
    )


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    client.get_embeddings = AsyncMock(return_value=[[0.1] * 1024])
    return client


def test_worker_initialization(settings):
    worker = IndexerWorker(settings)
    assert worker.settings == settings


@pytest.mark.asyncio
async def test_worker_process_markdown(settings, mock_embedding_client, tmp_path):
    # Create test file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nContent here.")

    settings.documents_path = str(tmp_path)
    worker = IndexerWorker(settings)
    worker.embedding_client = mock_embedding_client

    chunks = await worker._process_file(str(md_file), "markdown")

    assert len(chunks) >= 1
    assert chunks[0]["content"]
    assert "embedding" in chunks[0]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/indexer/test_worker.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/mirage/indexer/worker.py`:
```python
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mirage.indexer.chunking import Chunker
from mirage.indexer.parsers.epub import EPUBParser
from mirage.indexer.parsers.markdown import MarkdownParser
from mirage.indexer.parsers.pdf import PDFParser
from mirage.shared.config import Settings
from mirage.shared.db import ChunkTable, DocumentTable, IndexingTaskTable, get_engine
from mirage.shared.embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class IndexerWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedding_client = OllamaEmbedding(
            settings.ollama_url,
            settings.ollama_model,
        )
        self.parsers = {
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "epub": EPUBParser(),
        }

    async def _process_file(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
        parser = self.parsers.get(file_type)
        if not parser:
            raise ValueError(f"Unsupported file type: {file_type}")

        if file_type == "markdown":
            content = Path(file_path).read_text()
            parsed = parser.parse(content)

            all_chunks = []
            for section in parsed["sections"]:
                structure = {
                    "title": parsed["title"],
                    "heading": section["heading"],
                    "level": section["level"],
                    "parent_headings": section["parent_headings"],
                }
                chunks = self.chunker.chunk_text(section["content"], structure)
                all_chunks.extend(chunks)

        elif file_type == "pdf":
            parsed = parser.parse(file_path)

            all_chunks = []
            for page in parsed["pages"]:
                if page["content"]:
                    structure = {
                        "title": parsed["title"],
                        "page": page["page_number"],
                    }
                    chunks = self.chunker.chunk_text(page["content"], structure)
                    all_chunks.extend(chunks)

        elif file_type == "epub":
            parsed = parser.parse(file_path)

            all_chunks = []
            for chapter in parsed["chapters"]:
                if chapter["content"]:
                    structure = {
                        "title": parsed["title"],
                        "chapter": chapter["name"],
                    }
                    chunks = self.chunker.chunk_text(chapter["content"], structure)
                    all_chunks.extend(chunks)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Get embeddings
        if all_chunks:
            texts = [c.content for c in all_chunks]
            embeddings = await self.embedding_client.get_embeddings(texts)

            return [
                {
                    "content": chunk.content,
                    "position": chunk.position,
                    "structure": chunk.structure,
                    "embedding": emb,
                }
                for chunk, emb in zip(all_chunks, embeddings)
            ]

        return []

    async def process_task(self, session: AsyncSession, task: IndexingTaskTable) -> None:
        doc_result = await session.execute(
            select(DocumentTable).where(DocumentTable.id == task.document_id)
        )
        doc = doc_result.scalar_one_or_none()

        if not doc:
            logger.error(f"Document not found: {task.document_id}")
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            return

        try:
            # Update statuses
            task.status = "processing"
            task.started_at = datetime.utcnow()
            doc.status = "indexing"
            await session.commit()

            # Delete existing chunks if reindexing
            if task.task_type == "reindex":
                await session.execute(
                    ChunkTable.__table__.delete().where(
                        ChunkTable.document_id == doc.id
                    )
                )

            # Process file
            chunks_data = await self._process_file(doc.original_path, doc.file_type)

            # Save chunks
            for chunk_data in chunks_data:
                chunk = ChunkTable(
                    document_id=doc.id,
                    content=chunk_data["content"],
                    embedding_json=chunk_data["embedding"],
                    position=chunk_data["position"],
                    structure_json=chunk_data["structure"],
                )
                session.add(chunk)

            # Update statuses
            doc.status = "ready"
            doc.indexed_at = datetime.utcnow()
            task.status = "done"
            task.completed_at = datetime.utcnow()

            await session.commit()
            logger.info(f"Indexed document {doc.filename}: {len(chunks_data)} chunks")

        except Exception as e:
            logger.error(f"Failed to index {doc.filename}: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            await session.commit()

    async def run(self) -> None:
        engine = get_engine(self.settings.database_url)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        logger.info("Indexer worker started")

        while True:
            async with async_session() as session:
                # Get pending task
                result = await session.execute(
                    select(IndexingTaskTable)
                    .where(IndexingTaskTable.status == "pending")
                    .order_by(IndexingTaskTable.created_at)
                    .limit(1)
                )
                task = result.scalar_one_or_none()

                if task:
                    await self.process_task(session, task)
                else:
                    await asyncio.sleep(5)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/indexer/test_worker.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add indexer worker"
```

---

## Verification

После завершения всех задач:

```bash
uv run pytest tests/indexer/ -v
```

Все тесты должны проходить. Indexer готов. Можно переходить к Phase 4 (CLI + Integration).
