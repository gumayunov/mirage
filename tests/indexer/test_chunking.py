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


def test_chunker_splits_long_sentence_by_words():
    chunker = Chunker(chunk_size=20, overlap=5)
    # A single long sentence with no period-space breaks
    text = "word " * 100

    chunks = chunker.chunk_text(text, structure={"chapter": "Test"})

    assert len(chunks) > 1
    for chunk in chunks:
        tokens = chunker._count_tokens(chunk.content)
        assert tokens <= chunker.chunk_size, (
            f"Chunk has {tokens} tokens, exceeds limit {chunker.chunk_size}"
        )


def test_chunk_positions_are_sequential():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = "Word " * 100

    chunks = chunker.chunk_text(text, structure={})

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))
