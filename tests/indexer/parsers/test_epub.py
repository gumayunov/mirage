import pytest
from mirage.indexer.parsers.epub import EPUBParser


def test_epub_parser_handles_missing_file():
    parser = EPUBParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.epub")
