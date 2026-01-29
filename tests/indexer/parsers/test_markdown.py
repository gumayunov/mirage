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
