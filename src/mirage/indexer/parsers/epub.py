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
