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
