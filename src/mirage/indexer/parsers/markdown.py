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
