"""Convert markdown files to styled PDFs using fpdf2."""
import re
import sys
from pathlib import Path
from fpdf import FPDF


def _sanitize(text: str) -> str:
    """Replace Unicode chars that may not be in the font with ASCII equivalents."""
    return (text
        .replace("\u2014", "--")   # em dash
        .replace("\u2013", "-")    # en dash
        .replace("\u2018", "'")    # left single quote
        .replace("\u2019", "'")    # right single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2022", "-")  # bullet
        .replace("\u2192", "->")   # right arrow
        .replace("\u2190", "<-")   # left arrow
        .replace("\u2265", ">=")   # >=
        .replace("\u2264", "<=")   # <=
        .replace("\u00b0", "deg")  # degree
        .replace("\u25b6", ">")    # triangle
        .replace("\u25bc", "v")    # down triangle
        .replace("\u2502", "|")    # box drawing
        .replace("\u250c", "+")
        .replace("\u2510", "+")
        .replace("\u2514", "+")
        .replace("\u2518", "+")
        .replace("\u2500", "-")
    )


FONT_DIR = Path("C:/Windows/Fonts")

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Register Unicode TTF fonts
        self.add_font("Sans", "", str(FONT_DIR / "calibri.ttf"), uni=True)
        self.add_font("Sans", "B", str(FONT_DIR / "calibrib.ttf"), uni=True)
        self.add_font("Sans", "I", str(FONT_DIR / "calibrii.ttf"), uni=True)
        self.add_font("Mono", "", str(FONT_DIR / "consola.ttf"), uni=True)
        self.add_font("Mono", "B", str(FONT_DIR / "consolab.ttf"), uni=True)
        self.add_page()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_font("Sans", size=11)
        self._in_code_block = False
        self._in_table = False
        self._table_rows = []

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Sans", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)

    def _flush_table(self):
        if not self._table_rows:
            return
        # Calculate column widths
        n_cols = max(len(r) for r in self._table_rows)
        usable = self.w - 2 * self.l_margin
        col_w = usable / max(n_cols, 1)

        self.set_font("Sans", size=9)
        for ri, row in enumerate(self._table_rows):
            if ri == 0:
                self.set_font("Sans", "B", 9)
                self.set_fill_color(52, 73, 94)
                self.set_text_color(255, 255, 255)
            elif ri == 1 and all(c.strip().replace("-", "").replace(":", "") == "" for c in row):
                continue  # skip separator row
            else:
                self.set_font("Sans", size=9)
                self.set_text_color(0, 0, 0)
                if ri % 2 == 0:
                    self.set_fill_color(236, 240, 241)
                else:
                    self.set_fill_color(255, 255, 255)

            for ci, cell in enumerate(row):
                if ci < n_cols:
                    self.cell(col_w, 6, cell.strip()[:50], border=0, fill=True)
            self.ln()

        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        self.ln(3)
        self._table_rows = []
        self._in_table = False

    def _write_inline(self, text):
        """Write text with basic inline markdown (bold, italic, code)."""
        # Process inline formatting
        parts = re.split(r'(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)', text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font("Sans", "B", self.font_size_pt)
                self.write(5, part[2:-2])
                self.set_font("Sans", "", self.font_size_pt)
            elif part.startswith("`") and part.endswith("`"):
                self.set_font("Mono", "", self.font_size_pt - 1)
                self.set_text_color(192, 57, 43)
                self.write(5, part[1:-1])
                self.set_text_color(0, 0, 0)
                self.set_font("Sans", "", self.font_size_pt)
            elif part.startswith("*") and part.endswith("*"):
                self.set_font("Sans", "I", self.font_size_pt)
                self.write(5, part[1:-1])
                self.set_font("Sans", "", self.font_size_pt)
            else:
                self.write(5, part)

    def process_markdown(self, md_text: str):
        self.alias_nb_pages()
        md_text = _sanitize(md_text)
        lines = md_text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Code block toggle
            if line.strip().startswith("```"):
                if self._in_table:
                    self._flush_table()
                self._in_code_block = not self._in_code_block
                if self._in_code_block:
                    self.ln(2)
                else:
                    self.ln(2)
                i += 1
                continue

            # Inside code block
            if self._in_code_block:
                self.set_font("Mono", size=8)
                self.set_fill_color(44, 62, 80)
                self.set_text_color(236, 240, 241)
                self.cell(0, 5, "  " + line, ln=True, fill=True)
                self.set_text_color(0, 0, 0)
                self.set_fill_color(255, 255, 255)
                self.set_font("Sans", size=11)
                i += 1
                continue

            # Table row
            if "|" in line and line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip().split("|")[1:-1]]
                if cells:
                    self._in_table = True
                    self._table_rows.append(cells)
                    i += 1
                    continue

            # Flush table if we were in one
            if self._in_table:
                self._flush_table()

            # Horizontal rule
            if line.strip() in ("---", "***", "___"):
                self.ln(3)
                y = self.get_y()
                self.set_draw_color(189, 195, 199)
                self.line(self.l_margin, y, self.w - self.r_margin, y)
                self.ln(5)
                i += 1
                continue

            # Headers
            if line.startswith("# "):
                self.ln(8)
                self.set_font("Sans", "B", 22)
                self.set_text_color(41, 128, 185)
                self.multi_cell(0, 10, line[2:].strip())
                self.set_text_color(0, 0, 0)
                self.set_font("Sans", size=11)
                self.ln(3)
                i += 1
                continue
            if line.startswith("## "):
                self.ln(6)
                self.set_font("Sans", "B", 17)
                self.set_text_color(41, 128, 185)
                self.multi_cell(0, 9, line[3:].strip())
                self.set_text_color(0, 0, 0)
                self.set_font("Sans", size=11)
                self.ln(2)
                i += 1
                continue
            if line.startswith("### "):
                self.ln(4)
                self.set_font("Sans", "B", 14)
                self.set_text_color(52, 73, 94)
                self.multi_cell(0, 8, line[4:].strip())
                self.set_text_color(0, 0, 0)
                self.set_font("Sans", size=11)
                self.ln(2)
                i += 1
                continue
            if line.startswith("#### "):
                self.ln(3)
                self.set_font("Sans", "B", 12)
                self.multi_cell(0, 7, line[5:].strip())
                self.set_font("Sans", size=11)
                self.ln(1)
                i += 1
                continue

            # Bullet points
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                indent = len(line) - len(line.lstrip())
                bullet_text = line.strip()[2:]
                x = self.l_margin + indent * 2 + 5
                self.set_x(x)
                self.cell(5, 6, "-")
                self.set_font("Sans", size=11)
                self._write_inline(bullet_text)
                self.ln(6)
                i += 1
                continue

            # Numbered list
            m = re.match(r'^(\s*)(\d+)\.\s+(.*)', line)
            if m:
                indent = len(m.group(1))
                num = m.group(2)
                text = m.group(3)
                x = self.l_margin + indent * 2 + 5
                self.set_x(x)
                self.set_font("Sans", "B", 11)
                self.cell(8, 6, f"{num}.")
                self.set_font("Sans", size=11)
                self._write_inline(text)
                self.ln(6)
                i += 1
                continue

            # Empty line
            if not line.strip():
                self.ln(3)
                i += 1
                continue

            # Regular paragraph
            self.set_font("Sans", size=11)
            self._write_inline(line)
            self.ln(6)
            i += 1

        # Flush remaining table
        if self._in_table:
            self._flush_table()


def convert(md_path: str, pdf_path: str):
    text = Path(md_path).read_text(encoding="utf-8")
    pdf = MarkdownPDF()
    pdf.process_markdown(text)
    pdf.output(pdf_path)
    print(f"  {Path(md_path).name} -> {Path(pdf_path).name}")


if __name__ == "__main__":
    base = Path(__file__).parent
    pdf_dir = base / "pdf"
    pdf_dir.mkdir(exist_ok=True)

    files = [
        (base / "DEVLOG.md", pdf_dir / "DEVLOG.pdf"),
        (base / "DOCUMENTATION.md", pdf_dir / "DOCUMENTATION.pdf"),
        (base.parent / "README.md", pdf_dir / "README.pdf"),
    ]

    for md, pdf in files:
        if md.exists():
            convert(str(md), str(pdf))
        else:
            print(f"  SKIP: {md} not found")
