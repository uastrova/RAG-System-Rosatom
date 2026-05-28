from pathlib import Path
import re

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

from rosatom_rag.config import RAW_DIR, EXTRACTED_TEXT_DIR
from rosatom_rag.utils import print_header


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_block_items(parent):
    if isinstance(parent, DocumentType):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError(f"Unsupported parent type: {type(parent)}")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def iter_row_texts(row):
    cells = []

    for _ in range(row.grid_cols_before):
        cells.append("")

    for cell in row.cells:
        cells.append(normalize_text(cell.text))

    for _ in range(row.grid_cols_after):
        cells.append("")

    return cells


def iter_row_texts(row):
    cells = []

    for _ in range(row.grid_cols_before):
        cells.append("")

    for cell in row.cells:
        cells.append(normalize_text(cell.text))

    for _ in range(row.grid_cols_after):
        cells.append("")

    return cells


def clean_rows(rows):
    cleaned = []
    max_cols = 0

    for row in rows:
        row_cells = iter_row_texts(row)
        if any(cell.strip() for cell in row_cells):
            cleaned.append(row_cells)
            max_cols = max(max_cols, len(row_cells))

    if not cleaned:
        return []

    for row in cleaned:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))

    return cleaned


def is_pseudo_table(rows):
    if not rows:
        return True

    nonempty_counts = []
    for row in rows:
        cnt = sum(1 for cell in row if cell.strip())
        nonempty_counts.append(cnt)

    max_nonempty = max(nonempty_counts) if nonempty_counts else 0

    # если во всех строках по сути заполнена только одна ячейка —
    # это не настоящая таблица, а блочная верстка
    if max_nonempty <= 1:
        return True

    return False


def rows_to_plain_text(rows):
    parts = []

    for row in rows:
        cell_texts = [cell.strip() for cell in row if cell.strip()]
        if cell_texts:
            parts.append("\n".join(cell_texts))

    return "\n\n".join(parts).strip()


def rows_to_markdown(rows):
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]

    lines = []
    lines.append("|" + "|".join(header) + "|")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in body:
        lines.append("|" + "|".join(row) + "|")

    return "\n".join(lines).strip()


def table_to_text(table: Table) -> str:
    rows = clean_rows(table.rows)
    if not rows:
        return ""

    if is_pseudo_table(rows):
        return rows_to_plain_text(rows)

    return rows_to_markdown(rows)

def extract_docx_text(docx_path: Path) -> str:
    doc = Document(docx_path)
    blocks = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = normalize_text(block.text)
            if text:
                blocks.append(text)

        elif isinstance(block, Table):
            text = table_to_text(block)
            if text:
                if text.startswith("|") and "\n|---" in text:
                    blocks.append("[TABLE_START]")
                    blocks.append(text)
                    blocks.append("[TABLE_END]")
            else:
                blocks.append(text)

    return "\n\n".join(blocks).strip()


def main():
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(list(RAW_DIR.rglob("*.docx")) + list(RAW_DIR.rglob("*.DOCX")))

    print_header("EXTRACT DOCX TEXT")
    print("RAW_DIR:", RAW_DIR)
    print("EXTRACTED_TEXT_DIR:", EXTRACTED_TEXT_DIR)
    print("Найдено .docx файлов:", len(docx_files))

    saved = 0
    failed = 0

    for docx_path in docx_files:
        try:
            text = extract_docx_text(docx_path)

            if not text:
                print(f"[WARN] Пустой текст: {docx_path}")
                continue

            out_path = EXTRACTED_TEXT_DIR / f"{docx_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")

            saved += 1
            print(f"[OK] {docx_path.name} -> {out_path.name}")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {docx_path}: {e}")

    print()
    print("Сохранено txt:", saved)
    print("Ошибок:", failed)


if __name__ == "__main__":
    main()