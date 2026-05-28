from pathlib import Path
import re
import fitz
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
from rosatom_rag.config import RAW_DIR, EXTRACTED_TEXT_DIR
from rosatom_rag.utils import print_header


def normalize_block_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rect_intersects_any(rect: fitz.Rect, rects: list[fitz.Rect]) -> bool:
    return any(rect.intersects(other) for other in rects)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    page_texts = []

    try:
        for page_num, page in enumerate(doc, start=1):
            elements = []

            table_rects = []
            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []

            for table_idx, table in enumerate(tables, start=1):
                table_rect = fitz.Rect(table.bbox)
                table_rects.append(table_rect)

                table_md = table.to_markdown().strip()
                if table_md:
                    elements.append({
                        "y0": table_rect.y0,
                        "x0": table_rect.x0,
                        "kind": "table",
                        "text": f"[TABLE {table_idx}]\n{table_md}",
                    })

            blocks = page.get_text("blocks", sort=True)

            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                rect = fitz.Rect(x0, y0, x1, y1)

                if rect_intersects_any(rect, table_rects):
                    continue

                text = normalize_block_text(text)
                if not text:
                    continue

                elements.append({
                    "y0": y0,
                    "x0": x0,
                    "kind": "text",
                    "text": text,
                })

            elements.sort(key=lambda item: (round(item["y0"], 1), round(item["x0"], 1)))

            page_parts = [f"--- PAGE {page_num} ---"]

            for item in elements:
                if item["kind"] == "text":
                    page_parts.append(item["text"])
                else:
                    page_parts.append("[TABLE_START]")
                    page_parts.append(item["text"])
                    page_parts.append("[TABLE_END]")

            page_texts.append("\n\n".join(page_parts))

    finally:
        doc.close()

    return "\n\n".join(page_texts)


def process_one_pdf(pdf_path: Path, output_dir: Path) -> Path:
    text = extract_text_from_pdf(pdf_path)
    output_path = output_dir / f"{pdf_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main():
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(RAW_DIR.rglob("*.pdf")) + list(RAW_DIR.rglob("*.PDF")))

    print_header("PDF -> TXT (blocks + tables)")
    print("RAW_DIR:", RAW_DIR)
    print("EXTRACTED_TEXT_DIR:", EXTRACTED_TEXT_DIR)
    print("Найдено PDF:", len(pdf_files))

    for pdf_path in pdf_files:
        try:
            output_path = process_one_pdf(pdf_path, EXTRACTED_TEXT_DIR)
            text_len = len(output_path.read_text(encoding="utf-8"))
            print(f"[OK] {pdf_path.name} -> {output_path.name} | chars={text_len}")
        except Exception as e:
            print(f"[ERROR] {pdf_path.name}: {e}")


if __name__ == "__main__":
    main()