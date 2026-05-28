from pathlib import Path
import json
import re

from rosatom_rag.config import EXTRACTED_TEXT_DIR, CHUNKS_DIR
from rosatom_rag.utils import print_header


PAGE_PATTERN = re.compile(r"--- PAGE (\d+) ---\n", re.MULTILINE)
TABLE_BLOCK_PATTERN = re.compile(r"\[TABLE_START\].*?\[TABLE_END\]", re.DOTALL)


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_pages(full_text: str):
    matches = list(PAGE_PATTERN.finditer(full_text))
    pages = []

    if not matches:
        cleaned = clean_text(full_text)
        if cleaned:
            pages.append({
                "page_num": 1,
                "text": cleaned,
            })
        return pages

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        page_text = clean_text(full_text[start:end])
        if page_text:
            pages.append({
                "page_num": page_num,
                "text": page_text,
            })

    return pages


def split_paragraphs(text: str):
    text = clean_text(text)
    if not text:
        return []

    parts = re.split(r"\n\s*\n", text)
    return [clean_text(p) for p in parts if clean_text(p)]


def is_table_title(text: str) -> bool:
    text = normalize_spaces(text)
    return bool(re.match(r"^(Таблица|TABLE)\s+[A-Za-zА-Яа-я0-9\.\-]+$", text, re.IGNORECASE))


def is_short_caption(text: str) -> bool:
    text = normalize_spaces(text)
    if not text:
        return False
    if "[TABLE_" in text:
        return False
    return len(text) <= 300


def extract_semantic_blocks(page_text: str):
    text = clean_text(page_text)
    blocks = []
    pos = 0

    for match in TABLE_BLOCK_PATTERN.finditer(text):
        before = text[pos:match.start()]
        paragraphs = split_paragraphs(before)

        table_prefix = []
        if len(paragraphs) >= 2 and is_table_title(paragraphs[-2]) and is_short_caption(paragraphs[-1]):
            table_prefix = [paragraphs[-2], paragraphs[-1]]
            paragraphs = paragraphs[:-2]
        elif len(paragraphs) >= 1 and is_table_title(paragraphs[-1]):
            table_prefix = [paragraphs[-1]]
            paragraphs = paragraphs[:-1]

        for p in paragraphs:
            blocks.append({
                "type": "text",
                "text": p,
            })

        table_text = match.group(0).strip()
        if table_prefix:
            table_text = "\n\n".join(table_prefix + [table_text])

        blocks.append({
            "type": "table",
            "text": table_text,
        })

        pos = match.end()

    tail = text[pos:]
    for p in split_paragraphs(tail):
        blocks.append({
            "type": "text",
            "text": p,
        })

    return blocks


def chunk_plain_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = clean_text(text)
    chunks = []

    if not text:
        return chunks

    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            split_pos = text.rfind(" ", start, end)
            if split_pos != -1 and split_pos > start + chunk_size // 2:
                end = split_pos

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "char_start": start,
                "char_end": end,
                "chunk_type": "text",
            })

        if end >= text_len:
            break

        start = max(0, end - overlap)

    return chunks


def split_large_table_block(table_text: str, chunk_size: int = 1200, overlap_lines: int = 4):
    table_text = clean_text(table_text)

    if len(table_text) <= chunk_size:
        return [{
            "text": table_text,
            "char_start": 0,
            "char_end": len(table_text),
            "chunk_type": "table",
        }]

    lines = [line for line in table_text.splitlines() if line.strip()]

    prefix_lines = []
    table_lines = []
    inside_table = False

    for line in lines:
        if not inside_table and line.strip() != "[TABLE_START]":
            prefix_lines.append(line)
        else:
            inside_table = True
            table_lines.append(line)

    prefix_text = "\n".join(prefix_lines).strip()
    available = chunk_size - len(prefix_text) - 2
    available = max(available, 300)

    chunks = []
    start = 0

    while start < len(table_lines):
        current_lines = []
        current_len = 0
        i = start

        while i < len(table_lines):
            line = table_lines[i]
            added_len = len(line) + 1

            if current_lines and current_len + added_len > available:
                break

            current_lines.append(line)
            current_len += added_len
            i += 1

        chunk_parts = []
        if prefix_text:
            chunk_parts.append(prefix_text)
        chunk_parts.append("\n".join(current_lines))

        chunk_text = "\n\n".join(chunk_parts).strip()

        chunks.append({
            "text": chunk_text,
            "char_start": 0,
            "char_end": len(chunk_text),
            "chunk_type": "table",
        })

        if i >= len(table_lines):
            break

        start = max(start + 1, i - overlap_lines)

    return chunks


def chunk_blocks(blocks, chunk_size: int = 1200, overlap: int = 200):
    result = []
    text_buffer = []

    def flush_text_buffer():
        nonlocal text_buffer, result
        if not text_buffer:
            return

        joined = "\n\n".join(text_buffer).strip()
        if joined:
            result.extend(chunk_plain_text(joined, chunk_size=chunk_size, overlap=overlap))
        text_buffer = []

    for block in blocks:
        if block["type"] == "table":
            flush_text_buffer()
            result.extend(split_large_table_block(block["text"], chunk_size=chunk_size))
        else:
            text_buffer.append(block["text"])

    flush_text_buffer()
    return result


def build_chunks_for_all_txt(txt_dir: Path):
    txt_files = sorted(txt_dir.glob("*.txt"))
    all_chunks = []

    for txt_path in txt_files:
        full_text = txt_path.read_text(encoding="utf-8")
        pages = split_text_into_pages(full_text)

        for page in pages:
            blocks = extract_semantic_blocks(page["text"])
            page_chunks = chunk_blocks(blocks, chunk_size=1200, overlap=200)

            for chunk_idx, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "chunk_id": f"{txt_path.stem}_p{page['page_num']}_c{chunk_idx}",
                    "source_file": txt_path.name,
                    "source_stem": txt_path.stem,
                    "page_num": page["page_num"],
                    "chunk_index_on_page": chunk_idx,
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "chunk_type": chunk["chunk_type"],
                    "text": chunk["text"],
                })

    return all_chunks


def save_chunks_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    output_path = CHUNKS_DIR / "ntd_chunks.jsonl"
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    print_header("TXT -> CHUNKS")
    print("EXTRACTED_TEXT_DIR:", EXTRACTED_TEXT_DIR)
    print("OUTPUT_PATH:", output_path)

    all_chunks = build_chunks_for_all_txt(EXTRACTED_TEXT_DIR)
    save_chunks_jsonl(all_chunks, output_path)

    print("Всего chunks:", len(all_chunks))
    print("Сохранено в:", output_path)

    print("\nПервые 3 chunks:")
    for item in all_chunks[:3]:
        print("-" * 80)
        print("chunk_id:", item["chunk_id"])
        print("source_file:", item["source_file"])
        print("page_num:", item["page_num"])
        print("chunk_type:", item["chunk_type"])
        print(item["text"][:700])


if __name__ == "__main__":
    main()