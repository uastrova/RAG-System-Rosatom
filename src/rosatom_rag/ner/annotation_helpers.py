from typing import List, Dict


def show_text_with_indices(text: str):
    print(text)
    print()
    for i in range(0, len(text), 10):
        chunk = text[i:i+10]
        index_line = "".join(str((i + j) % 10) for j in range(len(chunk)))
        print(f"{i:04d}: {chunk}")
        print(f"      {index_line}")


def find_all_occurrences(text: str, substring: str) -> List[Dict]:
    results = []
    start = 0

    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break

        results.append({
            "text": substring,
            "start": idx,
            "end": idx + len(substring),
        })
        start = idx + 1

    return results


def make_entity(text: str, substring: str, label: str, occurrence_index: int = 0) -> Dict:
    matches = find_all_occurrences(text, substring)

    if not matches:
        raise ValueError(f"Подстрока не найдена: {substring}")

    if occurrence_index >= len(matches):
        raise ValueError(
            f"Для подстроки '{substring}' найдено только {len(matches)} вхождений, "
            f"а запрошено occurrence_index={occurrence_index}"
        )

    match = matches[occurrence_index]
    return {
        "text": match["text"],
        "label": label,
        "start": match["start"],
        "end": match["end"],
    }