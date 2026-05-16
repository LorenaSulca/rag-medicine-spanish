import argparse
import json
import os
import re


SECTION_HEADER_PATTERN = re.compile(
    r"(?m)^(?P<num>[1-6])\.\s+(?P<title>.+)$"
)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_sections(text: str) -> list[dict]:
    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    sections = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_num = match.group("num")
        section_title = match.group("title").strip()
        section_text = text[start:end].strip()

        sections.append({
            "section_id": f"section_{int(section_num):02d}",
            "section_number": int(section_num),
            "section_name": f"{section_num}. {section_title}",
            "start": start,
            "end": end,
            "section_text": section_text,
        })

    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = load_text(args.input)
    sections = extract_sections(text)

    if not sections:
        raise ValueError("No se detectaron secciones numeradas tipo '1. ...'.")

    save_json(args.output, {
        "document_id": args.document_id,
        "sections": sections,
    })

    print(f"Secciones extraídas: {len(sections)}")
    for s in sections:
        print("-", s["section_name"])


if __name__ == "__main__":
    main()