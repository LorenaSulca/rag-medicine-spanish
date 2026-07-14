import pdfplumber
import unicodedata
import re
import sys
import os


def safe_print(message: str):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("utf-8", errors="replace").decode("utf-8"))


def fix_mojibake(text: str) -> str:
    """
    Intenta reparar mojibake típico, pero sin romper textos correctos.
    """
    if not text:
        return ""

    try:
        fixed = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")

        # Solo aceptar si parece mejorar mojibake típico.
        if any(bad in text for bad in ["Ã", "Â", "â€", "�"]):
            return fixed

        return text
    except Exception:
        return text


def remove_unsupported_chars(text: str) -> str:
    """
    Elimina caracteres problemáticos para MEDSPANER / Windows / regex.
    Conserva letras acentuadas normales.
    """
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u00a0": " ",
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2022": "-",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Eliminar controles invisibles.
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text


def normalize_text(text: str) -> str:
    """
    Normalización segura para MEDSPANER.
    """
    if not text:
        return ""

    text = fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = remove_unsupported_chars(text)

    # Normalizar saltos de línea.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Evitar palabras pegadas por cortes raros.
    text = re.sub(r"-\n(?=\w)", "", text)

    # Espacios.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Quitar líneas casi vacías con basura.
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue

        # Evitar líneas compuestas solo por símbolos raros.
        if re.fullmatch(r"[-_=·•.\s]+", line):
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_page_text(page) -> str:
    """
    Extrae texto de una página con fallback.
    """
    try:
        text = page.extract_text(
            x_tolerance=1,
            y_tolerance=3,
            layout=False,
        )
        return text or ""
    except Exception:
        return ""


def extract_clean_text(pdf_path, txt_path=None):
    full_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = extract_page_text(page)
            page_text = normalize_text(page_text)

            if page_text:
                full_text_parts.append(page_text)
            else:
                safe_print(f"Advertencia: página {i} sin texto extraíble.")

    full_text = "\n\n".join(full_text_parts)
    clean_text = normalize_text(full_text)

    if not clean_text:
        raise ValueError("No se pudo extraer texto del PDF. Puede ser escaneado o contener imágenes.")

    if txt_path:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "w", encoding="utf-8", errors="replace", newline="\n") as f:
            f.write(clean_text)

    return clean_text


if __name__ == "__main__":
    # Fuerza salida UTF-8 en Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) < 3:
        safe_print("Uso: python pdf_text_extractor.py nombre_pdf.pdf output.txt")
        sys.exit(1)

    pdf_name = sys.argv[1]
    output_txt = sys.argv[2]

    base_dir = os.path.join("..")
    pdf_path = os.path.join(base_dir, "prospects", pdf_name)
    txt_path = os.path.join(base_dir, "data", output_txt)

    if not os.path.isfile(pdf_path):
        safe_print(f"El archivo PDF no existe en: {pdf_path}")
        sys.exit(1)

    try:
        extract_clean_text(pdf_path, txt_path=txt_path)
        safe_print(f"Extracción guardada en: {output_txt}")
    except Exception as exc:
        safe_print(f"Error extrayendo texto: {repr(exc)}")
        sys.exit(1)