import pdfplumber
import unicodedata
import re
import sys
import os

def fix_mojibake(text):
    """
    Repara problemas comunes de doble codificación UTF-8/Latin-1.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except:
        return text  
    
def normalize_text(text):
    """
    Normalización segura para MEDSPANER.
    """
    text = unicodedata.normalize("NFC", text)
    text = fix_mojibake(text)

    # Eliminar caracteres invisibles
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # Normalización de saltos y espacios
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    text = text.strip()
    return text

def extract_clean_text(pdf_path, txt_path=None):
    """
    Extrae texto de un PDF y lo limpia.
    """
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n\n"

    clean_text = normalize_text(full_text)

    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

    return clean_text

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Uso: python pdf_text_extractor.py nombre_pdf.pdf output.txt")
        sys.exit(1)

    pdf_name = sys.argv[1]
    output_txt = sys.argv[2]

    base_dir = os.path.join("..")
    pdf_path = os.path.join(base_dir, "prospects" ,pdf_name)
    txt_path = os.path.join(base_dir, "data", output_txt)

    if not os.path.isfile(pdf_path):
        print(f"El archivo PDF no existe en: {pdf_path}")
        sys.exit(1)

    extract_clean_text(pdf_path, txt_path=txt_path)

    print(f"Extraccion guardada en: {output_txt}")
