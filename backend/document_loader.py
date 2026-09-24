import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from cleaner import clean_text

DATA_DIR = Path(__file__).parent.parent / "data"

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


def _read_md_or_txt(file_path):
    return file_path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(file_path):
    reader = PdfReader(str(file_path))
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)
    return "\n\n".join(text_parts)


def _read_docx(file_path):
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_documents():
    """
    Walks through data/v1, data/v2, data/v3 and loads every supported
    file (.md, .txt, .pdf, .docx). Skips unsupported or broken files
    instead of crashing, and reports what happened so nothing silently
    disappears from the corpus.
    """
    documents = []
    skipped = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for version_folder in sorted(DATA_DIR.iterdir()):
        if not version_folder.is_dir():
            continue

        version = version_folder.name

        for file_path in sorted(version_folder.iterdir()):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append((file_path.name, "unsupported file type"))
                continue

            try:
                if ext in (".md", ".txt"):
                    text = _read_md_or_txt(file_path)
                elif ext == ".pdf":
                    text = _read_pdf(file_path)
                elif ext == ".docx":
                    text = _read_docx(file_path)
                else:
                    continue

                text = clean_text(text)

                if not text:
                    skipped.append((file_path.name, "empty after extraction"))
                    continue

                documents.append({
                    "text": text,
                    "source_file": file_path.name,
                    "version": version,
                })

            except Exception as e:
                skipped.append((file_path.name, f"failed to read: {e}"))

    return documents, skipped


if __name__ == "__main__":
    docs, skipped = load_documents()

    print(f"Loaded {len(docs)} documents:\n")
    for doc in docs:
        preview = doc["text"][:60].replace("\n", " ")
        print(f"  [{doc['version']}] {doc['source_file']}  ->  \"{preview}...\"")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")