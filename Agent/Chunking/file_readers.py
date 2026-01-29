"""
Ce fichier fournit des fonctions utilitaires pour charger du texte à partir de fichiers
de différents formats (PDF, DOCX, TXT) et le normaliser sous une structure commune.

Chaque fonction retourne une liste de dictionnaires contenant :
- page_number : le numéro de page (ou 1 pour les formats sans pagination)
- text : le contenu textuel extrait

Ce module est utilisé en amont du pipeline (ex. chunking, RAG) pour uniformiser
la lecture des documents sources.
"""

import fitz  # PyMuPDF : bibliothèque pour lire et parser des fichiers PDF
from docx import Document  # python-docx : bibliothèque pour lire les fichiers DOCX
from pathlib import Path  # Gestion des chemins de fichiers de manière portable


def read_pdf(path: Path):
    """
    Lit un fichier PDF et extrait le texte page par page.

    Args:
        path (Path): chemin vers le fichier PDF.

    Returns:
        list[dict]: liste de dictionnaires contenant le numéro de page et le texte extrait.
    """
    pages = []
    # Ouverture du PDF avec PyMuPDF
    with fitz.open(path) as doc:
        # Parcours des pages avec numérotation à partir de 1
        for i, page in enumerate(doc, start=1):
            # Extraction du texte brut de la page
            text = page.get_text("text").strip()
            # On ignore les pages vides
            if text:
                pages.append({"page_number": i, "text": text})
    return pages


def read_docx(path: Path):
    """
    Lit un fichier DOCX et extrait tout le texte.

    Le format DOCX n'ayant pas de notion de page fiable,
    tout le contenu est associé à la page 1.

    Args:
        path (Path): chemin vers le fichier DOCX.

    Returns:
        list[dict]: liste contenant un seul dictionnaire avec le texte extrait.
    """
    doc = Document(path)
    # Concaténation des paragraphes non vides
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page_number": 1, "text": text.strip()}]


def read_txt(path: Path):
    """
    Lit un fichier texte brut (.txt).

    Tout le contenu est associé à la page 1.

    Args:
        path (Path): chemin vers le fichier TXT.

    Returns:
        list[dict]: liste contenant un seul dictionnaire avec le texte extrait.
    """
    return [{
        "page_number": 1,
        # Lecture du fichier en UTF-8 avec tolérance aux erreurs
        "text": path.read_text(encoding="utf-8", errors="ignore").strip()
    }]


def load_text_from_file(path: Path):
    """
    Détecte automatiquement le type de fichier et appelle
    la fonction de lecture appropriée.

    Formats supportés :
    - PDF (.pdf)
    - DOCX (.docx)
    - TXT (.txt)

    Args:
        path (Path): chemin vers le fichier à charger.

    Returns:
        list[dict]: texte extrait sous forme normalisée.

    Raises:
        ValueError: si le format de fichier n'est pas supporté.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        print(f"📘 Lecture PDF : {path.name}")
        return read_pdf(path)

    if ext == ".docx":
        print(f"📄 Lecture DOCX : {path.name}")
        return read_docx(path)

    if ext == ".txt":
        print(f"📜 Lecture TXT : {path.name}")
        return read_txt(path)

    # Format non reconnu
    raise ValueError(f"Format non supporté : {ext}")