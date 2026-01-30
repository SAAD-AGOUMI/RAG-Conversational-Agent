"""
Ce fichier définit une fonction utilitaire pour traiter un paragraphe individuel
dans le pipeline de chunking.

Pour chaque paragraphe :
- un identifiant unique est généré
- un LLM (via AgenticChunker et Ollama) est utilisé pour découper le texte en chunks
- les métadonnées du paragraphe et les chunks générés sont retournés sous une
structure normalisée

Ce module est typiquement utilisé dans un pipeline parallèle ou itératif
pour traiter plusieurs paragraphes indépendamment.
"""

from uuid import uuid4  # Génération d'identifiants uniques

from Chunking.agentic_chunker_ollama import (  # Chunker basé sur un LLM via Ollama
    AgenticChunker,
)


def process_paragraph(args):
    """
    Traite un paragraphe unique :
    - génère un identifiant de paragraphe
    - appelle le LLM pour découper le texte en chunks sémantiques
    - retourne le paragraphe et ses chunks avec métadonnées

    Args:
        args (tuple):
            - paragraph_text (str): texte du paragraphe à découper
            - file_name (str): nom du document source
            - page_number (int): numéro de page du paragraphe

    Returns:
        dict: dictionnaire contenant :
            - "paragraph": métadonnées et texte du paragraphe
            - "chunks": liste des chunks générés par le LLM
    """
    # Déballage des arguments
    paragraph_text, file_name, page_number = args

    # Génération d'un identifiant court et unique pour le paragraphe
    paragraph_id = str(uuid4())[:8]

    # Initialisation du chunker LLM (modèle dédié au découpage)
    chunker = AgenticChunker()

    # Découpage du paragraphe en chunks via le LLM
    chunks = chunker.chunk_paragraph(
        paragraph_text,
        document_name=file_name,
        page_number=page_number,
        parent_id=paragraph_id,
    )

    # Structure de sortie normalisée
    return {
        "paragraph": {
            "paragraph_id": paragraph_id,
            "document_name": file_name,
            "page_number": page_number,
            "text": paragraph_text,
        },
        "chunks": chunks,
    }
