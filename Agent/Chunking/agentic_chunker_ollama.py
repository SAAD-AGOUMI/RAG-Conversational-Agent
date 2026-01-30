"""
AgenticChunker : Découpe des paragraphes en chunks via LLM Ollama.

Ce fichier définit la classe `AgenticChunker` qui utilise un modèle LLM
(Ollama) pour segmenter un texte en sous-parties cohérentes appelées "chunks".

Principales fonctionnalités :
1. `query_llm` : envoie un prompt au LLM et récupère la réponse.
2. `chunk_paragraph` : découpe un paragraphe et ajoute des métadonnées.
"""

from dotenv import load_dotenv
import os
import subprocess
import time
import ollama

# Charger la configuration depuis le fichier .env
load_dotenv()
model_name = os.getenv("LLM_CHUNKING")
OLLAMA_URL = os.getenv("OLLAMA_URL")
client = ollama.Client(host=OLLAMA_URL)

AGENTIC_PROMPT = """RÔLE
Tu es un agent de segmentation mécanique de texte.
Tu n'as PAS le droit de reformuler, corriger, traduire ou normaliser quoi que ce soit.

TÂCHE
Découper le texte fourni en sous-parties cohérentes appelées "chunks".

INVARIANT ABSOLU (CRITIQUE) :
Si l'on concatène tous les chunks EXACTEMENT dans l'ordre, en supprimant uniquement les séparateurs "/", on doit obtenir STRICTEMENT le texte original, caractère par caractère.

RÈGLES STRICTES :
- NE MODIFIE JAMAIS le texte (aucun mot, aucun espace, aucune ponctuation).
- NE CORRIGE PAS les fautes, même évidentes.
- NE TRADUIS PAS et ne changes pas la langue.
- Découpe uniquement aux frontières naturelles des idées (phrases, propositions, transitions).
- Chaque chunk doit rester compréhensible isolément.
- Évite les chunks < 50 caractères ou > 800 caractères.
- Ne fusionne pas artificiellement des idées distinctes pour respecter la taille.

FORMAT DE SORTIE (OBLIGATOIRE) :
- Une seule ligne.
- Les chunks séparés UNIQUEMENT par le caractère "/".
- Aucun "/" au début ou à la fin.
- Aucun retour à la ligne.
- Aucun texte explicatif.

EXEMPLE :
Input :
"Marie aime les pommes. Elle vit à Paris. Elle travaille dans une librairie."

Output :
"Marie aime les pommes./ Elle vit à Paris./ Elle travaille dans une librairie."

TEXTE À DÉCOUPER :
{paragraph}
"""

class AgenticChunker:
    """
    Le LLM décide du découpage des chunks.
    Chaque chunk hérite :
    - d'un parent_id unique (identifiant du paragraphe)
    - du numéro de page
    - du nom du document
    """

    def __init__(self, model=model_name, delay=1.0):
        """
        Initialise l'agent de découpage.

        Args:
            model (str): Nom du modèle LLM à utiliser.
            delay (float): Temps d'attente après chaque requête (en secondes).
        """
        self.model = model
        self.delay = delay
        self.chunks = {}

    # -----------------------------------------------------------
    # 1. Appel au modèle Ollama
    # -----------------------------------------------------------
    def query_llm(self, prompt):
        """
        Envoie un prompt au modèle LLM et récupère la réponse.

        Args:
            prompt (str): Texte à envoyer au modèle.

        Returns:
            str: Réponse du LLM.
        """
        try:
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.message.content
        except Exception as e:
            print(f"❌ Erreur Ollama : {e}")
            return ""

    # -----------------------------------------------------------
    # 2. Création des chunks
    # -----------------------------------------------------------
    def chunk_paragraph(self, paragraph_text, document_name, page_number, parent_id):
        """
        Découpe le texte via LLM et ajoute les métadonnées :
        - ID parent unique
        - nom du document
        - numéro de page

        Args:
            paragraph_text (str): Paragraphe à découper.
            document_name (str): Nom du document.
            page_number (int): Numéro de page.
            parent_id (str): Identifiant unique du paragraphe.

        Returns:
            list[dict]: Liste des chunks avec métadonnées.
        """
        
        prompt = AGENTIC_PROMPT.format(paragraph=paragraph_text)
        response = self.query_llm(prompt)
        raw_chunks = [c.strip() for c in response.split("/") if c.strip()]

        chunks_list = []
        for chunk_text in raw_chunks:
            chunks_list.append({
                "parent_paragraph_id": parent_id,
                "page_number": page_number,
                "document_name": document_name,
                "text": chunk_text
            })

        time.sleep(self.delay)
        return chunks_list