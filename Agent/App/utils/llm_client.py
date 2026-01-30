"""
🌐 Gestion de la connexion à Ollama et requêtes LLM

Ce module fournit :
1. La configuration du client Ollama pour un environnement conteneurisé (host = localhost)
2. La définition de la variable d'environnement appropriée pour le client Ollama
3. Une fonction pour interroger le modèle LLM avec un prompt utilisateur
4. Un message système sécurisant pour éviter les hallucinations et protéger la vie privée
"""

from dotenv import load_dotenv
import os
import ollama

# -----------------------------
# 🌐 Configuration du client Ollama pour conteneur
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_URL")

# Crée un client Ollama explicitement lié au host correct
client = ollama.Client(host=OLLAMA_HOST)

# Charger la configuration depuis le fichier .env
load_dotenv()

# -----------------------------
# 🛡️ Message système pour éviter les hallucinations et protéger la vie privée
# -----------------------------
SYSTEM_PROMPT = (
    "RÔLE\n"
    "Tu es un assistant factuel et non spéculatif.\n\n"

    "RÈGLES GÉNÉRALES\n"
    "- Réponds uniquement à partir des informations explicitement fournies par l'utilisateur ou par le contexte du prompt.\n"
    "- N'invente aucune information.\n"
    "- N'ajoute aucun détail qui ne figure pas dans les données fournies.\n"
    "- Si l'information demandée n'est pas présente, dis-le clairement.\n\n"

    "CONFIDENTIALITÉ\n"
    "- Tu ne fais aucune référence à des utilisateurs, conversations ou données non présentes dans le prompt.\n"
    "- Si une question porte sur des données non fournies, réponds que l'information n'est pas disponible.\n\n"

    "STYLE DE RÉPONSE\n"
    "- Réponse directe et concise.\n"
    "- Pas d'explication sur ton fonctionnement interne.\n"
    "- Pas de mention de règles ou de politiques."
)

# -----------------------------
# 💬 Fonction pour interroger le modèle LLM
# -----------------------------
def query_llm(prompt: str):
    """
    Envoie un prompt utilisateur au LLM Ollama et retourne la réponse.

    Paramètres :
    - prompt (str) : texte à envoyer au modèle

    Retour :
    - str : réponse générée par le modèle ou message d'erreur
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    model_name = os.getenv("LLM_RAG")

    try:
        response = client.chat(
            model=model_name,
            messages=messages,
            stream=False,
        )
    except Exception as e:
        return (
            f"⚠️ Erreur de connexion au modèle ({model_name}).\n"
            f"Assurez-vous que Ollama est en cours d'exécution.\n\n"
            f"Détails : {e}"
        )

    # -----------------------------
    # ✅ Extraction de la réponse de l'assistant selon le format du client Ollama
    # -----------------------------
    # Nouveau format Python Ollama client
    try:
        return response.message.content
    except:
        pass

    # Ancien format (dict)
    try:
        return response["message"]["content"]
    except:
        pass

    # Fallback (debug)
    return str(response)
