"""
Point d'entrée principal de l'Agent.

Ce script :
1. Attend que les services externes (Qdrant, Ollama) soient prêts.
2. Initialise l'environnement de l'Agent.
3. Lance l'application Streamlit (page d'accueil).
"""

import os
import subprocess

from utils.wait_for_services import wait_for_services

# ---------------------------------------------------------
# Vérification des services
# ---------------------------------------------------------
print("⌛ Vérification des services externes (Qdrant, Ollama)...")
wait_for_services()
print("🚀 Services prêts, démarrage de l'Agent...")


# ---------------------------------------------------------
# Lancer l'application Streamlit
# ---------------------------------------------------------
def main():
    """
    Lance l'interface Streamlit de l'Agent.
    """
    # Définit le chemin vers le dossier App
    app_dir = os.path.join(os.path.dirname(__file__), "App")

    # Lancement de Streamlit sur la page Home.py
    subprocess.run(
        [
            "streamlit",
            "run",
            os.path.join(app_dir, "Home.py"),
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
        ]
    )


# ---------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
