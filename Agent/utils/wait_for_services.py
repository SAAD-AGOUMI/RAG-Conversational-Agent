import os
import time

import requests
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL")


def wait_for_services():
    while True:
        try:
            # Test Qdrant (endpoint valide)
            qdrant_resp = requests.get(f"{QDRANT_URL}/collections", timeout=2)
            qdrant_resp.raise_for_status()

            # Test Ollama
            ollama_resp = requests.get(OLLAMA_URL, timeout=2)
            ollama_resp.raise_for_status()

            print("✅ Qdrant et Ollama sont prêts")
            break

        except Exception as e:
            print(f"⏳ En attente des services... ({e})")
            time.sleep(2)
