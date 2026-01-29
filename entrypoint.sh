#!/bin/sh
set -e

# Charger les variables depuis .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Vérifier que LLM_RAG est défini
if [ -z "$LLM_RAG" ]; then
  echo "Erreur : LLM_RAG n'est pas défini dans .env"
  exit 1
fi

# Vérifier que LLM_CHUNKING est défini
if [ -z "$LLM_CHUNKING" ]; then
  echo "Erreur : LLM_CHUNKING n'est pas défini dans .env"
  exit 1
fi

# Démarrer Ollama en arrière-plan
ollama serve &

# Attendre que le serveur soit prêt
echo "Waiting for Ollama server..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done

# Boucle pour s'assurer que le pull réussit (Pull du modèle LLM_RAG)
echo "Pulling model $LLM_RAG..."
while ! ollama pull $LLM_RAG; do
  echo "Pull failed, retrying in 5s..."
  sleep 5
done

echo "Model $LLM_RAG pulled successfully."

# Boucle pour s'assurer que le pull réussit (Pull du modèle LLM_CHUNKING)
echo "Pulling model $LLM_CHUNKING..."
while ! ollama pull $LLM_CHUNKING; do
  echo "Pull failed, retrying in 5s..."
  sleep 5
done

echo "Model $LLM_CHUNKING pulled successfully."

# Garder Ollama au premier plan
wait
