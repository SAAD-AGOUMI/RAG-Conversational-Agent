FROM ubuntu:22.04

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Dépendances nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier le script d’entrée
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Exposer le port Ollama
EXPOSE 11434

# Lancer Ollama
CMD ["/entrypoint.sh"]