# **Agent Conversationnel IA avec Gestion de Documents et RAG**

## **I. Architecture générale**

Le projet repose sur une architecture modulaire conçue pour le traitement, l’indexation et l’interrogation de documents via un pipeline de Retrieval-Augmented Generation (RAG).  
Il est structuré autour de trois composants fonctionnels distincts :

- **Chunking** : module chargé de la préparation des documents, incluant le nettoyage, la normalisation et le découpage sémantique en segments exploitables par les modèles de langage.
- **Embedding & Retrieval** : composant responsable de la vectorisation des segments, de leur indexation dans une base vectorielle Qdrant, ainsi que de la recherche sémantique et du re-ranking des résultats à l’aide d’un modèle cross-encoder.
- **Application (App)** : interface utilisateur développée avec Streamlit, permettant l’upload de documents, l’interaction avec le système RAG, la génération de réponses et la consultation de l’historique des échanges.

L’ensemble de l’infrastructure est entièrement conteneurisé via Docker, garantissant la reproductibilité des environnements, l’isolation des services et la facilité de déploiement.

---

## **II. Pour lancer le projet**

### **II.1. Démarrage des conteneurs Docker :**

```bash
sudo docker compose up
```

#### **💡 Remarques :**
- Cette commande démarre tous les conteneurs et bloque jusqu'à ce que le serveur Ollama ait complètement téléchargé les modèles LLM_RAG et LLM_CHUNKING.
- Pendant ce temps, les logs du conteneur Ollama affichent la progression du téléchargement des modèles.
- ⚠️ **Première exécution** : le premier lancement du chatbot ou de l’agent peut prendre **plus de 10 minutes**, car les modèles d’Embedding et de ré-ranking doivent être téléchargés depuis Hugging Face dans le conteneur. Les exécutions suivantes seront beaucoup plus rapides.
- Une fois la commande terminée, tous les services sont opérationnels et les modèles sont prêts à être utilisés.
- Dans les logs, vous verrez des messages comme :
```bash
pulling manifest
ollama-server  | pulling 2af3b81862c6: 100% ▕██████████████████▏ 637 MB
ollama-server  | ...
ollama-server  | verifying sha256 digest
ollama-server  | writing manifest
ollama-server  | success
ollama-server  | Model <LLM_RAG> pulled successfully.
ollama-server  | Model <LLM_CHUNKING> pulled successfully.
```
- Il faut attendre que ce processus soit terminé avant d’utiliser les modèles ou d’interagir avec l’Agent.

### **II.2. Première configuration et utilisation**

#### **Étape 1 : Accéder à l'interface**
- Ouvrez votre navigateur et allez à l'URL suivante : http://localhost:8501
- Attendez pour que l'interface se charge complètement.

![image_acceuil](./Screenshots/image_acceuil.png)

#### **Étape 2 : Créer un utilisateur administrateur**
- Sur la page d'accueil, vous verrez une interface de connexion.
- Cliquez sur "Créer un compte".
- Créez un compte administrateur avec :
    - Nom d'utilisateur admin
    - Mot de passe sécurisé

![image_creation_utilisateur](./Screenshots/image_creation_utilisateur.png)

- Pourquoi créer un admin ? L'administrateur est le seul utilisateur qui peut :
    - Ajouter des documents
    - Lancer le chunking des documents
    - Effectuer l'indexation

#### **Étape 3 : Se connecter en tant qu'admin**
- Utilisez les identifiants que vous venez de créer pour vous connecter.

![image_connexion](./Screenshots/image_connexion.png)

#### **Étape 4 : Uploader des documents**
- Dans la barre latérale (sidebar), cliquez sur "Upload"
- Vous verrez l'interface d'ajout de documents

![image_sidebar_upload](./Screenshots/image_sidebar_upload.png)

#### **Étape 5 : Ajouter des documents**
- Cliquez sur "Browse files"
- Sélectionnez les documents que vous souhaitez ajouter à la base de données
- Une fois les fichiers sélectionnés, confirmez l'ajout en cliquant sur le bouton approprié

#### **Étape 6 : Lancer le chunking**
- Après confirmation de l'ajout, cliquez sur le bouton : "Lancer le chunking des nouveaux documents"
- Ce processus prépare les documents pour le traitement

#### **Étape 7 : Révectoriser les checks**
- Une fois le chunking terminé, cliquez sur : "Re-vectoriser tous les chunks"
- Cette étape crée les embeddings et indexe les documents dans la base vectorielle

#### **Étape 8 : Utiliser le chatbot**
- Retournez dans le sidebar et cliquez sur "Chatbot"
- Vous pouvez maintenant communiquer avec votre chatbot
- Posez des questions basées sur les documents que vous avez uploadés

![image_chatbot](./Screenshots/image_chatbot.png)

#### **Étape 9 (Optionnelle) : Créer d'autres utilisateurs**
- Vous pouvez vous déconnecter (bouton "Déconnexion")
- Créer un nouveau compte utilisateur standard
- Vous connecter avec ce nouveau compte
- Le chatbot sera également accessible pour cet utilisateur, mais sans les privilèges d'administration

---

## **III. Installation des dépendances supplémentaires**

Cette partie concerne l’installation de dépendances additionnelles qui ne sont pas requises pour le fonctionnement normal de l’Agent.  
Ces packages sont nécessaires uniquement pour certains fichiers :  
- Le premier script sert à choisir le modèle d’Embedding à utiliser. (`benchmark_BGE-M3_Multilingual-E5-Large.py`)
- Le deuxième script sert à évaluer nos générations et résultats. (`Evaluation_RAG_Deepeval.py`) 
Ces dépendances sont donc **optionnelles** et installées manuellement pour des tests ou évaluations ponctuelles.

### **III.1. Entrer dans le conteneur Agent**

Pour installer ces dépendances, il faut d’abord accéder au conteneur où tourne l’Agent.  
Cette étape est nécessaire car l’installation se fait à l’intérieur du conteneur et n’interfère pas avec l’Agent principal.

```bash
docker exec -it agent /bin/bash
```

### **III.2. Installer les dépendances**

Une fois dans le conteneur, on installe les packages supplémentaires listés dans requirements_additional.txt. Ces packages ne sont pas inclus dans le build Docker par défaut car ils ne sont pas indispensables au fonctionnement de l’Agent.

```bash
pip install --no-cache-dir -r requirements_additional.txt
```

### **III.3. Exécution manuelle des scripts Python**

Après installation des dépendances supplémentaires, tu peux lancer les scripts spécifiques à des tests ou évaluations :  

- **Benchmark des modèles d’Embedding** : si tu veux tester le choix du modèle BGE-M3 ou Multilingual-E5-Large, exécute le script `benchmark_BGE-M3_Multilingual-E5-Large.py`.  
- **Évaluation des générations RAG avec DeepEval** : si tu veux évaluer les réponses générées par l’Agent et obtenir des métriques, exécute le script `Evaluation_RAG_Deepeval.py`.  

Ces commandes sont **optionnelles** et ne modifient en rien le fonctionnement normal de l’Agent.

#### **III.3.1. Benchmark des modèles d’Embedding**

Pour tester différents modèles d’Embedding (ex. BGE-M3 ou Multilingual-E5-Large) :

```bash
# Depuis le conteneur Agent
python Embedding/Benchmarks/benchmark_BGE-M3_Multilingual-E5-Large.py
```
- Le script va sauvegarder les résultats dans `Embedding/Benchmarks/evaluation_results.json`.

#### **III.3.2. Évaluation des générations RAG avec DeepEval**

##### **a- Préparer le modèle LLM pour l’évaluation**

Avant de lancer l’évaluation, assure-toi que le modèle LLM que tu souhaites utiliser pour DeepEval est installé dans le conteneur Ollama.

- Entrer dans le conteneur Ollama :
```bash
docker exec -it ollama-server /bin/bash
```

- Lister les modèles existants :
```bash
ollama list
```
Vérifier que le modèle désiré n’est pas déjà installé.

- Installer le modèle manuellement si nécessaire :
```bash
ollama pull NOM_DU_MODELE
```
Remplace NOM_DU_MODELE par le nom exact du modèle que tu souhaites utiliser.
L’installation peut prendre plusieurs minutes selon la taille du modèle.

- Vérifier l’installation :
```bash
ollama list
```
Le modèle doit maintenant apparaître dans la liste.

##### **b- Redirection du conteneur Agent vers Ollama pour DeepEval**

DeepEval ne peut pas contacter Ollama directement à travers les conteneurs, alors il est possible de créer un forward TCP pour exposer Ollama dans le conteneur Agent.

- Installer socat dans le conteneur Agent
```bash
docker exec -it agent bash
apt-get update && apt-get install -y socat
```

- Lancer le forward TCP
```bash
socat TCP-LISTEN:11434,fork TCP:ollama-server:11434 &
```

- Vérifier la connexion
```bash
python -c "import requests; print(requests.get('http://localhost:11434/v1/models').text)"
```
Si tu obtiens la liste des modèles Ollama → la redirection fonctionne correctement.

##### **c- Lancer l’évaluation DeepEval**

- Entrer dans le conteneur Agent (si ce n’est pas déjà fait) :
```bash
docker exec -it agent /bin/bash
```
- Exécuter le script :
```bash
python Evaluation/RAG/Evaluation_RAG_Deepeval.py
```
- Le script va sauvegarder les résultats dans `Evaluation/RAG/evaluation_results.json`.