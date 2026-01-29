"""
Page principale de discussion du Chatbot RAG INEO Defense.

Ce fichier Streamlit gère :
1. L'affichage de l'interface de chat avec titre, logo et sous-titre.
2. L'authentification de l'utilisateur.
3. La gestion des états de session pour les messages et l'historique.
4. La barre latérale permettant de configurer les options RAG, de créer
   un nouveau chat et de naviguer dans l'historique.
5. La pipeline RAG (Recherche + Re-ranking) pour interroger la base
   documentaire et fournir des réponses pertinentes.
6. L'interrogation du LLM pour générer la réponse finale.
7. La sauvegarde automatique de l'historique des conversations.
"""

import sys
import os
import base64
from datetime import datetime
import streamlit as st

# ---------------------------------------------------------
# Ajout du dossier racine pour l'import des modules utils
# ---------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

from utils.llm_client import query_llm
from utils.auth_local import require_login
from utils.config_loader import (
    load_config,
    list_history_files,
    load_history_for,
    save_history_for,
    user_folder,
    get_history_title,
)
from utils.history_utils import new_history_filename
from Embedding.search_and_rerank import search_and_rerank

# ---------------------------------------------------------
# Chargement de la configuration
# ---------------------------------------------------------
config = load_config()

st.set_page_config(
    page_title=f"{config['app']['title']} - Discussion",
    page_icon="💬",
    layout="wide"
)

# ---------------------------------------------------------
# Logo + titre principal
# ---------------------------------------------------------
logo_path = ROOT_DIR + "/App/utils/assets/images/ineo.jpg"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center;">
            <img src="data:image/png;base64,{logo_base64}"
                 style="width:100px; height:70px; margin-right:12px;"/>
            <h1 style="margin:0; padding:0;">{config['app']['title']} - Discussion</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title(f"{config['app']['title']} - Discussion")
    st.write("⚠️ Logo introuvable.")

st.markdown("<h3>💬 Posez-moi une question...</h3>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Authentification obligatoire
# ---------------------------------------------------------
require_login()
username = st.session_state["username"]
folder = user_folder(username)

# ---------------------------------------------------------
# Initialisation des états de session
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_history_file" not in st.session_state:
    st.session_state["current_history_file"] = None

if "chat_title" not in st.session_state:
    st.session_state["chat_title"] = "Nouveau chat"

if "pending_title" not in st.session_state:
    st.session_state["pending_title"] = None

# ---------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")

    st.subheader("🧠 Mode RAG (Recherche + Re-ranking)")
    use_rag = st.checkbox("Activer la recherche documentaire", value=True)
    top_k = st.slider("Top-K Qdrant", 5, 40, 20)
    final_k = st.slider("Nombre final de passages", 1, 10, 3)
    threshold = st.slider("Seuil de pertinence", -15.0, 5.0, -7.0)

    st.subheader("🆕 Nouveau Chat")
    custom_title = st.text_input("Nom du chat (optionnel)")
    if st.button("Créer un nouveau chat"):
        st.session_state["current_history_file"] = None
        st.session_state["messages"] = []
        st.session_state["pending_title"] = custom_title.strip() or None
        st.session_state["chat_title"] = custom_title.strip() or "Nouveau chat"
        st.rerun()

    st.subheader("📚 Historique")
    files = list_history_files(username)

    def _format_option(option: str) -> str:
        if option == "--":
            return "--"
        return get_history_title(username, option)

    selection = st.selectbox(
        "Sélectionnez une discussion",
        ["--"] + files,
        format_func=_format_option
    )

    if selection != "--" and selection != st.session_state["current_history_file"]:
        st.session_state["current_history_file"] = selection
        st.session_state["messages"] = load_history_for(username, selection)
        st.session_state["chat_title"] = get_history_title(username, selection)
        st.session_state["pending_title"] = None
        st.rerun()

# ---------------------------------------------------------
# Affichage du titre de la discussion
# ---------------------------------------------------------
st.caption(f"📄 Discussion : {st.session_state.get('chat_title', 'Nouveau chat')}")

# ---------------------------------------------------------
# Affichage des messages
# ---------------------------------------------------------
for role, content, timestamp in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)
        st.caption(f"⏰ {timestamp}")

# ---------------------------------------------------------
# Saisie de message utilisateur
# ---------------------------------------------------------
prompt = st.chat_input("Posez-moi une question...")

if prompt:
    now = datetime.now().isoformat(timespec="seconds")
    st.session_state["messages"].append(("user", prompt, now))

    # ---------------------------------------------------------
    # Pipeline RAG
    # ---------------------------------------------------------
    if use_rag:
        with st.spinner("Recherche dans la base documentaire..."):
            results, parents = search_and_rerank(
                query=prompt,
                top_k=top_k,
                final_k=final_k,
                threshold=threshold
            )

        chunks_recuperes = "\n\n".join(
            f"- Chunk (rank={r['rank']}, score={r['rerank_score']:.2f}) : {r['chunk']}"
            for r in results
        )

        parents_recuperes = "\n\n".join(
            f"- ParentID {pid} : {txt}"
            for pid, txt in parents.items()
        )

        final_prompt = f"""
        RÔLE
        Tu es un assistant d'extraction factuelle strictement limité aux textes fournis.
        Tu n'as aucune connaissance externe et tu n'as pas le droit d'en utiliser.

        CONTEXTE
        Question utilisateur :
        {prompt}

        EXTRAITS DISPONIBLES (avec scores déjà calculés, à ne pas commenter) :
        {chunks_recuperes}

        TEXTES SOURCES ASSOCIÉS :
        {parents_recuperes}

        PROCÉDURE OBLIGATOIRE (à suivre dans cet ordre, sans exception) :
        1. Parmi les extraits fournis, identifie au maximum DEUX extraits ayant la meilleure pertinence sémantique pour répondre à la question.
        2. Identifie les textes sources associés à ces extraits.
        3. Analyse EXCLUSIVEMENT le contenu de ces textes sources.
        4. Détermine si l'un de ces textes contient une information explicite, directe et suffisante pour répondre à la question.
        5. Si oui, formule la réponse uniquement à partir des formulations présentes dans ce texte.
        6. Si non, produis la réponse négative standard définie ci-dessous.

        CONTRAINTES ABSOLUES :
        - Interdiction totale d'utiliser des connaissances extérieures.
        - Interdiction d'inférer, déduire, compléter ou interpréter au-delà du texte.
        - Interdiction d'ajouter des détails absents des textes.
        - Interdiction de mentionner la méthode, les scores ou les identifiants techniques.
        - Le vocabulaire doit rester au plus proche de celui du texte source.

        FORMAT DE SORTIE :
        - Réponse directe, factuelle, sans introduction ni conclusion.
        - Une seule réponse finale, pas d'explication du raisonnement.

        RÉPONSE NÉGATIVE STANDARD (à utiliser obligatoirement si l'information n'est pas trouvée) :
        « L'information demandée n'apparaît pas dans les extraits fournis. »
        """.strip()

    else:
        final_prompt = prompt

    # ---------------------------------------------------------
    # Réponse du LLM
    # ---------------------------------------------------------
    with st.spinner("Génération de la réponse..."):
        response = query_llm(final_prompt)

    now_resp = datetime.now().isoformat(timespec="seconds")
    st.session_state["messages"].append(("assistant", response, now_resp))

    # ---------------------------------------------------------
    # Création du fichier d'historique si c'est le premier message
    # ---------------------------------------------------------
    if st.session_state["current_history_file"] is None:
        filename = new_history_filename(folder)
        st.session_state["current_history_file"] = filename

        if st.session_state.get("pending_title"):
            chat_title = st.session_state["pending_title"]
        else:
            title_prompt = f"""
            RÔLE
            Tu es un générateur de titres courts pour l'historique d'un chatbot.

            CONTEXTE
            La discussion commence par la question suivante :
            {prompt}

            OBJECTIF
            Produire un titre synthétique décrivant le sujet principal de la discussion.

            CONTRAINTES STRICTES :
            - Maximum 6 mots.
            - Le titre doit être dans la même langue que la question utilisateur.
            - Style nominal (pas de phrase complète).
            - Pas de verbe conjugué.
            - Pas de ponctuation.
            - Pas de guillemets.
            - Pas d'article inutile (le, la, un, une) sauf si indispensable au sens.
            - Pas de commentaire ou d'explication.

            FORMAT DE SORTIE :
            - Une seule ligne.
            - Le titre uniquement.
            """.strip()

            chat_title = query_llm(title_prompt).strip()

        st.session_state["chat_title"] = chat_title

    # ---------------------------------------------------------
    # Sauvegarde de l'historique
    # ---------------------------------------------------------
    save_history_for(
        username,
        st.session_state["messages"],
        st.session_state["current_history_file"],
        title=st.session_state.get("chat_title")
    )

    st.rerun()