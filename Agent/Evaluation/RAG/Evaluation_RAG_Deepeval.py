"""
📊 Benchmark DeepEval pour évaluation d'un système RAG

Ce fichier permet d'évaluer un système de Retrieval-Augmented Generation (RAG) 
en utilisant DeepEval et un LLM Ollama.

Fonctionnalités principales :
1. Chargement du dataset de test (`Golden_dataset.json`) contenant :
   - Les questions à poser (queries)
   - Le corpus de documents
   - Les réponses attendues (ground_truth)
2. Génération de réponses à partir des chunks de documents récupérés via 
   `search_and_rerank` et re-ranking avec un modèle cross-encoder.
3. Évaluation automatique des réponses générées selon 4 métriques DeepEval :
   - Faithfulness (fidélité)
   - Contextual Precision (précision contextuelle)
   - Contextual Recall (rappel contextuel)
   - Answer Relevancy (pertinence de la réponse)
4. Calcul du score global moyen et sauvegarde des résultats dans un fichier JSON.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric
)
from deepeval.evaluate import AsyncConfig
from deepeval.models import OllamaModel

# -------------------------------------------------
# Timeout par tentative de DeepEval prolongé (10h)
# -------------------------------------------------
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "36000" 

# -------------------------------------------------
# Définition de la racine du projet
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "App"))
sys.path.append(str(ROOT_DIR / "Embedding"))

from utils.llm_client import query_llm

# -------------------------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------------------------
def load_json(path):
    """Charge un fichier JSON depuis un chemin donné"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

dataset_path = ROOT_DIR / "Evaluation" / "RAG" / "Golden_dataset.json"
dataset = load_json(dataset_path)  # contient queries, corpus, ground_truth
queries = dataset["queries"]
corpus = dataset["corpus"]
ground_truth = dataset["ground_truth"]

corpus_ids = [d["id"] for d in corpus]
corpus_texts = [d["text"] for d in corpus]

# -------------------------------------------------
# GÉNÉRATION DE LA RÉPONSE
# -------------------------------------------------
from search_and_rerank import search_and_rerank

def generate_answer(question):
    """Génère la réponse à une question en utilisant RAG"""
    results, parents = search_and_rerank(question)

    print(f"\n🔍 Résultats pour : {question}\n")

    chunks_recuperes = "\n\n".join(
        f"- Chunk (rank={r['rank']}, doc={r['doc']}, page={r['page']}, score={r['rerank_score']:.2f}) : {r['chunk'][:400]}..."
        for r in results
    )

    chunks_list = [r["chunk"] for r in results]

    prompt = f"""
    RÔLE
    Tu es un assistant factuel chargé de répondre à une question uniquement à partir des passages fournis.
    Tu n'as pas le droit d'inventer ou d'ajouter des informations.

    CONTEXTE
    Question utilisateur :
    {question}

    EXTRAITS DISPONIBLES :
    {chunks_recuperes}

    TEXTES SOURCES ASSOCIÉS (parents) :
    {parents}

    PROCÉDURE OBLIGATOIRE :
    1. Sélectionne jusqu'à DEUX extraits les plus pertinents.
    2. Identifie les textes sources associés.
    3. Analyse EXCLUSIVEMENT ces textes sources.
    4. Si un texte contient l'information demandée, répond uniquement avec celle-ci.
    5. Si aucun texte ne contient l'information, répond exactement :
    "L'information demandée n'apparaît pas dans les extraits fournis."

    CONTRAINTES STRICTES :
    - Ne rien inventer.
    - Reformule au plus près du texte.
    - Ne mentionne jamais les mots “chunk”, “parent”, “score”, “reranker_score”.
    - Réponse directe, factuelle, sans introduction, sans conclusion.
    """

    reponse_final = query_llm(prompt)
    return reponse_final.strip(), chunks_list

# -------------------------------------------------
# ÉVALUATION GÉNÉRATION (DeepEval + Ollama)
# -------------------------------------------------
# Charger les variables d'environnement
load_dotenv()
ollama_model_name = os.getenv("LLM_EVALUATION")
eval_llm = OllamaModel(model=ollama_model_name)

def evaluate_generation(queries, ground_truth):
    """Évalue toutes les questions du dataset avec DeepEval"""
    print("\n🔹 Évaluation avec DeepEval...")
    print(f"📌 Total de questions : {len(queries)}\n")

    test_cases = []

    for idx, q in enumerate(queries):
        print(f"⏳ Traitement question {idx + 1}/{len(queries)}: {q['text'][:150]}...")
        
        qid = q["id"]
        relevant_docs = [c for c in corpus if c["id"] in ground_truth.get(qid, [])]

        response, chunks_list = generate_answer(q["text"])
        expected_answer = "\n".join([d["text"] for d in relevant_docs])

        test_case = LLMTestCase(
            input=q["text"],
            actual_output=response,
            expected_output=expected_answer,
            retrieval_context=chunks_list
        )
        test_cases.append(test_case)

    metrics = [
        FaithfulnessMetric(model=eval_llm),
        ContextualPrecisionMetric(model=eval_llm),
        ContextualRecallMetric(model=eval_llm),
        AnswerRelevancyMetric(model=eval_llm),
    ]

    _ = evaluate(
        test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False)
    )

    deepeval_file = ROOT_DIR / ".deepeval" / ".latest_test_run.json"

    if deepeval_file.exists():
        with open(deepeval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics_summary = data["testRunData"]["metricsScores"]

        results_dict = {
            "total_questions": len(test_cases),
            "evaluation_time_seconds": data["testRunData"]["runDuration"],
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {}
        }

        for metric_data in metrics_summary:
            metric_name = metric_data["metric"]
            scores = metric_data["scores"]
            passed = metric_data["passes"]
            total = passed + metric_data["fails"]
            avg_score = sum(scores) / len(scores) if scores else 0
            pass_rate = (passed / total * 100) if total > 0 else 0

            results_dict["metrics"][metric_name] = {
                "pass_rate_percent": round(pass_rate, 2),
                "passed": passed,
                "total": total,
                "average_score": round(avg_score, 4),
                "scores": scores
            }

            print(f"{metric_name}: {pass_rate:.2f}% pass rate")

        all_avg = [results_dict["metrics"][m]["average_score"] for m in results_dict["metrics"]]
        global_score = sum(all_avg) / len(all_avg) if all_avg else 0
        results_dict["global_average_score"] = round(global_score, 4)

        print("\n" + "="*80)
        print(f"Score Global Moyen: {global_score:.4f}")
        print("="*80 + "\n")

        output_path = ROOT_DIR / "Evaluation" / "RAG" / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ Résultats sauvegardés dans: {output_path}")
        print(f"✅ Résultats DeepEval complets dans: {deepeval_file}\n")
        return results_dict
    else:
        print(f"❌ ERREUR: Fichier {deepeval_file} introuvable!")
        return None

# -------------------------------------------------
# Lancement de l'évaluation
# -------------------------------------------------
if __name__ == "__main__":
    evaluate_generation(queries, ground_truth)
