"""
MATCHING RAG V3 : Avec cache des embeddings + validation renforcée
- 50x plus rapide que la V2
- Validation de l'ID retourné par le LLM
"""

import json
import ollama
import numpy as np
import re
import time
from datetime import datetime
import os

# ============================================
# IMPORT DU CACHE
# ============================================
from cache_manager import charger_cache_embeddings, verifier_cache_a_jour, generer_cache_embeddings

# ============================================
# CONFIGURATION
# ============================================
DATASET_FILE   = "dataset_4.json"
INPUT_FILE     = "mistral_resultat.json"

EMBED_MODEL    = "nomic-embed-text"
LLM_MODEL      = "mistral"
THRESHOLD      = 0.85
DOMINANCE_GAP  = 0.05
TOP_K          = 3

# ============================================
# VARIABLE GLOBALE : Cache des embeddings
# ============================================
EMBEDDINGS_CACHE = None


# ============================================
# PHASE 0 : INITIALISATION DU CACHE
# ============================================
def initialiser_cache():
    global EMBEDDINGS_CACHE

    if not verifier_cache_a_jour():
        print("🔄 Régénération du cache...")
        generer_cache_embeddings()

    EMBEDDINGS_CACHE = charger_cache_embeddings()

    if EMBEDDINGS_CACHE is None:
        raise Exception("❌ Impossible de charger le cache des embeddings !")


# ============================================
# PHASE 1 : CHARGER LE DATASET
# ============================================
def charger_dataset(fichier=DATASET_FILE):
    print(f"\n📂 Chargement de {fichier}...")
    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ {len(data['intentions'])} intentions chargées")
    return data


# ============================================
# PHASE 2 : LIRE LE RÉSUMÉ
# ============================================
def lire_resume(fichier=INPUT_FILE):
    print(f"\n📖 Lecture du résumé depuis {fichier}...")

    # Trouver le fichier le plus récent si plusieurs existent
    import glob
    fichiers = glob.glob("mistral_resultat_*.json")
    if fichiers:
        fichier = max(fichiers, key=os.path.getctime)

    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)
    resume = data["resume"]
    print(f"✅ Résumé : {resume}")
    return resume


# ============================================
# PHASE 3 : EMBEDDING (UNIQUEMENT POUR L'INPUT)
# ============================================
def get_embedding(texte):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=texte)
    return np.array(response["embedding"])


def cosine_similarity(vec1, vec2):
    dot   = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ============================================
# PHASE 4 : RAG LAYER V3
# ============================================
def rag_top_k(resume, intentions, k=TOP_K):
    global EMBEDDINGS_CACHE

    print(f"\n🔍 RAG Layer V3 (Top-{k}) en cours...")
    print(f"   Embedding de : '{resume}'")

    debut = time.time()

    input_vec = get_embedding(resume)

    scores = []
    for intention in intentions:
        intent_id = intention["id"]
        intent_vec = EMBEDDINGS_CACHE[intent_id]
        score = cosine_similarity(input_vec, intent_vec)

        scores.append({
            "id":          intention["id"],
            "description": intention["description"],
            "score":       round(float(score), 4),
            "services":    intention["services"],
            "QoS":         intention["QoS"],
            "weight":      intention.get("weight", 0)
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    fin   = time.time()
    duree = round(fin - debut, 2)

    top_k = scores[:k]

    print(f"\n   📊 Top-{k} candidats RAG (en {duree}s) :")
    for i, s in enumerate(top_k):
        print(f"   {i+1}. {s['id']} | score={s['score']} | {s['description'][:45]}...")

    top1 = top_k[0]
    top2 = top_k[1] if len(top_k) > 1 else None
    gap = round(top1["score"] - top2["score"], 4) if top2 else 1.0

    print(f"\n   Gap top1-top2 : {gap} (seuil dominance : {DOMINANCE_GAP})")

    if top1["score"] >= THRESHOLD and gap >= DOMINANCE_GAP:
        print(f"✅ RAG DIRECT : {top1['id']} (score={top1['score']}, gap={gap})")
        return top_k, "rag_direct", duree

    elif top1["score"] >= 0.70:
        print(f"🔄 AMBIGUÏTÉ détectée → LLM Rerank sur Top-{k}")
        return top_k, "llm_rerank", duree

    else:
        print(f"⚠️ Score trop faible ({top1['score']}) → LLM Fallback complet")
        return top_k, "llm_fallback", duree


# ============================================
# PHASE 5 : LLM RERANK (AVEC VALIDATION)
# ============================================
def llm_rerank(resume, top_k_candidats, intentions_valides):
    """
    NOUVEAU : intentions_valides = liste des IDs existants pour validation
    """
    print(f"\n🤖 LLM Reranking (Mistral) sur {len(top_k_candidats)} candidats...")

    candidats_txt = ""
    for c in top_k_candidats:
        candidats_txt += f"- {c['id']}: {c['description']} (score={c['score']})\n"

    debut = time.time()

    try:
        reponse = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an industrial maintenance expert.
Your job is to select the MOST RELEVANT intention for a worker's request.
Reply ONLY with the intention id. No explanation."""
                },
                {
                    "role": "user",
                    "content": f"""Select the most relevant intention for this request.

Worker request: "{resume}"

Candidate intentions (pre-selected by semantic search):
{candidats_txt}

Rules:
- Reply ONLY with the id (example: i10)
- Choose the MOST SPECIFIC and RELEVANT one
- Consider the exact meaning of the request

Answer:"""
                }
            ]
        )

        fin   = time.time()
        duree = round(fin - debut, 2)

        raw       = reponse["message"]["content"].strip()
        match_id  = re.search(r'i\d+', raw)
        intent_id = match_id.group() if match_id else None

        # ✅ VALIDATION : vérifier que l'ID existe
        if intent_id and intent_id not in intentions_valides:
            print(f"⚠️  LLM a retourné {intent_id} mais il n'existe pas !")
            intent_id = None

        if intent_id is None:
            print(f"⚠️  LLM Rerank n'a pas trouvé d'ID valide")
            return None, duree

        print(f"✅ LLM Rerank choisi : {intent_id} (en {duree}s)")
        return intent_id, duree

    except Exception as e:
        print(f"❌ Erreur LLM Rerank : {e}")
        return None, 0


# ============================================
# PHASE 6 : LLM FALLBACK (AVEC VALIDATION)
# ============================================
def llm_fallback(resume, intentions, intentions_valides):
    print(f"\n🤖 LLM Fallback complet (Mistral)...")

    liste = ""
    for i in intentions:
        liste += f"- {i['id']}: {i['description']}\n"

    debut = time.time()

    try:
        reponse = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an industrial maintenance expert.
Match the worker request to the most relevant intention.
Reply ONLY with the intention id. No explanation."""
                },
                {
                    "role": "user",
                    "content": f"""Worker request: "{resume}"

Available intentions:
{liste}

Answer (only the id):"""
                }
            ]
        )

        fin   = time.time()
        duree = round(fin - debut, 2)

        raw       = reponse["message"]["content"].strip()
        match_id  = re.search(r'i\d+', raw)
        intent_id = match_id.group() if match_id else None

        # ✅ VALIDATION
        if intent_id and intent_id not in intentions_valides:
            print(f"⚠️  LLM Fallback a retourné {intent_id} mais il n'existe pas !")
            intent_id = None

        if intent_id is None:
            print(f"⚠️  LLM Fallback n'a pas trouvé d'ID valide")
            return None, duree

        print(f"✅ LLM Fallback choisi : {intent_id} (en {duree}s)")
        return intent_id, duree

    except Exception as e:
        print(f"❌ Erreur LLM Fallback : {e}")
        return None, 0


# ============================================
# PHASE 7 : TROUVER L'INTENTION COMPLÈTE
# ============================================
def get_intent_by_id(intent_id, intentions):
    for i in intentions:
        if i["id"] == intent_id:
            return {
                "id":          i["id"],
                "description": i["description"],
                "services":    i["services"],
                "QoS":         i["QoS"],
                "weight":      i.get("weight", 0),
                "score":       None
            }
    return None


# ============================================
# PHASE 8 : VALIDATION
# ============================================
def valider_intent(intent_result, dataset):
    print(f"\n🔎 Validation de l'intent...")

    services_dataset = [s["id"] for s in dataset["services"]]
    erreurs = []

    for sid in intent_result["services"]:
        if sid not in services_dataset:
            erreurs.append(f"Service {sid} inexistant !")

    qos = intent_result["QoS"]
    if qos.get("latency", 0) <= 0:
        erreurs.append("Latency QoS invalide !")
    if qos.get("bandwidth", 0) <= 0:
        erreurs.append("Bandwidth QoS invalide !")

    if erreurs:
        print(f"❌ Validation échouée :")
        for e in erreurs:
            print(f"   → {e}")
        return False

    print(f"✅ Validation OK !")
    return True


# ============================================
# PHASE 9 : SAUVEGARDER
# ============================================
def sauvegarder_intent(resume, intent_result, source, duree_rag, duree_llm=None):
    data = {
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resume":      resume,
        "source":      source,
        "duree_rag_s": duree_rag,
        "duree_llm_s": duree_llm,
        "intent": {
            "id":          intent_result["id"],
            "description": intent_result["description"],
            "services":    intent_result["services"],
            "QoS":         intent_result["QoS"],
            "weight":      intent_result["weight"],
            "score_rag":   intent_result.get("score")
        }
    }

    # 🔴 CORRECTION : nom unique généré ICI à chaque appel
    filename = f"intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n💾 Intent sauvegardé dans {filename} ✅")


# ============================================
# PROGRAMME PRINCIPAL V3 CORRIGÉ
# ============================================
if __name__ == "__main__":
   

    print("=" * 60)
    print("🚀 MATCHING RAG V3 : Cache des embeddings")
    print(f"   Threshold RAG direct : {THRESHOLD}")
    print(f"   Dominance gap        : {DOMINANCE_GAP}")
    print(f"   Top-K candidats      : {TOP_K}")
    print(f"   Embed model          : {EMBED_MODEL}")
    print(f"   LLM model            : {LLM_MODEL}")
    print("=" * 60)

    try:
        initialiser_cache()

        dataset    = charger_dataset()
        intentions = dataset["intentions"]

        # ✅ Liste des IDs valides pour validation
        intentions_valides = {i["id"] for i in intentions}

        resume     = lire_resume()

        intent_result = None
        duree_llm     = None

        # ÉTAPE 1 : RAG Top-K
        top_k, decision, duree_rag = rag_top_k(resume, intentions)

        # ÉTAPE 2 : Décision selon résultat RAG
        if decision == "rag_direct":
            intent_result = top_k[0]
            intent_result["score"] = top_k[0]["score"]
            source = "rag_direct"

        elif decision == "llm_rerank":
            intent_id, duree_llm = llm_rerank(resume, top_k, intentions_valides)
            if intent_id:
                intent_result = get_intent_by_id(intent_id, intentions)
            source = "llm_rerank"

        else:
            intent_id, duree_llm = llm_fallback(resume, intentions, intentions_valides)
            if intent_id:
                intent_result = get_intent_by_id(intent_id, intentions)
            source = "llm_fallback"

        # ÉTAPE 3 : Validation + Sauvegarde
        if intent_result is not None:
            valide = valider_intent(intent_result, dataset)
            if valide:
                sauvegarder_intent(resume, intent_result, source, duree_rag, duree_llm)
            else:
                print("\n❌ Intent invalide après validation !")
        else:
            print("\n❌ Aucune intention trouvée !")

        # Résumé final
        print("\n" + "=" * 60)
        print(f"📊 RÉSUMÉ FINAL :")
        print(f"   Décision      : {source}")
        print(f"   Durée RAG     : {duree_rag}s")
        print(f"   Durée LLM     : {duree_llm}s")
        if intent_result:
            print(f"   Intent trouvé : {intent_result['id']}")
            print(f"   Description   : {intent_result['description']}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Erreur globale : {e}")
        import traceback
        traceback.print_exc()