"""
Gestionnaire de la base de connaissances des placements (RAG Placement).
Stocke et récupère les placements historiques réussis.
CORRECTION : Utilise les embeddings (nomic-embed-text) au lieu du simple overlap de mots.
"""

import json
import os
import numpy as np
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

KNOWLEDGE_FILE = "placement_knowledge.json"
EMBED_MODEL    = "nomic-embed-text"


# ============================================
# UTILITAIRES EMBEDDINGS
# ============================================
def _get_embedding(texte):
    """Génère un embedding pour un texte donné."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=texte)
    return np.array(response["embedding"])


def _cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot   = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _similarite_mots(texte1, texte2):
    """Similarité textuelle par overlap de mots — utilisée comme fallback."""
    words1 = set(texte1.lower().split())
    words2 = set(texte2.lower().split())
    return len(words1 & words2) / max(len(words1), len(words2), 1)


# ============================================
# CHARGEMENT / SAUVEGARDE
# ============================================
def charger_knowledge():
    """
    Charge la base de connaissances des placements.
    Si le fichier n'existe pas, retourne une structure vide.
    """
    if not os.path.exists(KNOWLEDGE_FILE):
        print("📚 Base de connaissances vide (premier démarrage)")
        return {"patterns": []}

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📚 Base de connaissances chargée : {len(data['patterns'])} patterns")
    return data


def sauvegarder_knowledge(data):
    """
    Sauvegarde la base de connaissances dans le fichier.
    """
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Base de connaissances sauvegardée ({len(data['patterns'])} patterns)")


# ============================================
# AJOUT D'UN PATTERN
# ============================================
def ajouter_pattern(intent_description, services, placements, score=1.0):
    """
    Ajoute un nouveau pattern de placement réussi à la base.
    Si un pattern identique existe déjà, met à jour sa confiance.
    """
    data = charger_knowledge()

    # Vérifier si un pattern similaire existe déjà
    for p in data["patterns"]:
        if p["services"] == services and p["intent_description"] == intent_description:
            p["score"]       = min(p["score"] + 0.1, 1.0)
            p["date"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            p["occurrences"] = p.get("occurrences", 1) + 1
            sauvegarder_knowledge(data)
            print(f"🔄 Pattern mis à jour (occurrences : {p['occurrences']})")
            return

    # Nouveau pattern
    pattern = {
        "id":                 f"p{len(data['patterns']) + 1}",
        "intent_description": intent_description,
        "services":           services,
        "placements":         placements,
        "score":              score,
        "date":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "occurrences":        1
    }

    data["patterns"].append(pattern)
    sauvegarder_knowledge(data)
    print(f"✅ Nouveau pattern ajouté : {pattern['id']}")


# ============================================
# RECHERCHE DE PATTERNS SIMILAIRES
# ============================================
def trouver_patterns_similaires(intent_description, services, top_k=2):
    """
    Trouve les patterns les plus similaires à l'intention actuelle.
    CORRECTION : Utilise les embeddings pour la similarité sémantique.
    Fallback automatique sur overlap de mots si Ollama indisponible.
    """
    data = charger_knowledge()

    if not data["patterns"]:
        return []

    # Tenter de générer l'embedding de l'input
    use_embeddings = False
    input_vec      = None

    if OLLAMA_AVAILABLE:
        try:
            input_vec      = _get_embedding(intent_description)
            use_embeddings = True
            print("   🔢 Similarité par embeddings")
        except Exception as e:
            print(f"   ⚠️  Embedding indisponible, fallback mots : {e}")
    else:
        print("   ⚠️  Ollama non disponible, fallback mots")

    scores = []

    for pattern in data["patterns"]:
        services_match = set(pattern["services"]) == set(services)

        # Calculer la similarité textuelle
        if use_embeddings:
            try:
                pattern_vec = _get_embedding(pattern["intent_description"])
                text_sim    = _cosine_similarity(input_vec, pattern_vec)
            except Exception:
                # Fallback mots pour ce pattern uniquement
                text_sim = _similarite_mots(intent_description, pattern["intent_description"])
        else:
            text_sim = _similarite_mots(intent_description, pattern["intent_description"])

        # Score combiné : services identiques = forte priorité
        if services_match:
            combined_score = 0.7 + 0.3 * text_sim
        else:
            combined_score = 0.3 * text_sim

        scores.append({
            "pattern": pattern,
            "score":   round(combined_score, 4)
        })

    # Trier par score décroissant
    scores.sort(key=lambda x: x["score"], reverse=True)

    # Retourner les top-k au-dessus du seuil
    resultats = [s for s in scores[:top_k] if s["score"] > 0.5]

    if resultats:
        print(f"\n🔍 Patterns similaires trouvés : {len(resultats)}")
        for i, r in enumerate(resultats):
            method = "embed" if use_embeddings else "mots"
            print(f"   {i+1}. Score={r['score']:.2f} [{method}] | "
                  f"{r['pattern']['intent_description'][:50]}...")

    return resultats


# ============================================
# FORMATAGE POUR LE LLM
# ============================================
def formater_pattern_pour_llm(patterns):
    """
    Formate les patterns trouvés pour les inclure dans le prompt du LLM.
    """
    if not patterns:
        return ""

    texte = "\n📋 PLACEMENTS HISTORIQUES SIMILAIRES (à utiliser comme référence) :\n"

    for i, p in enumerate(patterns):
        pat    = p["pattern"]
        texte += f"\nPattern {i+1} (confiance: {pat['score']:.2f}) :\n"
        texte += f"Description : {pat['intent_description']}\n"
        texte += f"Services    : {pat['services']}\n"
        texte += f"Placement réussi :\n"
        for pl in pat["placements"]:
            texte += f"  - {pl['service_id']} → {pl['node_id']}\n"

    texte += "\n💡 CONSEIL : Réutilise ces placements si les nœuds sont toujours disponibles.\n"

    return texte


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    # Test d'ajout
    ajouter_pattern(
        "Perform full diagnostics",
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        [
            {"service_id": "s1", "node_id": "n1"},
            {"service_id": "s2", "node_id": "n1"},
            {"service_id": "s3", "node_id": "n2"}
        ]
    )

    ajouter_pattern(
        "Run complete system check",
        ["s1", "s2", "s3"],
        [
            {"service_id": "s1", "node_id": "n2"},
            {"service_id": "s2", "node_id": "n3"},
            {"service_id": "s3", "node_id": "n2"}
        ]
    )

    # Test de recherche
    print("\n--- Test recherche avec embeddings ---")
    similaires = trouver_patterns_similaires(
        "Perform full system diagnostics",
        ["s1", "s2", "s3", "s4", "s5", "s6"]
    )

    print("\n" + formater_pattern_pour_llm(similaires))