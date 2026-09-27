"""
Cache Manager pour les embeddings du dataset.
Génère et maintient un cache des embeddings des intentions.
"""

import json
import os
import numpy as np
import ollama
from datetime import datetime

CACHE_FILE = "embeddings_cache.json"
DATASET_FILE = "dataset_4.json"
EMBED_MODEL = "nomic-embed-text"


def generer_cache_embeddings():
    """
    Génère le cache des embeddings pour toutes les intentions du dataset.
    À appeler quand le dataset change.
    """
    print("🔄 Génération du cache des embeddings...")

    # Charger le dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    intentions = data["intentions"]
    cache = {
        "date_generation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_file": DATASET_FILE,
        "embeddings": {}
    }

    for intention in intentions:
        intent_id = intention["id"]
        description = intention["description"]

        print(f"   Embedding {intent_id} : {description[:40]}...")

        response = ollama.embeddings(model=EMBED_MODEL, prompt=description)
        embedding = response["embedding"]

        cache["embeddings"][intent_id] = {
            "description": description,
            "embedding": embedding
        }

    # Sauvegarder
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"✅ Cache généré : {len(intentions)} intentions dans {CACHE_FILE}")


def charger_cache_embeddings():
    """
    Charge le cache des embeddings en mémoire.
    Retourne un dictionnaire {intent_id: numpy_array}
    """
    if not os.path.exists(CACHE_FILE):
        print("⚠️  Cache inexistant, génération en cours...")
        generer_cache_embeddings()

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Convertir en numpy arrays pour calcul rapide
    embeddings_dict = {}
    for intent_id, data in cache["embeddings"].items():
        embeddings_dict[intent_id] = np.array(data["embedding"])

    print(f"✅ Cache chargé : {len(embeddings_dict)} embeddings")
    return embeddings_dict


def verifier_cache_a_jour():
    """
    Vérifie si le cache est à jour par rapport au dataset.
    Retourne True si OK, False si besoin de régénérer.
    """
    if not os.path.exists(CACHE_FILE):
        return False

    # Vérifier la date du dataset
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset {DATASET_FILE} introuvable !")

    dataset_mtime = os.path.getmtime(DATASET_FILE)
    cache_mtime = os.path.getmtime(CACHE_FILE)

    # Si le dataset est plus récent que le cache → régénérer
    if dataset_mtime > cache_mtime:
        print("🔄 Dataset modifié depuis le dernier cache")
        return False

    # Vérifier que tous les IDs du dataset sont dans le cache
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    dataset_ids = {i["id"] for i in data["intentions"]}
    cache_ids = set(cache["embeddings"].keys())

    if dataset_ids != cache_ids:
        print(f"🔄 IDs différents : dataset={len(dataset_ids)}, cache={len(cache_ids)}")
        return False

    print("✅ Cache à jour")
    return True


# Test
if __name__ == "__main__":
    generer_cache_embeddings()
    embeddings = charger_cache_embeddings()
    print(f"\nTest : {len(embeddings)} embeddings chargés")