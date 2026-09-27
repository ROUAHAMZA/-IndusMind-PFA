"""
Gestionnaire d'état persistant pour le système de placement.
Sauvegarde et restaure l'état des services déployés et des capacités.
"""

import json
import os
from datetime import datetime

ETAT_FILE = "etat_systeme.json"


def sauvegarder_etat(services_deployes, capacites_actuelles):
    """
    Sauvegarde l'état actuel du système dans un fichier JSON.
    À appeler APRÈS chaque placement réussi.
    """
    etat = {
        "date_sauvegarde": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "services_deployes": services_deployes,
        "capacites": capacites_actuelles
    }
    
    with open(ETAT_FILE, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=4)
    
    print(f"💾 État sauvegardé dans {ETAT_FILE}")


def charger_etat(nodes_reference=None):
    """
    Charge l'état précédent du système depuis le fichier.
    À appeler AU DÉMARRAGE du programme.
    
    Si le fichier n'existe pas, retourne un état vierge.
    Si nodes_reference est fourni, vérifie la cohérence.
    """
    if not os.path.exists(ETAT_FILE):
        print("⚠️  Aucun état précédent trouvé. Démarrage à zéro.")
        return None, None
    
    print(f"\n📂 Chargement de l'état depuis {ETAT_FILE}...")
    
    with open(ETAT_FILE, "r", encoding="utf-8") as f:
        etat = json.load(f)
    
    services_deployes = etat.get("services_deployes", {})
    capacites = etat.get("capacites", {})
    
    print(f"✅ État chargé :")
    print(f"   → {len(services_deployes)} services déployés")
    print(f"   → Dernière sauvegarde : {etat.get('date_sauvegarde', 'inconnue')}")
    
    # Vérifier la cohérence si on a les nœuds de référence
    if nodes_reference:
        capacites = verifier_coherence(capacites, nodes_reference, services_deployes)
    
    return services_deployes, capacites


def verifier_coherence(capacites_chargees, nodes_reference, services_deployes):
    capacites_finales = {}
    
    # ✅ Travailler sur une COPIE
    services_deployes_local = dict(services_deployes)
    
    for node in nodes_reference:
        nid = node["id"]
        if nid in capacites_chargees:
            capacites_finales[nid] = capacites_chargees[nid]
        else:
            capacites_finales[nid] = {
                "CPU":  node["capacity"]["CPU"],
                "MEM":  node["capacity"]["MEM"],
                "DISK": node["capacity"]["DISK"],
                "BW":   node["capacity"]["BW"]
            }
            print(f"   → Nouveau nœud détecté : {nid}, capacités initialisées")
    
    # Vérifier les nœuds supprimés
    nodes_ids = {n["id"] for n in nodes_reference}
    for nid in list(capacites_chargees.keys()):
        if nid not in nodes_ids:
            print(f"   ⚠️  Nœud {nid} supprimé du dataset, services orphelins :")
            for sid, node_id in list(services_deployes_local.items()):
                if node_id == nid:
                    print(f"      → {sid} était sur {nid}, marqué comme à replacer")
                    del services_deployes_local[sid]
    
    # ✅ Mettre à jour le dict original proprement
    services_deployes.clear()
    services_deployes.update(services_deployes_local)
    
    return capacites_finales


def reinitialiser_etat():
    """
    Supprime le fichier d'état. À utiliser pour un redémarrage complet.
    """
    if os.path.exists(ETAT_FILE):
        os.remove(ETAT_FILE)
        print("🗑️  État réinitialisé (fichier supprimé)")
    else:
        print("ℹ️  Aucun état à réinitialiser")


# Test
if __name__ == "__main__":
    # Test de sauvegarde
    test_services = {"s1": "n3", "s2": "n5"}
    test_capacites = {
        "n1": {"CPU": 10, "MEM": 32},
        "n3": {"CPU": 2, "MEM": 8}
    }
    sauvegarder_etat(test_services, test_capacites)
    
    # Test de chargement
    services, caps = charger_etat()
    print(f"\nServices : {services}")
    print(f"Capacités : {caps}")