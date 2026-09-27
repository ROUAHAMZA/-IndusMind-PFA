"""
PLACEMENT MULTI-AGENT V3 : Avec persistance + RAG + validation renforcee
- Sauvegarde automatique apres chaque placement
- Restauration au demarrage
- RAG : memoire des placements historiques
- VALIDATION CUMULEE : verifie la somme des ressources par noeud
- APPRENTISSAGE APRES VALIDATION : n'apprend que les placements valides
"""

import json
import ollama
import re
import time
import os
from datetime import datetime

# ============================================
# IMPORT DU RAG PLACEMENT
# ============================================
from placement_knowledge import (
    trouver_patterns_similaires,
    formater_pattern_pour_llm,
    ajouter_pattern
)

# ============================================
# IMPORT DU GESTIONNAIRE D'ETAT
# ============================================
from etat_manager import (
    sauvegarder_etat,
    charger_etat,
    reinitialiser_etat,
    ETAT_FILE
)

# ============================================
# CONFIGURATION
# ============================================
DATASET_FILE = "dataset_test.json"
OUTPUT_FILE = "placement.json"
LLM_MODEL = "mistral"


# ============================================
# PHASE 1 : CHARGER LE DATASET
# ============================================
def charger_dataset(fichier=DATASET_FILE):
    print(f"\n📂 Chargement de {fichier}...")
    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Dataset charge :")
    print(f"   → {len(data['services'])} services")
    print(f"   → {len(data['nodes'])} noeuds")
    print(f"   → {len(data['intentions'])} intentions")
    return data


# ============================================
# PHASE 2 : INITIALISER L'ETAT
# ============================================
def initialiser_etat_avec_persistence(dataset):
    nodes = dataset["nodes"]

    services_deployes, capacites = charger_etat(nodes_reference=nodes)

    if services_deployes is None:
        print("\n🆕 Premier demarrage, initialisation a zero...")
        services_deployes = {}
        capacites = {}
        for node in nodes:
            nid = node["id"]
            capacites[nid] = {
                "CPU":  node["capacity"]["CPU"],
                "MEM":  node["capacity"]["MEM"],
                "DISK": node["capacity"]["DISK"],
                "BW":   node["capacity"]["BW"]
            }

    return services_deployes, capacites


# ============================================
# PHASE 3 : CALCULER LATENCE MOYENNE
# ============================================
def latence_moyenne(node_id, latency):
    lats = latency[node_id]
    return round(sum(lats) / len(lats), 2)


# ============================================
# PHASE 4 : AGENT 1 - RESOURCE MANAGER
# ============================================
def agent1_resource_manager(intention, services_data, nodes, capacites_actuelles, services_deployes):
    """
    Agent 1 : Resource Manager
    🔴 CORRECTION : Verifier si le 'reuse' est coherent avec l'etat reel
    """
    print(f"\n   🤖 AGENT 1 : Resource Manager")

    services_ids = intention["services"]
    resultats = []

    for sid in services_ids:
        service_info = None
        for s in services_data:
            if s["id"] == sid:
                service_info = s
                break

        if service_info is None:
            print(f"      ❌ Service {sid} introuvable !")
            continue

        res = service_info["resources"]

        # 🔴 CORRECTION : Verifier si 'reuse' est COHERENT
        if sid in services_deployes:
            noeud_existant = services_deployes[sid]

            # Verifier que le nœud existe dans les capacites actuelles
            if noeud_existant in capacites_actuelles:
                cap = capacites_actuelles[noeud_existant]

                # 🔴 CORRECTION : Trouver le nœud dans la LISTE (pas dictionnaire)
                node_info = next((n for n in nodes if n["id"] == noeud_existant), None)

                if node_info:
                    # 🔴 VERIFICATION CLE : Les ressources sont-elles VRAIMENT consommees ?
                    ressources_consommees = (
                        cap["CPU"] < node_info["capacity"]["CPU"] or
                        cap["MEM"] < node_info["capacity"]["MEM"] or
                        cap["DISK"] < node_info["capacity"]["DISK"] or
                        cap["BW"] < node_info["capacity"]["BW"]
                    )

                    if ressources_consommees:
                        # ✅ Vrai reuse : les ressources sont consommees
                        print(f"      ♻️  {sid} deja deploye sur {noeud_existant} → reutilisation")
                        resultats.append({
                            "service_id": sid,
                            "action": "reuse",
                            "node_id": noeud_existant,
                            "noeuds_compatibles": [noeud_existant]
                        })
                        continue
                    else:
                        # ❌ Faux reuse : le nœud a toutes ses ressources = etat vierge
                        print(f"      ⚠️  {sid} marque 'reuse' sur {noeud_existant} mais ressources a 100%")
                        print(f"      🚀 Forcer DEPLOY a la place")
                        # On ne fait PAS continue, on passe au deploy
                else:
                    # Nœud n'existe plus dans la liste
                    print(f"      ⚠️  Nœud {noeud_existant} introuvable dans la liste")
                    del services_deployes[sid]
            else:
                # Nœud n'existe plus dans les capacites
                print(f"      ⚠️  Nœud {noeud_existant} supprime, {sid} a replacer")
                del services_deployes[sid]

        # 🔴 Si on arrive ici, c'est un DEPLOY (pas de reuse)
        # Trouver les nœuds compatibles
        noeuds_compatibles = []
        for node in nodes:
            nid = node["id"]
            cap = capacites_actuelles[nid]

            cpu_ok  = cap["CPU"]  >= res["CPU"]
            mem_ok  = cap["MEM"]  >= res["MEM"]
            disk_ok = cap["DISK"] >= res["DISK"]
            bw_ok   = cap["BW"]   >= res["BW"]

            if cpu_ok and mem_ok and disk_ok and bw_ok:
                noeuds_compatibles.append(nid)

        print(f"      🔍 {sid} : {len(noeuds_compatibles)} nœuds compatibles → {noeuds_compatibles}")

        resultats.append({
            "service_id": sid,
            "action": "deploy",  # 🔴 FORCER DEPLOY
            "node_id": None,
            "noeuds_compatibles": noeuds_compatibles,
            "ressources": res
        })

    return resultats


# ============================================
# PARAMÈTRE GLOBAL (à placer en haut du fichier)
# ============================================
ALPHA = 0.5   # poids latence  (0.0 → 1.0)
BETA  = 0.5   # poids charge   (0.0 → 1.0)
# ALPHA + BETA doit toujours = 1.0


# ============================================
# PHASE 5 : AGENT 2 - QoS ANALYZER (modifié)
# ============================================
def agent2_qos_analyzer(intention, resultats_agent1, latency, capacites_actuelles, nodes):
    print(f"\n   🤖 AGENT 2 : QoS Analyzer")

    qos = intention["QoS"]
    latency_max = qos["latency"]
    resultats_qos = []

    # Calcul des capacités MAX (pour normaliser la charge)
    capacites_max = {n["id"]: n["capacity"] for n in nodes}

    for r in resultats_agent1:
        sid = r["service_id"]
        noeuds_valides = []

        for nid in r["noeuds_compatibles"]:
            lat_moy = latence_moyenne(nid, latency)

            if lat_moy > latency_max:
                continue  # éliminé par QoS latence

            # --- NOUVEAU : calcul de la charge actuelle du nœud ---
            cap_rest = capacites_actuelles[nid]
            cap_max  = capacites_max[nid]

            cpu_charge  = 1 - (cap_rest["CPU"]  / cap_max["CPU"])  if cap_max["CPU"]  else 0
            mem_charge  = 1 - (cap_rest["MEM"]  / cap_max["MEM"])  if cap_max["MEM"]  else 0
            bw_charge   = 1 - (cap_rest["BW"]   / cap_max["BW"])   if cap_max["BW"]   else 0
            disk_charge = 1 - (cap_rest["DISK"] / cap_max["DISK"]) if cap_max["DISK"] else 0

            # Charge = pire ressource (goulot d'étranglement)
            charge_noeud = max(cpu_charge, mem_charge, bw_charge, disk_charge)

            # Latence normalisée entre 0 et 1
            lat_norm = lat_moy / latency_max

            # --- Score composite : plus bas = meilleur ---
            score = ALPHA * lat_norm + BETA * charge_noeud

            noeuds_valides.append({
                "node_id":    nid,
                "latence_moy": lat_moy,
                "charge":     round(charge_noeud, 3),
                "score":      round(score, 4)    # <-- nouveau champ
            })

        # Tri par score composite croissant (plus bas = meilleur)
        noeuds_valides.sort(key=lambda x: x["score"])

        print(f"      📊 {sid} : {len(noeuds_valides)} nœuds valides "
              f"(latence ≤ {latency_max}ms)")
        if noeuds_valides:
            top3 = noeuds_valides[:3]
            for n in top3:
                print(f"         {n['node_id']} | lat={n['latence_moy']}ms "
                      f"charge={n['charge']*100:.0f}% score={n['score']:.3f}")

        resultats_qos.append({
            "service_id":    sid,
            "action":        r["action"],
            "node_id":       r["node_id"],
            "noeuds_valides": noeuds_valides,
            "ressources":    r.get("ressources", {})
        })

    return resultats_qos


# ============================================
# RETRY + REPAIR JSON
# ============================================
def reparer_json(raw):
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    code_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    raw_clean = raw.replace("'", '"').replace('\n', ' ').strip()
    json_match2 = re.search(r'\{.*\}', raw_clean, re.DOTALL)
    if json_match2:
        try:
            return json.loads(json_match2.group())
        except json.JSONDecodeError:
            pass

    return None


# ============================================
# PHASE 6 : AGENT 3 - PLACEMENT MASTER
# ============================================
def agent3_placement_master(intention, resultats_agent2, capacites_actuelles, services_deployes, max_retries=2):
    """
    Agent 3 avec RAG : Utilise les placements historiques pour guider le LLM.
    """
    print(f"\n   🤖 AGENT 3 : Placement Master (mistral) + RAG")

    # RAG - Rechercher des patterns similaires
    services_ids = intention["services"]
    patterns_similaires = trouver_patterns_similaires(
        intention["description"],
        services_ids,
        top_k=2
    )
    contexte_rag = formater_pattern_pour_llm(patterns_similaires)

    # Preparer le contexte des services
    contexte_services = ""
    for r in resultats_agent2:
        sid = r["service_id"]
        action = r["action"]

        if action == "reuse":
            contexte_services += f"""
Service {sid}:
  - Action recommandee: REUSE (already deployed on {r['node_id']})
  - No redeployment needed
"""
        else:
            noeuds_txt = ""
            for n in r["noeuds_valides"][:5]:
                cap = capacites_actuelles[n["node_id"]]
                noeuds_txt += (f"    * {n['node_id']} : "
                               f"CPU={cap['CPU']}, MEM={cap['MEM']}, "
                               f"latence={n['latence_moy']}ms\n")

            if not noeuds_txt:
                noeuds_txt = "    * No valid node found\n"

            res = r.get("ressources", {})
            contexte_services += f"""
Service {sid} (needs CPU={res.get('CPU',0)}, MEM={res.get('MEM',0)},
               DISK={res.get('DISK',0)}, BW={res.get('BW',0)}):
  - Action: DEPLOY on best node
  - Valid nodes (resources + QoS OK):
{noeuds_txt}"""

    debut = time.time()

    for tentative in range(1, max_retries + 1):
        print(f"      🔄 Tentative {tentative}/{max_retries}...")

        try:
            reponse = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert edge computing placement system.
Your job is to assign services to the best nodes.
Rules:
- REUSE already deployed services (no redeployment)
- Choose nodes with lowest latency AND best load balance
- Reply ONLY with valid JSON, nothing else
- JSON format: {"placements": [{"service_id": "s1", "node_id": "n3", "action": "reuse/deploy"}]}
- IMPORTANT: Include ALL services in your response, no exceptions"""
                    },
                    {
                        "role": "user",
                        "content": f"""Intent: "{intention['description']}"\nQoS requirements: latency<<={intention['QoS']['latency']}ms,\n                  bandwidth>={intention['QoS']['bandwidth']}\n\n{contexte_rag}\n\nServices to place:\n{contexte_services}\n\nCRITICAL: You MUST include ALL services listed above in your JSON response.\nServices required: {intention['services']}\n\nReply ONLY with JSON:"""
                    }
                ]
            )

            raw = reponse["message"]["content"].strip()
            plan = reparer_json(raw)

            if plan is None:
                print(f"      ❌ Tentative {tentative} : JSON invalide")
                if tentative < max_retries:
                    print(f"      ↩️  Retry...")
                continue

            services_dans_plan = {p["service_id"] for p in plan.get("placements", [])}
            services_manquants = [s for s in intention["services"] if s not in services_dans_plan]

            if services_manquants:
                print(f"      ⚠️  Tentative {tentative} : services manquants {services_manquants}")
                if tentative < max_retries:
                    print(f"      ↩️  Retry...")
                continue

            fin = time.time()
            duree = round(fin - debut, 2)
            print(f"      ✅ LLM plan recu (tentative {tentative}, {duree}s)")

            # ✅ NE PAS sauvegarder le pattern ici ! On le fera apres validation
            return plan, duree

        except Exception as e:
            print(f"      ❌ Erreur LLM tentative {tentative} : {e}")
            if tentative < max_retries:
                continue
            return None, round(time.time() - debut, 2)

    fin = time.time()
    duree = round(fin - debut, 2)
    print(f"      ❌ Toutes les tentatives echouees → Greedy Fallback")
    return None, duree


# ============================================
# PHASE 7 : VALIDATION PYTHON (AVEC CUMUL)
# ============================================
def valider_placement(plan_llm, resultats_agent2, capacites_actuelles, intention, latency, services_deployes):
    """
    VERSION CORRIGEE : Verifie la somme cumulee des ressources sur chaque noeud
    🔴 CORRECTION : S'assurer que tous les placements ont un champ 'action'
    """
    print(f"\n   🔎 Validation Python (avec cumul des ressources)...")

    if plan_llm is None:
        print(f"      ❌ Plan LLM vide → fallback greedy")
        return False, []

    placements = plan_llm.get("placements", [])
    erreurs = []
    valide = True

    # ✅ SIMULATION : copie des capacites pour verifier le cumul
    capacites_simulees = {nid: dict(cap) for nid, cap in capacites_actuelles.items()}

    map_agent2 = {r["service_id"]: r for r in resultats_agent2}

    for p in placements:
        sid = p.get("service_id")
        nid = p.get("node_id")
        action = p.get("action", "deploy")
        if "/" in str(action):
            action = "reuse" if sid in services_deployes else "deploy"
            p["action"] = action
            print(f"      🔧 {sid} : action normalisée → '{action}'")

        # 🔴 CORRECTION CLE : S'assurer que 'action' est dans le placement
        if "action" not in p:
            p["action"] = action
            print(f"      🔧 {sid} : action manquante → forcer '{action}'")

        if sid not in map_agent2:
            erreurs.append(f"{sid} : service inconnu")
            valide = False
            continue

        r = map_agent2[sid]

        if action == "reuse":
            if not nid or nid == "":
                if sid in services_deployes:
                    p["node_id"] = services_deployes[sid]
                    p["action"] = "reuse"  # 🔴 S'assurer que action est bien "reuse"
                    print(f"      🔧 {sid} → corrige : {services_deployes[sid]}")
                else:
                    erreurs.append(f"{sid} : reuse mais node_id vide")
                    valide = False
            else:
                print(f"      ✅ {sid} → reutilisation sur {nid} OK")
            continue

        # Verifier que le noeud est dans la liste des valides
        noeuds_valides_ids = [n["node_id"] for n in r["noeuds_valides"]]
        if nid not in noeuds_valides_ids:
            erreurs.append(f"{sid} → {nid} : noeud non valide !")
            valide = False
            continue

        # ✅ VERIFICATION CUMULEE : verifier les ressources restantes apres les placements precedents
        res = r.get("ressources", {})
        cap = capacites_simulees[nid]  # Utilise les capacites simulees, pas les reelles

        if cap["CPU"] < res.get("CPU", 0):
            erreurs.append(f"{sid} → {nid} : CPU insuffisant (reste {cap['CPU']}, besoin {res.get('CPU',0)})")
            valide = False
        if cap["MEM"] < res.get("MEM", 0):
            erreurs.append(f"{sid} → {nid} : MEM insuffisante (reste {cap['MEM']}, besoin {res.get('MEM',0)})")
            valide = False
        if cap["DISK"] < res.get("DISK", 0):
            erreurs.append(f"{sid} → {nid} : DISK insuffisant (reste {cap['DISK']}, besoin {res.get('DISK',0)})")
            valide = False
        if cap["BW"] < res.get("BW", 0):
            erreurs.append(f"{sid} → {nid} : BW insuffisante (reste {cap['BW']}, besoin {res.get('BW',0)})")
            valide = False

        # ✅ SIMULER la consommation pour les placements suivants
        if valide:
            cap["CPU"] -= res.get("CPU", 0)
            cap["MEM"] -= res.get("MEM", 0)
            cap["DISK"] -= res.get("DISK", 0)
            cap["BW"] -= res.get("BW", 0)
            print(f"      ✅ {sid} → {nid} : validation OK (capacites restantes simulees)")

    # Verifier que tous les services sont presents
    services_places = {p["service_id"] for p in placements}
    for sid in intention["services"]:
        if sid not in services_places:
            print(f"      ⚠️  {sid} manquant → ajout automatique")
            if sid in services_deployes:
                placements.append({
                    "service_id": sid,
                    "node_id": services_deployes[sid],
                    "action": "reuse"
                })
            else:
                erreurs.append(f"{sid} : manquant et non deploye")
                valide = False

    plan_llm["placements"] = placements

    if erreurs:
        print(f"      ❌ Erreurs :")
        for e in erreurs:
            print(f"         → {e}")

    return valide, erreurs


# ============================================
# PHASE 8 : GREEDY FALLBACK (AVEC VALIDATION)
# ============================================
def greedy_fallback(intention, services_data, nodes, capacites_actuelles, latency, services_deployes):
    print(f"\n   🔄 Greedy Fallback...")

    qos = intention["QoS"]
    services_ids = intention["services"]
    placements = []

    # ✅ Copie des capacites pour simuler
    capacites_simulees = {nid: dict(cap) for nid, cap in capacites_actuelles.items()}

    for sid in services_ids:
        if sid in services_deployes:
            print(f"      ♻️  {sid} reutilise sur {services_deployes[sid]}")
            placements.append({
                "service_id": sid,
                "node_id": services_deployes[sid],
                "action": "reuse"
            })
            continue

        res = None
        for s in services_data:
            if s["id"] == sid:
                res = s["resources"]
                break

        if res is None:
            continue

        meilleur = None
        meilleur_lat = float('inf')

        for node in nodes:
            nid = node["id"]
            cap = capacites_simulees[nid]

            cpu_ok = cap["CPU"] >= res["CPU"]
            mem_ok = cap["MEM"] >= res["MEM"]
            disk_ok = cap["DISK"] >= res["DISK"]
            bw_ok = cap["BW"] >= res["BW"]

            if cpu_ok and mem_ok and disk_ok and bw_ok:
                lat = latence_moyenne(nid, latency)
                if lat <= qos["latency"] and lat < meilleur_lat:
                    meilleur = nid
                    meilleur_lat = lat

        if meilleur:
            print(f"      ✅ {sid} → {meilleur} (latence={meilleur_lat}ms)")
            placements.append({
                "service_id": sid,
                "node_id": meilleur,
                "action": "deploy"
            })
            # ✅ Simuler la consommation
            capacites_simulees[meilleur]["CPU"] -= res["CPU"]
            capacites_simulees[meilleur]["MEM"] -= res["MEM"]
            capacites_simulees[meilleur]["DISK"] -= res["DISK"]
            capacites_simulees[meilleur]["BW"] -= res["BW"]
        else:
            print(f"      ❌ {sid} → aucun noeud disponible !")
            placements.append({
                "service_id": sid,
                "node_id": None,
                "action": "failed"
            })

    return placements


# ============================================
# PHASE 9 : RESERVER LES RESSOURCES + SAUVEGARDER ETAT
# ============================================
def reserver_ressources(placements, services_data, capacites_actuelles, services_deployes):
    for p in placements:
        sid = p["service_id"]
        nid = p["node_id"]
        action = p.get("action", "deploy")  # 🔴 CORRECTION : Utiliser .get() avec default

        if nid is None or action == "reuse":
            continue

        for s in services_data:
            if s["id"] == sid:
                res = s["resources"]
                # ✅ Double verification avant de deduire
                cap = capacites_actuelles[nid]
                if cap["CPU"] < res["CPU"] or cap["MEM"] < res["MEM"] or \
                   cap["DISK"] < res["DISK"] or cap["BW"] < res["BW"]:
                    print(f"   ⚠️  {sid} → {nid} : ressources insuffisantes au moment de la reservation !")
                    continue

                capacites_actuelles[nid]["CPU"] -= res["CPU"]
                capacites_actuelles[nid]["MEM"] -= res["MEM"]
                capacites_actuelles[nid]["DISK"] -= res["DISK"]
                capacites_actuelles[nid]["BW"] -= res["BW"]
                services_deployes[sid] = nid
                break

    sauvegarder_etat(services_deployes, capacites_actuelles)
    print("   💾 Etat sauvegarde automatiquement")


# ============================================
# PHASE 10 : PLACEMENT COMPLET (1 intention)
# ============================================
def placer_intention(intention, dataset, capacites_actuelles, services_deployes):
    services_data = dataset["services"]
    nodes = dataset["nodes"]
    latency = dataset["latency"]

    print(f"\n{'─'*55}")
    print(f"📋 Placement : {intention['id']} | {intention['description'][:45]}...")
    print(f"   Services : {intention['services']} | QoS : {intention['QoS']}")

    debut_total = time.time()
    source = None

    resultats_a1 = agent1_resource_manager(
        intention, services_data, nodes,
        capacites_actuelles, services_deployes
    )

    resultats_a2 = agent2_qos_analyzer(intention, resultats_a1, latency,
                                    capacites_actuelles, nodes)

    tous_valides = all(
        r["action"] == "reuse" or len(r["noeuds_valides"]) > 0
        for r in resultats_a2
    )

    placements_finaux = None
    duree_llm = None

    if tous_valides:
        plan_llm, duree_llm = agent3_placement_master(
            intention, resultats_a2,
            capacites_actuelles, services_deployes
        )

        valide, erreurs = valider_placement(
            plan_llm, resultats_a2,
            capacites_actuelles, intention, latency, services_deployes
        )

        if valide and plan_llm:
            placements_finaux = plan_llm["placements"]
            source = "multiagent"

            # ✅ APPRENTISSAGE APRES VALIDATION : sauvegarder le pattern reussi
            ajouter_pattern(
                intention["description"],
                intention["services"],
                placements_finaux
            )

        else:
            print(f"\n   ⚠️ Validation echouee → Greedy Fallback")
            placements_finaux = greedy_fallback(
                intention, services_data, nodes,
                capacites_actuelles, latency, services_deployes
            )
            source = "greedy_fallback"
    else:
        print(f"\n   ⚠️ Noeuds insuffisants → Greedy Fallback direct")
        placements_finaux = greedy_fallback(
            intention, services_data, nodes,
            capacites_actuelles, latency, services_deployes
        )
        source = "greedy_fallback"

    # Reserver + sauvegarder l'etat
    reserver_ressources(
        placements_finaux, services_data,
        capacites_actuelles, services_deployes
    )

    fin_total = time.time()
    duree_total = round(fin_total - debut_total, 2)

    succes = all(p["node_id"] is not None for p in placements_finaux)

    print(f"\n   {'✅' if succes else '❌'} Resultat : {source} | {duree_total}s")
    for p in placements_finaux:
        action_icon = "♻️" if p["action"] == "reuse" else "🚀"
        print(f"   {action_icon} {p['service_id']} → {p['node_id']} ({p['action']})")

    return {
        "intention_id": intention["id"],
        "description": intention["description"],
        "services": intention["services"],
        "source": source,
        "succes": succes,
        "duree_s": duree_total,
        "duree_llm_s": duree_llm,
        "placements": placements_finaux
    }


# ============================================
# PHASE 11 : TRAITER TOUTES LES INTENTIONS
# ============================================
def placer_toutes_intentions(dataset):
    print(f"\n{'='*55}")
    print("🚀 PLACEMENT MULTI-AGENT V3 - PERSISTANCE + RAG + VALIDATION CUMULEE")
    print(f"   LLM : {LLM_MODEL}")
    print(f"{'='*55}")

    services_deployes, capacites_actuelles = initialiser_etat_avec_persistence(dataset)

    intentions = sorted(
        dataset["intentions"],
        key=lambda x: x.get("weight", 0),
        reverse=True
    )

    resultats = []
    places = 0
    non_places = 0
    reuses = 0
    debut_total = time.time()

    for intention in intentions:
        try:
            resultat = placer_intention(
                intention, dataset,
                capacites_actuelles, services_deployes
            )
            resultats.append(resultat)

            if resultat["succes"]:
                places += 1
                reuses += sum(
                    1 for p in resultat["placements"]
                    if p["action"] == "reuse"
                )
            else:
                non_places += 1

        except Exception as e:
            print(f"❌ Erreur sur intention {intention['id']} : {e}")
            non_places += 1

    fin_total = time.time()
    temps_total = round(fin_total - debut_total, 2)
    taux = round(places / len(intentions) * 100, 2) if intentions else 0

    print(f"\n{'='*55}")
    print(f"📊 RESUME FINAL :")
    print(f"   Intentions placees     : {places}/{len(intentions)}")
    print(f"   Intentions non placees : {non_places}/{len(intentions)}")
    print(f"   Taux de placement      : {taux}%")
    print(f"   Reutilisations         : {reuses} (evite redeploiement)")
    print(f"   Temps total            : {temps_total}s")
    print(f"   Etat sauvegarde dans   : {ETAT_FILE}")
    print(f"{'='*55}")

    return resultats, taux, temps_total, reuses


# ============================================
# PHASE 12 : SAUVEGARDER RESULTATS
# ============================================
def sauvegarder_resultats(resultats, taux, temps_total, reuses):
    data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modele": LLM_MODEL,
        "algorithme": "Multi-Agent V3 (persistance + RAG + validation cumulee)",
        "taux_placement": f"{taux}%",
        "temps_total_s": temps_total,
        "reuses_total": reuses,
        "resultats": resultats
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n💾 Resultats sauvegardes dans {OUTPUT_FILE} ✅")


# ============================================
# PROGRAMME PRINCIPAL V3
# ============================================
if __name__ == "__main__":
    try:
        dataset = charger_dataset()

        resultats, taux, temps_total, reuses = placer_toutes_intentions(dataset)

        sauvegarder_resultats(resultats, taux, temps_total, reuses)

    except Exception as e:
        print(f"❌ Erreur globale : {e}")
        import traceback
        traceback.print_exc()
        