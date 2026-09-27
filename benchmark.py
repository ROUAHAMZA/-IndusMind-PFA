"""
IndusMind - Calcul automatique des métriques pour rapport PFE
Lit les fichiers JSON générés par le système et calcule toutes les métriques.
"""

import json
import os
import glob
from datetime import datetime


# ============================================
# COULEURS CONSOLE
# ============================================
G = "\033[92m"  # vert
B = "\033[94m"  # bleu
Y = "\033[93m"  # jaune
R = "\033[91m"  # rouge
W = "\033[0m"   # reset


def titre(texte):
    print(f"\n{B}{'='*55}")
    print(f"  {texte}")
    print(f"{'='*55}{W}")


def sous_titre(texte):
    print(f"\n{Y}  ── {texte} ──{W}")


# ============================================
# 1. MÉTRIQUES RAG MATCHING
# ============================================
def metriques_rag_matching():
    titre("MODULE 1 : RAG Matching")

    fichiers = glob.glob("intent_*.json") 

    if not fichiers:
        print(f"  {R}Aucun fichier intent_*.json trouvé{W}")
        print(f"  → Lance le pipeline au moins une fois pour générer des données")
        return {}

    durees_rag   = []
    durees_llm   = []
    methodes     = {"rag_direct": 0, "llm_rerank": 0, "llm_fallback": 0, "keyword_fallback": 0}
    scores       = []

    for f in fichiers:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            source = data.get("source", "inconnu")
            if source in methodes:
                methodes[source] += 1

            duree_rag = data.get("duree_rag_s")
            if duree_rag:
                durees_rag.append(float(duree_rag))

            duree_llm = data.get("duree_llm_s")
            if duree_llm:
                durees_llm.append(float(duree_llm))

            score = data.get("intent", {}).get("score_rag")
            if score:
                scores.append(float(score))

        except Exception as e:
            print(f"  ⚠️  Erreur lecture {f} : {e}")

    total = len(fichiers)

    sous_titre("Résultats")
    print(f"  Nombre de requêtes analysées  : {total}")

    if durees_rag:
        print(f"  Temps RAG moyen               : {round(sum(durees_rag)/len(durees_rag), 3)}s")
        print(f"  Temps RAG min / max           : {min(durees_rag)}s / {max(durees_rag)}s")

    if durees_llm:
        print(f"  Temps LLM moyen               : {round(sum(durees_llm)/len(durees_llm), 3)}s")

    if scores:
        print(f"  Score cosinus moyen           : {round(sum(scores)/len(scores), 4)}")
        print(f"  Score cosinus min / max       : {round(min(scores), 4)} / {round(max(scores), 4)}")

    sous_titre("Distribution des méthodes")
    for methode, count in methodes.items():
        if total > 0:
            pct = round(count / total * 100, 1)
            barre = "█" * int(pct / 5)
            print(f"  {methode:<20} : {count:>3} ({pct:>5}%)  {barre}")

    return {
        "total_requetes":   total,
        "duree_rag_moy":    round(sum(durees_rag)/len(durees_rag), 3) if durees_rag else None,
        "duree_llm_moy":    round(sum(durees_llm)/len(durees_llm), 3) if durees_llm else None,
        "score_cos_moy":    round(sum(scores)/len(scores), 4) if scores else None,
        "methodes":         methodes,
        "taux_rag_direct":  round(methodes["rag_direct"] / total * 100, 1) if total > 0 else 0,
    }


# ============================================
# 2. MÉTRIQUES RÉSUMÉ MISTRAL
# ============================================
def metriques_resume():
    titre("MODULE 2 : Résumé Mistral")

    fichiers = glob.glob("mistral_resultat_*.json") + glob.glob("mistral_resultat.json")

    if not fichiers:
        print(f"  {R}Aucun fichier mistral_resultat*.json trouvé{W}")
        return {}

    taux_compression = []
    nb_mots_resume   = []

    for f in fichiers:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            texte    = data.get("texte_original", "")
            resume   = data.get("resume", "")
            mots_in  = len(texte.split())
            mots_out = len(resume.split())

            if mots_in > 0:
                taux = round(mots_out / mots_in * 100, 1)
                taux_compression.append(taux)
                nb_mots_resume.append(mots_out)

        except Exception as e:
            print(f"  ⚠️  Erreur : {e}")

    sous_titre("Résultats")
    print(f"  Nombre de résumés analysés    : {len(fichiers)}")

    if taux_compression:
        print(f"  Taux de compression moyen     : {round(sum(taux_compression)/len(taux_compression), 1)}%")
        print(f"  Mots résumé moyen             : {round(sum(nb_mots_resume)/len(nb_mots_resume), 1)} mots")
        print(f"  Mots résumé min / max         : {min(nb_mots_resume)} / {max(nb_mots_resume)} mots")

    return {
        "nb_resumes":           len(fichiers),
        "taux_compression_moy": round(sum(taux_compression)/len(taux_compression), 1) if taux_compression else None,
        "mots_resume_moy":      round(sum(nb_mots_resume)/len(nb_mots_resume), 1) if nb_mots_resume else None,
    }


# ============================================
# 3. MÉTRIQUES PLACEMENT MULTI-AGENT
# ============================================
def metriques_placement():
    titre("MODULE 3 : Placement Multi-Agent")

    fichiers = glob.glob("placement_*.json")

    if not fichiers:
        print(f"  {R}Aucun fichier placement_*.json trouvé{W}")
        return {}

    total_intentions  = 0
    total_places      = 0
    total_echecs      = 0
    total_reuses      = 0
    total_deploys     = 0
    total_failed      = 0
    durees_placement  = []
    durees_llm        = []
    sources           = {"multiagent": 0, "greedy_fallback": 0}

    for f in fichiers:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            resultats = data.get("resultats", [])

            for r in resultats:
                total_intentions += 1

                if r.get("succes"):
                    total_places += 1
                else:
                    total_echecs += 1

                duree = r.get("duree_s")
                if duree:
                    durees_placement.append(float(duree))

                duree_llm = r.get("duree_llm_s")
                if duree_llm:
                    durees_llm.append(float(duree_llm))

                source = r.get("source", "inconnu")
                if source in sources:
                    sources[source] += 1

                for p in r.get("placements", []):
                    action = str(p.get("action", "")).lower()
                    if "reuse"   in action:
                        total_reuses += 1
                    elif "deploy" in action:
                        total_deploys += 1
                    elif "failed" in action:
                        total_failed += 1

        except Exception as e:
            print(f"  ⚠️  Erreur : {e}")

    taux_placement = round(total_places / total_intentions * 100, 1) if total_intentions > 0 else 0
    taux_reuse     = round(total_reuses / (total_reuses + total_deploys) * 100, 1) if (total_reuses + total_deploys) > 0 else 0

    sous_titre("Résultats")
    print(f"  Intentions totales            : {total_intentions}")
    print(f"  Intentions placées            : {total_places}  ({taux_placement}%)")
    print(f"  Intentions échouées           : {total_echecs}")

    sous_titre("Services")
    print(f"  Déploiements (deploy)         : {total_deploys}")
    print(f"  Réutilisations (reuse)        : {total_reuses}  ({taux_reuse}% d'économie)")
    print(f"  Échecs service                : {total_failed}")

    sous_titre("Performances")
    if durees_placement:
        print(f"  Temps placement moyen         : {round(sum(durees_placement)/len(durees_placement), 2)}s")
        print(f"  Temps placement min / max     : {round(min(durees_placement), 2)}s / {round(max(durees_placement), 2)}s")

    if durees_llm:
        print(f"  Temps LLM placement moyen     : {round(sum(durees_llm)/len(durees_llm), 2)}s")

    sous_titre("Sources de placement")
    for src, count in sources.items():
        if total_intentions > 0:
            pct  = round(count / total_intentions * 100, 1)
            barre = "█" * int(pct / 5)
            print(f"  {src:<20} : {count:>3} ({pct:>5}%)  {barre}")

    return {
        "total_intentions":  total_intentions,
        "taux_placement":    taux_placement,
        "taux_reuse":        taux_reuse,
        "total_deploys":     total_deploys,
        "total_reuses":      total_reuses,
        "duree_moy_s":       round(sum(durees_placement)/len(durees_placement), 2) if durees_placement else None,
        "sources":           sources,
    }


# ============================================
# 4. MÉTRIQUES RESSOURCES NŒUDS
# ============================================
def metriques_noeuds():
    titre("MODULE 4 : Utilisation des nœuds")

    if not os.path.exists("etat_systeme.json"):
        print(f"  {R}Fichier etat_systeme.json non trouvé{W}")
        return {}

    try:
        with open("etat_systeme.json", "r", encoding="utf-8") as f:
            etat = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Erreur lecture état : {e}")
        return {}

    capacites    = etat.get("capacites", {})
    services_dep = etat.get("services_deployes", {})

    if not os.path.exists("dataset_4.json"):
        print(f"  {R}Fichier dataset_4.json non trouvé pour calculer les max{W}")
        return {}

    with open("dataset_4.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    nodes_ref = {n["id"]: n["capacity"] for n in dataset["nodes"]}

    sous_titre("Utilisation par nœud")
    print(f"  {'Nœud':<8} {'CPU%':>6} {'MEM%':>6} {'DISK%':>6} {'BW%':>6}  Services")
    print(f"  {'─'*55}")

    utils_cpu  = []
    utils_mem  = []
    utils_disk = []
    utils_bw   = []

    for nid, cap_max in nodes_ref.items():
        cap_rest = capacites.get(nid, cap_max)

        cpu_used  = max(0, cap_max["CPU"]  - cap_rest.get("CPU",  cap_max["CPU"]))
        mem_used  = max(0, cap_max["MEM"]  - cap_rest.get("MEM",  cap_max["MEM"]))
        disk_used = max(0, cap_max["DISK"] - cap_rest.get("DISK", cap_max["DISK"]))
        bw_used   = max(0, cap_max["BW"]   - cap_rest.get("BW",   cap_max["BW"]))

        cpu_pct  = round(cpu_used  / cap_max["CPU"]  * 100, 1) if cap_max["CPU"]  else 0
        mem_pct  = round(mem_used  / cap_max["MEM"]  * 100, 1) if cap_max["MEM"]  else 0
        disk_pct = round(disk_used / cap_max["DISK"] * 100, 1) if cap_max["DISK"] else 0
        bw_pct   = round(bw_used   / cap_max["BW"]   * 100, 1) if cap_max["BW"]   else 0

        utils_cpu.append(cpu_pct)
        utils_mem.append(mem_pct)
        utils_disk.append(disk_pct)
        utils_bw.append(bw_pct)

        nb_svcs = sum(1 for sid, nid2 in services_dep.items() if nid2 == nid)
        print(f"  {nid:<8} {cpu_pct:>5}% {mem_pct:>5}% {disk_pct:>5}% {bw_pct:>5}%  ({nb_svcs} services)")

    sous_titre("Moyennes globales")
    print(f"  CPU  moyen : {round(sum(utils_cpu)/len(utils_cpu), 1)}%")
    print(f"  MEM  moyen : {round(sum(utils_mem)/len(utils_mem), 1)}%")
    print(f"  DISK moyen : {round(sum(utils_disk)/len(utils_disk), 1)}%")
    print(f"  BW   moyen : {round(sum(utils_bw)/len(utils_bw), 1)}%")
    print(f"  Services déployés au total : {len(services_dep)}")

    return {
        "cpu_moy":       round(sum(utils_cpu)/len(utils_cpu), 1),
        "mem_moy":       round(sum(utils_mem)/len(utils_mem), 1),
        "disk_moy":      round(sum(utils_disk)/len(utils_disk), 1),
        "bw_moy":        round(sum(utils_bw)/len(utils_bw), 1),
        "services_total": len(services_dep),
    }


# ============================================
# 5. RÉSUMÉ FINAL POUR RAPPORT
# ============================================
def rapport_final(m_rag, m_resume, m_placement, m_noeuds):
    titre("RÉSUMÉ FINAL — TABLEAU DE BORD RAPPORT")

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  IndusMind — Métriques système end-to-end        │
  ├────────────────────────┬────────────────────────┤
  │  MATCHING RAG          │  PLACEMENT             │
  │  Temps RAG moy : {str(m_rag.get('duree_rag_moy','N/A'))+'s':<7} │  Taux succès : {str(m_placement.get('taux_placement','N/A'))+'%':<8} │
  │  Temps LLM moy : {str(m_rag.get('duree_llm_moy','N/A'))+'s':<7} │  Taux reuse  : {str(m_placement.get('taux_reuse','N/A'))+'%':<8} │
  │  Score cos moy : {str(m_rag.get('score_cos_moy','N/A')):<8} │  Temps moy   : {str(m_placement.get('duree_moy_s','N/A'))+'s':<8} │
  │  RAG direct    : {str(m_rag.get('taux_rag_direct','N/A'))+'%':<7} │  Multi-agent : {str(m_placement.get('sources',{}).get('multiagent','N/A')):<8} │
  ├────────────────────────┼────────────────────────┤
  │  RÉSUMÉ MISTRAL        │  RESSOURCES NŒUDS      │
  │  Compression   : {str(m_resume.get('taux_compression_moy','N/A'))+'%':<7} │  CPU moy     : {str(m_noeuds.get('cpu_moy','N/A'))+'%':<8} │
  │  Mots résumé   : {str(m_resume.get('mots_resume_moy','N/A')):<8} │  MEM moy     : {str(m_noeuds.get('mem_moy','N/A'))+'%':<8} │
  │                        │  Services total: {str(m_noeuds.get('services_total','N/A')):<6} │
  └────────────────────────┴────────────────────────┘
""")

    # Sauvegarder dans un JSON pour le rapport
    rapport = {
        "date_generation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rag_matching":    m_rag,
        "resume_mistral":  m_resume,
        "placement":       m_placement,
        "ressources_noeuds": m_noeuds,
    }

    with open("metriques_rapport.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=4)

    print(f"  {G}✅ Métriques sauvegardées dans metriques_rapport.json{W}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print(f"\n{G}🚀 IndusMind — Calcul des métriques pour rapport PFE{W}")
    print(f"   Dossier analysé : {os.getcwd()}")

    m_rag       = metriques_rag_matching()
    m_resume    = metriques_resume()
    m_placement = metriques_placement()
    m_noeuds    = metriques_noeuds()

    rapport_final(m_rag, m_resume, m_placement, m_noeuds)