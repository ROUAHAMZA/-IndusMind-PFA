"""
app.py - Dashboard Flask avec pipeline backend COMPLET
Intègre : matching_rag_v3 + placement_multiagent_v3 (Solution C)
CORRECTION : Synchronisation cohérente nodes_state ↔ backend
CORRECTION 2 : Utiliser .get("action") pour éviter KeyError
"""

from collections import deque
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import json
import time
import threading
import queue
from datetime import datetime
import os
import ssl
import re
import numpy as np
import urllib
import urllib

# ============================================
# INFLUXDB CONFIGURATION
# ============================================
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = "indusmind"
INFLUXDB_BUCKET = "indusmind"

influxdb_client = None
influxdb_write_api = None

def init_influxdb():
    global influxdb_client, influxdb_write_api
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS

        influxdb_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        influxdb_write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)
        print("✅ InfluxDB connecté")
        return True
    except ImportError:
        print("⚠️  influxdb_client non installé. Installez avec: pip install influxdb-client")
        return False
    except Exception as e:
        print(f"⚠️  Erreur InfluxDB: {e}")
        return False

def send_to_influxdb(node_id, cpu_used, mem_used, disk_used, bw_used, 
                     cpu_max, mem_max, disk_max, bw_max, services_count):
    if not influxdb_write_api:
        return
    try:
        from influxdb_client import Point
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        point = Point("node_metrics")            .tag("node_id", node_id)            .tag("node_type", "gateway" if node_id.startswith("g") else "computing")            .field("cpu_used", float(cpu_used))            .field("mem_used", float(mem_used))            .field("disk_used", float(disk_used))            .field("bw_used", float(bw_used))            .field("cpu_max", float(cpu_max))            .field("mem_max", float(mem_max))            .field("disk_max", float(disk_max))            .field("bw_max", float(bw_max))            .field("cpu_pct", round(cpu_used / cpu_max * 100, 1) if cpu_max else 0)            .field("mem_pct", round(mem_used / mem_max * 100, 1) if mem_max else 0)            .field("disk_pct", round(disk_used / disk_max * 100, 1) if disk_max else 0)            .field("bw_pct", round(bw_used / bw_max * 100, 1) if bw_max else 0)            .field("services_count", int(services_count))            .time(now)

        influxdb_write_api.write(bucket=INFLUXDB_BUCKET, record=point)
    except Exception as e:
        print(f"⚠️  Erreur envoi InfluxDB: {e}")



# ============================================
# IMPORTS DU PIPELINE BACKEND
# ============================================
try:
    from cache_manager import charger_cache_embeddings, verifier_cache_a_jour, generer_cache_embeddings
    from matching_rag_v3 import (
        rag_top_k, llm_rerank, llm_fallback,
        get_intent_by_id, valider_intent, sauvegarder_intent,
        initialiser_cache, get_embedding, cosine_similarity,
        THRESHOLD, DOMINANCE_GAP, TOP_K
    )
    from placement_multiagent_v2 import (
        placer_intention,
        reserver_ressources,
        charger_dataset
    )
    from placement_knowledge import (
        trouver_patterns_similaires, formater_pattern_pour_llm,
        ajouter_pattern, charger_knowledge, sauvegarder_knowledge
    )
    from etat_manager import (
        sauvegarder_etat, charger_etat, reinitialiser_etat,
        verifier_coherence, ETAT_FILE
    )
    BACKEND_MODULES = True
    print("✅ [app.py] Backend chargé")

except ImportError as e:
    print(f"⚠️  [app.py] Backend non trouvé: {e}")
    BACKEND_MODULES = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

DATASET_FILE = "dataset_4.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# ============================================
# CHARGER RAG URL
# ============================================
def load_rag_url():
    if os.path.exists("rag_url.txt"):
        with open("rag_url.txt", "r") as f:
            url = f.read().strip()
        if url and url.startswith("http"):
            return url
    return None

RAG_URL = load_rag_url()

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

event_queues = []
event_queues_lock = threading.Lock()
nodes_state = {}
services_deployed = {}
history = []
metrics_history = deque(maxlen=500)

# ============================================
# CHARGER DATASET
# ============================================
def load_dataset():
    global nodes_state
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node in data["nodes"]:
            cap = node["capacity"]
            nodes_state[node["id"]] = {
                "id": node["id"], "type": node.get("type", "computing"),
                "CPU_max": cap["CPU"], "MEM_max": cap["MEM"],
                "DISK_max": cap["DISK"], "BW_max": cap["BW"],
                "CPU_used": 0.0, "MEM_used": 0.0,
                "DISK_used": 0.0, "BW_used": 0.0,
                "services": []
            }
        if BACKEND_MODULES:
            try:
                svcs, caps = charger_etat(nodes_reference=data["nodes"])
                if svcs is not None:
                    services_deployed.update(svcs)

                    # 🔴 CORRECTION : Restaurer de manière cohérente
                    # 1. D'abord, ajouter les services aux listes des nœuds
                    for sid, nid in services_deployed.items():
                        if nid in nodes_state and sid not in nodes_state[nid]["services"]:
                            nodes_state[nid]["services"].append(sid)

                    # 2. Recalculer les ressources utilisées depuis les capacités restantes (source de vérité)
                    for nid, cap in caps.items():
                        if nid in nodes_state:
                            nodes_state[nid]["CPU_used"] = max(0, nodes_state[nid]["CPU_max"] - cap.get("CPU", 0))
                            nodes_state[nid]["MEM_used"] = max(0, nodes_state[nid]["MEM_max"] - cap.get("MEM", 0))
                            nodes_state[nid]["DISK_used"] = max(0, nodes_state[nid]["DISK_max"] - cap.get("DISK", 0))
                            nodes_state[nid]["BW_used"] = max(0, nodes_state[nid]["BW_max"] - cap.get("BW", 0))

                    print(f"✅ État restauré: {len(services_deployed)} services, {len(caps)} nœuds")
            except Exception as e:
                print(f"⚠️  Pas d'état: {e}")
        return data
    except Exception as ex:
        print(f"[load_dataset] error: {ex}")
        return None

dataset = load_dataset()

if dataset and OLLAMA_AVAILABLE and BACKEND_MODULES:
    try:
        initialiser_cache()
        print("✅ Cache RAG V3 initialisé")
    except Exception as e:
        print(f"⚠️  Cache: {e}")

# ============================================
# UTILITAIRES
# ============================================
def node_remaining(node):
    return {
        "CPU": max(0, node["CPU_max"] - node["CPU_used"]),
        "MEM": max(0, node["MEM_max"] - node["MEM_used"]),
        "DISK": max(0, node["DISK_max"] - node["DISK_used"]),
        "BW": max(0, node["BW_max"] - node["BW_used"]),
    }

def latence_moyenne(node_id, latency):
    lats = latency.get(node_id, [])
    return round(sum(lats) / len(lats), 2) if lats else 999.0

# ============================================
# SSE
# ============================================
def send_event(data):
    with event_queues_lock:
        dead = []
        for q in event_queues:
            try:
                q.put_nowait(data)
            except:
                dead.append(q)
        for q in dead:
            event_queues.remove(q)

@app.route('/stream')
def stream():
    def generate():
        q = queue.Queue(maxsize=50)
        with event_queues_lock:
            event_queues.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except GeneratorExit:
            with event_queues_lock:
                if q in event_queues:
                    event_queues.remove(q)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ============================================
# PIPELINE COMPLET (CORRIGÉ)
# ============================================
def run_pipeline(user_input, is_voice=False):
    send_event({"type": "pipeline_start", "input": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")})
    time.sleep(0.3)

    # STEP 1
    send_event({"type": "step", "step": 1, "status": "running", "label": "Acquisition Input"})
    time.sleep(0.5)
    text_input = user_input
    send_event({"type": "step_detail", "step": 1,
                "detail": f"{'🎤 Audio' if is_voice else '✍️ Texte'} → {text_input[:50]}..."})
    time.sleep(0.3)
    send_event({"type": "step", "step": 1, "status": "done", "label": "Acquisition Input"})

    # STEP 2
    send_event({"type": "step", "step": 2, "status": "running", "label": "Résumé Mistral"})
    resume = None
    if OLLAMA_AVAILABLE:
        try:
            resp = ollama.chat(model=LLM_MODEL, messages=[
                {"role": "system", "content": "Summarize worker commands into 5 words max. English only."},
                {"role": "user", "content": f"Summarize: {text_input}"}
            ])
            resume = resp["message"]["content"].strip()
            # Sauvegarder mistral_resultat pour les métriques
            try:
                mistral_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "modele": "mistral",
                    "texte_original": text_input,
                "resume": resume
             }
                filename = f"mistral_resultat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(mistral_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"⚠️ Sauvegarde mistral_resultat: {e}")
        except:
            pass
    if not resume:
        resume = " ".join(text_input.split()[:5])
    send_event({"type": "step_detail", "step": 2, "detail": f"📝 {resume}"})
    time.sleep(0.3)
    send_event({"type": "step", "step": 2, "status": "done", "label": "Résumé Mistral"})

    # STEP 3-6 : BACKEND
    send_event({"type": "step", "step": 3, "status": "running", "label": "RAG Matching"})
    send_event({"type": "step", "step": 4, "status": "running", "label": "Resource Manager"})
    send_event({"type": "step", "step": 5, "status": "running", "label": "QoS Analyzer"})
    send_event({"type": "step", "step": 6, "status": "running", "label": "Placement Master"})

    intent_result = None
    placements = []
    source_matching = "unknown"
    source_placement = "unknown"
    succes = False

    if BACKEND_MODULES and dataset:
        try:
            initialiser_cache()
            intentions = dataset["intentions"]
            intentions_valides = {i["id"] for i in intentions}

            # MATCHING
            top_k, decision, duree_rag = rag_top_k(resume, intentions)
            send_event({"type": "step_detail", "step": 3,
                       "detail": f"🔍 RAG: {decision} | {duree_rag}s"})

            duree_llm = None
            if decision == "rag_direct":
                intent_result = top_k[0]
                intent_result["score"] = top_k[0]["score"]
                source_matching = "rag_direct"
                send_event({"type": "step_detail", "step": 3,
                           "detail": f"✅ RAG direct: {intent_result['id']} (score={intent_result['score']})"})
            elif decision == "llm_rerank":
                intent_id, duree_llm = llm_rerank(resume, top_k, intentions_valides)
                if intent_id:
                    intent_result = get_intent_by_id(intent_id, intentions)
                source_matching = "llm_rerank"
                send_event({"type": "step_detail", "step": 3,
                           "detail": f"🔄 LLM rerank: {intent_id}"})
            else:
                intent_id, duree_llm = llm_fallback(resume, intentions, intentions_valides)
                if intent_id:
                    intent_result = get_intent_by_id(intent_id, intentions)
                source_matching = "llm_fallback"
                send_event({"type": "step_detail", "step": 3,
                           "detail": f"⚠️ LLM fallback: {intent_id}"})

            if intent_result and valider_intent(intent_result, dataset):
                sauvegarder_intent(resume, intent_result, source_matching, duree_rag, duree_llm)
                send_event({"type": "step_detail", "step": 3,
                           "detail": f"✅ Intent validé: {intent_result['id']}"})
            else:
                intent_result = None

            # Fallback
            if not intent_result:
                query = (resume + " " + text_input).lower()
                best_score = -1
                best_intent = None
                for intent in intentions:
                    desc_words = set(intent["description"].lower().split())
                    query_words = set(query.split())
                    overlap = len(desc_words & query_words) / max(len(desc_words | query_words), 1)
                    if overlap > best_score:
                        best_score = overlap
                        best_intent = intent
                if best_intent:
                    intent_result = {
                        "id": best_intent["id"],
                        "description": best_intent["description"],
                        "services": best_intent["services"],
                        "QoS": best_intent.get("QoS", {"latency": 200, "bandwidth": 50}),
                        "weight": best_intent.get("weight", 0),
                        "score": round(best_score, 3)
                    }
                    source_matching = "keyword_fallback"

            # PLACEMENT
            if intent_result:
                # 🔴 CORRECTION CLÉ : Construire capacites et services_deployes depuis nodes_state
                # nodes_state = SEULE SOURCE DE VÉRITÉ
                # Ne JAMAIS recharger depuis le fichier pendant l'exécution

                capacites = {}
                services_deployes = {}

                for nid, node in nodes_state.items():
                    capacites[nid] = {
                        "CPU": max(0, node["CPU_max"] - node["CPU_used"]),
                        "MEM": max(0, node["MEM_max"] - node["MEM_used"]),
                        "DISK": max(0, node["DISK_max"] - node["DISK_used"]),
                        "BW": max(0, node["BW_max"] - node["BW_used"])
                    }

                for sid, nid in services_deployed.items():
                    services_deployes[sid] = nid

                intention = {
                    "id": intent_result["id"],
                    "description": intent_result["description"],
                    "services": intent_result["services"],
                    "QoS": intent_result["QoS"],
                    "weight": intent_result.get("weight", 0)
                }

                send_event({"type": "step_detail", "step": 4,
                           "detail": f"🔍 {len(intention['services'])} services"})
                send_event({"type": "step_detail", "step": 5,
                           "detail": f"📊 QoS: lat≤{intention['QoS']['latency']}ms"})

                resultat = placer_intention(intention, dataset, capacites, services_deployes)

                placements = resultat["placements"]
                source_placement = resultat["source"]
                succes = resultat["succes"]
                # Sauvegarder placement.json pour les métriques
                try:
                    placement_data = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "modele": LLM_MODEL,
                            "algorithme": "Multi-Agent V2 (persistance + RAG) + Greedy Fallback",
        "taux_placement": f"{100.0 if succes else 0.0}%",
        "temps_total_s": resultat["duree_s"],
        "reuses_total": sum(1 for p in placements if "reuse" in str(p.get("action","")).lower()),
        "resultats": [resultat]
    }
                    filename_p = f"placement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename_p, "w", encoding="utf-8") as f:
                            json.dump(placement_data, f, ensure_ascii=False, indent=4)
                except Exception as e:
                        print(f"⚠️ Sauvegarde placement.json: {e}")

                        send_event({"type": "step_detail", "step": 6,
                           "detail": f"✅ {source_placement} | {len(placements)} services"})

                # 🔴 CORRECTION CLÉ : Synchroniser nodes_state depuis les placements
                # Puis sauvegarder dans le fichier pour persistance au redémarrage
                for p in placements:
                    sid = p["service_id"]
                    nid = p["node_id"]
                    action = p.get("action", "deploy")  # 🔴 CORRECTION : Utiliser .get() avec default

                    action_str = str(action).lower()
                    is_reuse = "reuse" in action_str
                    is_deploy = "deploy" in action_str

                    if is_deploy and nid:
                        res = next((s["resources"] for s in dataset.get("services", []) if s["id"] == sid), {})

                        # Mettre à jour nodes_state (source de vérité)
                        nodes_state[nid]["CPU_used"] += res.get("CPU", 0)
                        nodes_state[nid]["MEM_used"] += res.get("MEM", 0)
                        nodes_state[nid]["DISK_used"] += res.get("DISK", 0)
                        nodes_state[nid]["BW_used"] += res.get("BW", 0)

                        if sid not in nodes_state[nid]["services"]:
                            nodes_state[nid]["services"].append(sid)

                        # Mettre à jour services_deployed (global Flask)
                        services_deployed[sid] = nid

                        print(f"DEBUG DEPLOY: {sid} → {nid} | CPU+={res.get('CPU',0)} MEM+={res.get('MEM',0)}")

                    elif is_reuse and nid:
                        if sid not in nodes_state[nid]["services"]:
                            nodes_state[nid]["services"].append(sid)
                        services_deployed[sid] = nid
                        print(f"DEBUG REUSE: {sid} → {nid}")

                # 🔴 NOUVEAU : Sauvegarder l'état cohérent dans le fichier
                # pour que le redémarrage recharge l'état correct
                capacites_sauvegarde = {}
                for nid, node in nodes_state.items():
                    capacites_sauvegarde[nid] = {
                        "CPU": max(0, node["CPU_max"] - node["CPU_used"]),
                        "MEM": max(0, node["MEM_max"] - node["MEM_used"]),
                        "DISK": max(0, node["DISK_max"] - node["DISK_used"]),
                        "BW": max(0, node["BW_max"] - node["BW_used"])
                    }

                try:
                    sauvegarder_etat(services_deployed, capacites_sauvegarde)
                    print(f"💾 État synchronisé sauvegardé: {len(services_deployed)} services")
                except Exception as e:
                    print(f"⚠️ Erreur sauvegarde état: {e}")

        except Exception as e:
            print(f"❌ Backend error: {e}")
            import traceback
            traceback.print_exc()

    # Marquer étapes done
    for i in range(3, 7):
        labels = ["RAG Matching", "Resource Manager", "QoS Analyzer", "Placement Master"]
        send_event({"type": "step", "step": i, "status": "done", "label": labels[i-3]})

    # ENVOYER LA MISE À JOUR DES NŒUDS (pour les ressources)
    nodes_for_canvas = []
    for n in nodes_state.values():
        nodes_for_canvas.append({
            "id": n["id"], "type": n["type"],
            "CPU": round(n["CPU_used"], 2), "MEM": round(n["MEM_used"], 2),
            "DISK": round(n["DISK_used"], 2), "BW": round(n["BW_used"], 2),
            "CPU_max": n["CPU_max"], "MEM_max": n["MEM_max"],
            "DISK_max": n["DISK_max"], "BW_max": n["BW_max"],
            "services": n["services"]
        })
    send_event({"type": "nodes_update", "nodes": nodes_for_canvas})

    # Envoyer les placements (pour animations)
    for p in placements:
        sid = p["service_id"]
        nid = p["node_id"]
        action = p.get("action", "deploy")  # 🔴 CORRECTION : Utiliser .get() avec default

        # Vérifier que le nœud existe dans nodes_state
        if nid and nid in nodes_state:
            send_event({
                "type": "service_placed",
                "placement": p,
                "node": {
                    "id": nid, "type": nodes_state[nid]["type"],
                    "CPU": round(nodes_state[nid]["CPU_used"], 2),
                    "MEM": round(nodes_state[nid]["MEM_used"], 2),
                    "DISK": round(nodes_state[nid]["DISK_used"], 2),
                    "BW": round(nodes_state[nid]["BW_used"], 2),
                    "CPU_max": nodes_state[nid]["CPU_max"],
                    "MEM_max": nodes_state[nid]["MEM_max"],
                    "DISK_max": nodes_state[nid]["DISK_max"],
                    "BW_max": nodes_state[nid]["BW_max"],
                    "services": list(nodes_state[nid]["services"])
                }
            })
            icon = "♻️" if "reuse" in str(action).lower() else "🚀"
            send_event({"type": "step_detail", "step": 6,
                       "detail": f"{icon} {sid} → {nid} ({action})"})
            time.sleep(0.3)
        else:
            print(f"⚠️ Placement sans nœud valide: {sid} → {nid}")

    # FINAL
    result = {
        "intent_id": intent_result.get("id", "unknown") if intent_result else "unknown",
        "description": intent_result.get("description", resume) if intent_result else resume,
        "services": intent_result.get("services", []) if intent_result else [],
        "placements": placements,
        "rag_score": intent_result.get("score", 0) if intent_result else 0,
        "rag_method": source_matching,
        "source": source_placement,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    history.append(result)

    send_event({"type": "pipeline_done", "result": result})

# ============================================
# ROUTES API
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    data = request.json
    user_input = data.get('text', '').strip()
    is_voice = data.get('is_voice', False)
    if not user_input:
        return jsonify({"error": "Input vide"}), 400
    t = threading.Thread(target=run_pipeline, args=(user_input, is_voice))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})

@app.route('/api/nodes')
def get_nodes():
    result = []
    for n in nodes_state.values():
        result.append({
            "id": n["id"], "type": n["type"],
            "CPU": round(n["CPU_used"], 2), "MEM": round(n["MEM_used"], 2),
            "DISK": round(n["DISK_used"], 2), "BW": round(n["BW_used"], 2),
            "CPU_max": n["CPU_max"], "MEM_max": n["MEM_max"],
            "DISK_max": n["DISK_max"], "BW_max": n["BW_max"],
            "services": n["services"]
        })
    return jsonify(result)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json or {}
    question = data.get('message', '')
    lang = data.get('lang', 'fr')
    if not RAG_URL:
        return jsonify({"answer": "Chatbot non configuré"}), 503
    if not question or len(question.strip()) < 2:
        return jsonify({"answer": "Pose une question"}), 400
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{RAG_URL}/api/chatbot",
                data=json.dumps({"message": question, "lang": lang}).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=8, context=ssl_context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return jsonify({"answer": result.get("answer", "")})
        except Exception as e:
            print(f"Chatbot attempt {attempt+1}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return jsonify({"answer": "Chatbot temporairement indisponible"}), 503

@app.route('/api/stats')
def get_stats():
    total = len(history)
    placed = sum(
        sum(1 for p in h.get("placements", []) if "deploy" in str(p.get("action", "")).lower())
        for h in history
    )
    reuses = sum(
        sum(1 for p in h.get("placements", []) if "reuse" in str(p.get("action", "")).lower())
        for h in history
    )
    return jsonify({
        "total_intents": total,
        "placed": placed,
        "reuses": reuses,
        "success_rate": 100 if total > 0 else 0
    })

@app.route('/api/reset', methods=['POST'])
def reset_nodes():
    global services_deployed, history
    services_deployed = {}
    history = []
    for nid, node in nodes_state.items():
        node['CPU_used'] = 0.0
        node['MEM_used'] = 0.0
        node['DISK_used'] = 0.0
        node['BW_used'] = 0.0
        node['services'] = []
    if BACKEND_MODULES:
        try:
            reinitialiser_etat()
            print("🗑️  État backend réinitialisé")
        except Exception as e:
            print(f"⚠️  Reset backend: {e}")
        try:
            from placement_knowledge import KNOWLEDGE_FILE
            if os.path.exists(KNOWLEDGE_FILE):
                os.remove(KNOWLEDGE_FILE)
                print(f"🗑️  Knowledge supprimée")
        except:
            pass
    send_event({'type': 'nodes_update', 'nodes': [
        {'id': n['id'], 'type': n['type'],
         'CPU': 0, 'MEM': 0, 'DISK': 0, 'BW': 0,
         'CPU_max': n['CPU_max'], 'MEM_max': n['MEM_max'],
         'DISK_max': n['DISK_max'], 'BW_max': n['BW_max'],
         'services': []}
        for n in nodes_state.values()
    ]})
    return jsonify({'status': 'reset'})

@app.route('/api/metrics/history')
def get_metrics_history():
    node_filter = request.args.get('node', None)
    if not metrics_history:
        return jsonify([])
    return jsonify([m for m in metrics_history if not node_filter or m["node"] == node_filter])

# ============================================
# METRICS THREAD
# ============================================
import time as time_module

def record_metrics():
    # Initialiser InfluxDB au démarrage du thread
    init_influxdb()

    while True:
        ts = int(time_module.time() * 1000)
        for nid, node in nodes_state.items():
            metrics_history.append({
                "timestamp": ts, "node": nid,
                "cpu": round(node["CPU_used"] / node["CPU_max"] * 100, 1) if node["CPU_max"] else 0,
                "mem": round(node["MEM_used"] / node["MEM_max"] * 100, 1) if node["MEM_max"] else 0,
                "bw": round(node["BW_used"] / node["BW_max"] * 100, 1) if node["BW_max"] else 0,
                "disk": round(node["DISK_used"] / node["DISK_max"] * 100, 1) if node["DISK_max"] else 0
            })

            # Envoyer à InfluxDB
            send_to_influxdb(
                node_id=nid,
                cpu_used=node["CPU_used"],
                mem_used=node["MEM_used"],
                disk_used=node["DISK_used"],
                bw_used=node["BW_used"],
                cpu_max=node["CPU_max"],
                mem_max=node["MEM_max"],
                disk_max=node["DISK_max"],
                bw_max=node["BW_max"],
                services_count=len(node.get("services", []))
            )

        time.sleep(10)

threading.Thread(target=record_metrics, daemon=True).start()

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 IndusMind - Backend Intégré")
    print(f"   Backend: {'✅' if BACKEND_MODULES else '❌'}")
    print(f"   Ollama: {'✅' if OLLAMA_AVAILABLE else '❌'}")
    print(f"   RAG URL: {RAG_URL or '❌'}")
    print("=" * 50)
    app.run(debug=True, threaded=True, port=5000)