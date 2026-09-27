import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import json
import ollama
from faster_whisper import WhisperModel
from datetime import datetime

# ============================================
# PHASE 1 : ENREGISTREMENT DEPUIS LE MICRO
# ============================================
def enregistrer_audio(samplerate=16000):
    print("🎤 Parlez... (appuyez ENTRÉE pour arrêter)")

    frames = []

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate,
                        channels=1,
                        dtype='int16',
                        callback=callback):
        input()

    audio = np.concatenate(frames, axis=0)
    wav.write("temp_audio.wav", samplerate, audio)
    print("✅ Enregistrement terminé !")


# ============================================
# PHASE 2 : TRANSCRIPTION AVEC FASTER-WHISPER
# ============================================
def transcrire_audio():
    print("🔄 Transcription en cours...")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe("temp_audio.wav", language="fr")

    texte_complet = ""
    for segment in segments:
        texte_complet += segment.text

    texte_complet = texte_complet.strip()
    print(f"📝 Texte transcrit : {texte_complet}")
    return texte_complet


# ============================================
# PHASE 3 : RÉSUMÉ INTELLIGENT AVEC MISTRAL
# ============================================
def resumer_texte(texte):
    print("🤖 Résumé en cours avec mistral...")

    reponse = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": """You are an industrial maintenance assistant 
in a smart factory. Your job is to summarize 
worker voice commands into short technical descriptions."""
            },
            {
                "role": "user",
                "content": f"""Summarize the following sentence in maximum 5 words.
Keep technical terms. English output only.

Examples:
Sentence: "je veux remplacer l'unité de puissance de la machine"
Summary: replace power unit machine

Sentence: "j'ai besoin d'instructions de maintenance instantanées"
Summary: instant maintenance instructions needed

Sentence: "afficher les erreurs pendant l'assemblage"
Summary: highlight assembly errors display

Now summarize this:
Sentence: {texte}
Summary:"""
            }
        ]
    )

    resume = reponse["message"]["content"].strip()
    print(f"✂️ Résumé : {resume}")
    return resume


# ============================================
# PHASE 4 : COLLECTER PLUSIEURS INTENTIONS
# ============================================
def collecter_intentions():
    print("\n" + "="*50)
    print("🎯 COLLECTE DES INTENTIONS UTILISATEURS")
    print("="*50)

    intentions_collectees = []
    user_id = 1

    while True:
        print(f"\n👷 User {user_id} — Appuyez ENTRÉE pour commencer à parler")
        print("   (ou tapez 'fin' pour terminer la collecte)")

        choix = input("→ ")

        if choix.lower() == "fin":
            break

        # Enregistrer et traiter la voix
        enregistrer_audio()
        texte = transcrire_audio()

        if texte:
            resume = resumer_texte(texte)

            intentions_collectees.append({
                "user_id": user_id,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "texte_original": texte,
                "resume": resume
            })

            print(f"✅ Intention User {user_id} enregistrée !")
        else:
            print(f"❌ Aucun texte détecté pour User {user_id} !")

        user_id += 1

        print(f"\n➕ Ajouter un autre utilisateur ? (ENTRÉE = oui / 'fin' = terminer)")
        continuer = input("→ ")
        if continuer.lower() == "fin":
            break

    return intentions_collectees


# ============================================
# PHASE 5 : SAUVEGARDER TOUTES LES INTENTIONS
# ============================================
def sauvegarder_intentions(intentions_collectees):
    data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modele": "mistral",
        "nombre_users": len(intentions_collectees),
        "intentions": intentions_collectees
    }

    with open("intentions_users.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print(f"💾 {len(intentions_collectees)} intentions sauvegardées dans intentions_users.json ✅")
    print("="*50)
    print(json.dumps(data, ensure_ascii=False, indent=4))


# ============================================
# PROGRAMME PRINCIPAL
# ============================================
if __name__ == "__main__":
    intentions = collecter_intentions()

    if intentions:
        sauvegarder_intentions(intentions)
        print("\n🚀 Prêt pour le placement ! Lance : python placement.py")
    else:
        print("❌ Aucune intention collectée !")