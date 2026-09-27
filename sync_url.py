"""
sync_url.py - Synchronise l'URL ngrok entre Colab et le projet local
À exécuter dans VS Code avant de lancer app.py

Usage : python sync_url.py
"""

import os
import time

RAG_URL_FILE = "rag_url.txt"

def read_url():
    """Lit l'URL sauvegardée."""
    if os.path.exists(RAG_URL_FILE):
        with open(RAG_URL_FILE, "r") as f:
            url = f.read().strip()
        if url and url.startswith("http"):
            return url
    return None

def main():
    print("🔄 Synchronisation de l'URL RAG...")
    
    url = read_url()
    
    if url:
        print(f"✅ URL trouvée : {url}")
        print(f"\n🚀 Tu peux maintenant lancer : python app.py")
        print(f"   L'API RAG sera accessible via : {url}/api/chatbot")
        return True
    else:
        print(f"❌ Aucune URL dans {RAG_URL_FILE}")
        print("\n📋 Instructions :")
        print("   1. Lance le Script 9 dans Colab")
        print("   2. Attends que l'URL ngrok s'affiche")
        print("   3. Copie l'URL dans rag_url.txt")
        print("   4. Relance sync_url.py")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)