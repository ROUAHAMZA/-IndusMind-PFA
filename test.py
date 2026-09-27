import urllib.request
import json
import ssl

# Créer un contexte SSL qui ne vérifie pas les certificats
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://enzyme-affiliate-chip.ngrok-free.dev/api/chatbot"
data = json.dumps({"message": "Qu'est-ce que l'Industrie 4.0 ?", "lang": "fr"}).encode('utf-8')

req = urllib.request.Request(
    url,
    data=data,
    headers={
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': '1',
    },
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(f"✅ Réponse : {result}")
except Exception as e:
    print(f"❌ Erreur : {e}")