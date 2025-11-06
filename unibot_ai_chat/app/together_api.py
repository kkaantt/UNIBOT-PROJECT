import os
import requests
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3-8b-chat-hf"  # Llama modeli

def query_llama3(prompt: str) -> str:
    if not TOGETHER_API_KEY:
        return "❌ API anahtarı bulunamadı. .env dosyasını kontrol edin."

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for university students."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"❌ API Hatası: {e.response.status_code} {e.response.reason}"
    except Exception as e:
        return f"❌ Bir hata oluştu: {str(e)}"

