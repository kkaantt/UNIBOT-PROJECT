from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from app.faiss_utils import search_similar_chunks
from app.together_api import query_llama3
from app.prompt_builder import build_prompt

# Ortam değişkenlerini yükle
load_dotenv()

# FastAPI  başlat
app = FastAPI()

# CORS ayarları (frontend'in erişebilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API'ye gelecek istek veri yapısı
class ChatRequest(BaseModel):
    message: str

# API endpoint: /chat
@app.post("/chat")
async def chat_endpoint(chat: ChatRequest):
    query = chat.message

    # FAISS'ten benzer içerikleri al
    chunks = search_similar_chunks(query)

    # Prompt oluştur
    prompt = build_prompt(query, chunks)

    # LLaMA 3'e sor
    response = query_llama3(prompt)

    return {"response": response}
