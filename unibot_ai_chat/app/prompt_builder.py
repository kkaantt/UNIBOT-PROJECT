def build_prompt(query: str, retrieved_chunks: list) -> str:
    context_parts = []

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict) and "text" in chunk:
            context_parts.append(chunk["text"])
        else:
            context_parts.append(str(chunk))

    context = "\n\n".join(context_parts)

    prompt = f"""
    Sen bir üniversite yapay zeka asistanısın. Aşağıda bir öğrencinin sorusu ve bu soruyla ilgili içerikler yer alıyor.

    Cevap verirken sadece verilen içerikleri kullan. İçerikte açıkça belirtilmeyen hiçbir bilgiyi varsayma.
    Eğer aynı ders farklı yerlerde geçiyorsa, bilgileri sadeleştir ve tekrar etme. 
    Kredi, AKTS veya ders kodu gibi detaylar içerikte yazılıysa sadece o zaman cevapla.
    Soru Türkçe ise Türkçe, İngilizce ise İngilizce cevap ver. İngilizce sorulara türkçe veya Türkçe sorulara İngilizce cevap verme.
    Eğer kullanıcının sorduğu ders içerikte yoksa böyle bir ders bulunmadığını belirt.

📚 Bilgi:
{context}

❓ Soru:
{query}

💬 Yanıt:
"""
    return prompt.strip()
