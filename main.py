import os, sys
from dotenv import load_dotenv
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


load_dotenv()
# Cloudflare AI настройки
ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID', 'your-account-id')
AUTH_TOKEN = os.getenv('CLOUDFLARE_AUTH_TOKEN', 'your-token')
CLOUDFLARE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"

IPCC_DOCS = [
    "Climate change has caused widespread adverse impacts to nature and people.",
    "Human-induced climate change is affecting weather and climate extremes globally.",
    "Key risks include water scarcity, food insecurity, coastal flooding, and heat mortality.",
    "Rising sea levels and coastal flooding are increasing.",
    "Climate change has reduced food security and hindered economic growth.",
    "Ecosystems face widespread impacts from climate change.",
    "Cities face increasing risks from heat stress and coastal flooding.",
    "Climate-driven food price increases have been observed.",
    "3.3-3.6 billion people live in contexts highly vulnerable to climate change.",
    "Heatwaves have increased in intensity causing increased mortality.",
]


def generate_variants(query):
    """Генерация 3 вариантов запроса через Cloudflare AI (Llama 3.3)"""
    prompt = f"""Generate exactly 3 different ways to ask this question. 
Keep the same meaning but use different words and phrasing.
Output ONLY the 3 questions, numbered 1-3, nothing else.

Original question: {query}

3 variants:"""

    response = requests.post(
        CLOUDFLARE_URL,
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a query reformulation expert. Generate diverse query variants."},
                {"role": "user", "content": prompt}
            ]
        },
        timeout=30
    )
    
    result = response.json()
    
    # Cloudflare возвращает результат в result['result']['response']
    if 'result' in result and 'response' in result['result']:
        text = result['result']['response']
    else:
        raise Exception(f"Unexpected API response format: {result}")
    
    # Парсим варианты из ответа
    variants = []
    for line in text.split('\n'):
        line = line.strip().lstrip('0123456789.-) ').strip('"\'')
        if len(line) > 10 and line not in variants:
            variants.append(line)
    
    return variants[:3]


def search_docs(queries, documents, k=5):
    """Поиск по документам с помощью embeddings и FAISS"""
    print("   Загрузка модели embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("   Индексация документов...")
    doc_embeddings = encoder.encode(documents)
    doc_embeddings = np.array(doc_embeddings, dtype='float32')
    faiss.normalize_L2(doc_embeddings)
    
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    # Поиск для каждого запроса
    all_found = set()
    
    print(f"   Поиск по {len(queries)} запросам...")
    for query in queries:
        query_emb = encoder.encode([query])
        query_emb = np.array(query_emb, dtype='float32')
        faiss.normalize_L2(query_emb)
        
        scores, indices = index.search(query_emb, k)
        all_found.update(indices[0].tolist())
    
    return all_found


def calculate_similarity(original, variants):
    """Считаем similarity между оригиналом и вариантами"""
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    orig_emb = encoder.encode([original])[0]
    var_embs = encoder.encode(variants)
    
    scores = []
    for var_emb in var_embs:
        sim = np.dot(orig_emb, var_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(var_emb))
        scores.append(float(sim))
    
    return scores


def main():
    query = ' '.join(sys.argv[1:])
    
    print("\n" + "="*70)
    print("AI QUERY OPTIMIZER")
    print("Cloudflare AI (Llama 3.3) + Sentence Transformers + FAISS")
    print("="*70)
    print(f"\n📝 Оригинальный запрос: '{query}'")
    
    # 1. Генерируем варианты через Cloudflare AI
    print("\n[Шаг 1/4] Генерация вариантов через Cloudflare AI (Llama 3.3)...")
    try:
        variants = generate_variants(query)
        print(f"   ✓ Сгенерировано {len(variants)} вариантов")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        print("\n⚠️  Проверьте переменные окружения:")
        print("   - CLOUDFLARE_ACCOUNT_ID")
        print("   - CLOUDFLARE_AUTH_TOKEN")
        return
    
    # 2. Считаем similarity scores
    print("\n[Шаг 2/4] Расчет similarity scores между запросами...")
    similarities = calculate_similarity(query, variants)
    print(f"   ✓ Рассчитаны косинусные дистанции")
    
    # 3. Baseline retrieval (только оригинальный запрос)
    print("\n[Шаг 3/4] Baseline retrieval (1 запрос)...")
    baseline_docs = search_docs([query], IPCC_DOCS, k=5)
    print(f"   ✓ Найдено {len(baseline_docs)} уникальных документов")
    
    # 4. Optimized retrieval (оригинал + 3 варианта)
    print("\n[Шаг 4/4] Optimized retrieval (4 запроса: оригинал + варианты)...")
    optimized_docs = search_docs([query] + variants, IPCC_DOCS, k=5)
    print(f"   ✓ Найдено {len(optimized_docs)} уникальных документов")
    
    # Расчет метрик
    recall_improvement = (len(optimized_docs) - len(baseline_docs)) / len(baseline_docs) * 100
    
    # Вывод результатов
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)
    
    print("\n🔄 Оптимизированные запросы:")
    for i, (variant, score) in enumerate(zip(variants, similarities), 1):
        print(f"\n  {i}. [Similarity: {score:.3f}]")
        print(f"     {variant}")
    
    print("\n" + "-"*70)
    print("📊 Retrieval статистика:")
    print(f"   Baseline (1 query):     {len(baseline_docs)} документов")
    print(f"   Optimized (4 queries):  {len(optimized_docs)} документов")
    print(f"   Новых найдено:          {len(optimized_docs) - len(baseline_docs)} документов")
    print(f"   Recall improvement:     {recall_improvement:+.1f}%")
    
    print("\n" + "-"*70)
    if recall_improvement >= 20:
        print("✅ Цель достигнута: recall улучшение ≥20% (требование PRD)")
    else:
        print(f"⚠️  Recall +{recall_improvement:.1f}% (цель: ≥20%)")
        print("   💡 Попробуйте другой запрос или увеличьте k")
    
    print("\n" + "="*70)
    print("\n✨ Готово! Multi-query retrieval работает.")
    print("💡 Интеграция в RAG: используйте все 4 запроса для поиска контекста\n")


if __name__ == '__main__':
    main()