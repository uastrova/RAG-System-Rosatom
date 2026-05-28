# Скрипты для проверки работы


## Запуск обычного RAG baseline

Обычный baseline использует FAISS-поиск и LLM-ответ по найденному контексту.

Команда:

```bash
PYTHONPATH=src python -m rosatom_rag.eval.run_rag_baseline
```

Ожидаемый результат:

```text
================================================================================
RAG BASELINE EVAL
================================================================================
OUTPUT_DIR: data/eval/rag_baseline_v2
--------------------------------------------------------------------------------
QUESTION: Какие требования предъявляются к заземлению?
SAVED: data/eval/rag_baseline_v2/test_001.json
ANSWER PREVIEW:
...
Готово.
```

Результаты сохраняются в:

```text
data/eval/rag_baseline_v2/
```

Каждый вопрос сохраняется отдельным JSON-файлом:

```text
test_001.json
test_002.json
test_003.json
...
```

Пример структуры результата:

```json
{
  "question": "Какие требования предъявляются к заземлению?",
  "top_k": 4,
  "answer": "...",
  "sources": [
    {
      "source_file": "...",
      "page_num": 10,
      "chunk_id": "..."
    }
  ]
}
```

---

## Запуск hybrid baseline

Hybrid baseline объединяет несколько каналов поиска:

- FAISS;
- BM25;
- exact search по явным ссылкам на таблицы;
- reranker.

Команда:

```bash
PYTHONPATH=src python -m rosatom_rag.eval.run_rag_hybrid_baseline
```

Ожидаемый результат:

```text
================================================================================
RAG HYBRID BASELINE
================================================================================
OUTPUT_DIR: data/eval/rag_hybrid_v1
--------------------------------------------------------------------------------
QUESTION: Какие требования предъявляются к заземлению?
SAVED: data/eval/rag_hybrid_v1/test_001.json
ANSWER PREVIEW:
...
Готово.
```

Результаты сохраняются в:

```text
data/eval/rag_hybrid_v1/
```

---

## Запуск RAG с NER

NER-версия использует дополнительный entity search по сущностям, найденным NER-моделью.

Команда:

```bash
PYTHONPATH=src python -m rosatom_rag.eval.run_rag_hybrid_ner
```

Ожидаемый результат:

```text
================================================================================
RAG HYBRID NER
================================================================================
OUTPUT_DIR: data/eval/rag_hybrid_ner_v1
--------------------------------------------------------------------------------
QUESTION: Какие требования предъявляются к заземлению?
SAVED: data/eval/rag_hybrid_ner_v1/test_001.json
ANSWER PREVIEW:
...
Готово.
```

Результаты сохраняются в:

```text
data/eval/rag_hybrid_ner_v1/
```

---

## Как задать свой вопрос через retrieval (посмотреть чанки)

```bash
PYTHONPATH=src python - <<'PY'
from rosatom_rag.retrieval.hybrid_search_ner import similarity_search_hybrid_ner_reranked

question = "Что указано в таблице 1.7.7?"

docs = similarity_search_hybrid_ner_reranked(
    question,
    k=3,
    faiss_k=20,
    bm25_k=20,
    exact_k=10,
    entity_k=5,
    max_candidates=20,
)

for i, doc in enumerate(docs, start=1):
    print("=" * 80)
    print("DOC", i)
    print("source_file:", doc.metadata.get("source_file"))
    print("page_num:", doc.metadata.get("page_num"))
    print("chunk_id:", doc.metadata.get("chunk_id"))
    print("faiss_rank:", doc.metadata.get("faiss_rank"))
    print("bm25_rank:", doc.metadata.get("bm25_rank"))
    print("exact_rank:", doc.metadata.get("exact_rank"))
    print("entity_rank:", doc.metadata.get("entity_rank"))
    print("reranker_score:", doc.metadata.get("reranker_score"))
    print()
    print(doc.page_content[:1000])
PY
```

Ожидаемый результат:

```text
================================================================================
DOC 1
source_file: ...
page_num: ...
chunk_id: ...
faiss_rank: ...
bm25_rank: ...
exact_rank: ...
entity_rank: ...
reranker_score: ...
...
```

---

## Как задать свой вопрос и получить ответ LLM

```bash
PYTHONPATH=src python - <<'PY'
from rosatom_rag.retrieval.hybrid_search_ner import similarity_search_hybrid_ner_reranked
from rosatom_rag.llm.llama_client import format_docs, collect_sources, ask_llm_with_context

question = "Что указано в таблице 1.7.7?"

retrieved_docs = similarity_search_hybrid_ner_reranked(
    question,
    k=3,
    faiss_k=20,
    bm25_k=20,
    exact_k=10,
    entity_k=5,
    max_candidates=20,
)

context = format_docs(retrieved_docs)
answer = ask_llm_with_context(question, context)
sources = collect_sources(retrieved_docs)

print("=" * 80)
print("ANSWER")
print("=" * 80)
print(answer)

print()
print("=" * 80)
print("SOURCES")
print("=" * 80)
for source in sources:
    print(source)
PY
```

Ожидаемый результат:

```text
================================================================================
ANSWER
================================================================================
1. Краткий ответ.
...

2. Какие источники использованы:
file: ...
page: ...

================================================================================
SOURCES
================================================================================
{'source_file': '...', 'page_num': ..., 'chunk_id': '...'}
```

---

## Сравнение результатов

После запусков можно сравнивать файлы:

```text
data/eval/rag_hybrid_v1/test_001.json
data/eval/rag_hybrid_ner_v1/test_001.json
```

Что смотреть:

1. Какие `sources` выбраны.
2. Попал ли нужный нормативный документ.
3. Попала ли нужная таблица, пункт или раздел.
4. Стал ли ответ точнее после добавления NER.
5. Уменьшилось ли количество нерелевантных источников.

