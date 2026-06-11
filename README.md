# Rosatom RAG

RAG-система для поиска и генерации ответов по нормативно-технической документации.

Проект реализует полный pipeline:

- ingest PDF/DOCX документов;
- извлечение текста;
- разбиение корпуса на чанки;
- построение FAISS-индекса;
- BM25-поиск по чанкам;
- NER-enhanced reranking;
- генерация ответа через локальную LLM;
- CLI-интерфейс для пользователя;
- evaluation pipeline для сравнения retrieval-подходов.

## Основной pipeline

```text
FAISS + BM25 + RRF + NER/entity reranking + LLM
```

Пользователь задаёт вопрос, система находит релевантные фрагменты документов, передаёт их в LLM и возвращает ответ с источниками.

## Возможности
- PDF/DOCX ingest.
- Chunking нормативных документов.
- FAISS dense retrieval.
- BM25 sparse retrieval.
- NER-enhanced reranking.
- Локальная LLM через llama.cpp или другой OpenAI-compatible server.
- Консольное меню для запуска RAG.
- Evaluation для сравнения baseline и NER-enhanced retrieval.
- Метрики: Hit@K, Recall@K, Precision@K, MRR@K, nDCG@K.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   ├── install_llama_cpp.md
│   └── install_python_env.md
├── data/
│   ├── raw/ntd/                         # сюда кладутся PDF/DOCX
│   ├── processed/
│   │   ├── extracted_text/              # извлечённый текст
│   │   ├── chunks/                      # jsonl чанки
│   │   └── vectorstore/                 # FAISS index
│   ├── eval/                            # questions/qrels для оценки
│   └── ner/
│       ├── labeled/                     # примеры NER-разметки
│       ├── predictions/                 # локальная генерация
│       └── entity_index/                # локальная генерация
├── models/
│   ├── emb/                             # embedding модель
│   └── ner/                             # NER модель
└── src/rosatom_rag/
    ├── cli.py
    ├── config.py
    ├── ingest/
    ├── retrieval/
    ├── ner/
    ├── llm/
    └── eval/
```    
## Что не хранится в git

В репозитории не хранятся:
- исходные PDF/DOCX документы;
- извлечённые тексты;
- чанки;
- FAISS index;
- NER predictions;
- chunk-level NER metadata;
- embedding model;
- NER model;
- GGUF LLM model;
- реальные рабочие NER corpus-файлы.
- В git оставлены только .gitkeep и example-файлы для демонстрации структуры и форматов.

## Установка

```bash
python3 -m venv .env
source .env/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Настройка LLM
Проект ожидает OpenAI-compatible endpoint:

http://127.0.0.1:8080/v1

Пример:
```bash
cp .env.example .env.local
export LLAMA_MODEL="bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"
```

Инструкция по llama.cpp:
```text
docs/install_llama_cpp.md
```

## Подготовка корпуса

### 1. Положить документы

```text
data/raw/ntd/
```

### 2. Извлечь текст

```bash
python -m rosatom_rag.ingest.extract_pdf_text
python -m rosatom_rag.ingest.extract_docx_text
```

### 3. Сделать чанки

```bash
python -m rosatom_rag.ingest.make_chunks
```

Результат:

```text
data/processed/chunks/ntd_chunks.jsonl
```

### 4. Построить FAISS index

```bash
python -m rosatom_rag.retrieval.build_faiss
```

Результат:

```text
data/processed/vectorstore/faiss_ntd/
```

## Подготовка NER metadata

```bash
python -m rosatom_rag.ner.predict_ner \
  --output data/ner/predictions/chunk_entities_full_v2.jsonl

python -m rosatom_rag.retrieval.build_chunk_ner_metadata
```

Результат:
```text
data/ner/entity_index/chunk_ner_metadata_v2.jsonl
```

## Запуск RAG

Перед запуском должны быть готовы:

```text
data/processed/chunks/ntd_chunks.jsonl
data/processed/vectorstore/faiss_ntd/
data/ner/entity_index/chunk_ner_metadata_v2.jsonl
```

### Интерактивное меню

```bash
python -m rosatom_rag.cli
```

Меню:

1 - Получить ответ от RAG системы на свой вопрос
2 - Запустить сравнение пайплайнов без NER и с NER
0 - Выход


### Задать вопрос без меню

```bash
python -m rosatom_rag.cli ask \
  --question "Согласно СП 484.1311500.2020, какой радиус контроля теплового точечного пожарного извещателя при высоте перекрытия до 3,5 м?"
```


## Evaluation

Сравниваются два retrieval-пайплайна:

```text
baseline:   FAISS + BM25 + RRF
enhanced:   FAISS + BM25 + RRF + NER/entity reranking
```


### Формат questions

```jsonl
{"id": 1, "discipline": "Электротехника", "question": "Согласно СП 484.1311500.2020, какой радиус контроля теплового точечного пожарного извещателя при высоте перекрытия до 3,5 м?", "source_doc_hint": "СП 484.1311500.2020"}
```

### Формат qrels

```jsonl
{"id": 1, "relevant_chunk_ids": ["chunk_id_from_ntd_chunks_jsonl"]}
```

Example-файлы:

```text
data/eval/questions.example.jsonl
data/eval/qrels.example.jsonl
```

### Проверить eval dataset

```bash
python -m rosatom_rag.eval.validate_dataset \
  --questions data/eval/questions.jsonl \
  --qrels data/eval/qrels.jsonl \
  --chunks data/processed/chunks/ntd_chunks.jsonl
```

### Запустить сравнение

```bash
python -m rosatom_rag.cli eval \
  --questions data/eval/questions.jsonl \
  --qrels data/eval/qrels.jsonl \
  --output-dir data/eval/final_hybrid_ner_cli
```

## Основные команды

```bash
# RAG интерфейс
python -m rosatom_rag.cli

# Evaluation
python -m rosatom_rag.cli eval

# Validate eval dataset
python -m rosatom_rag.eval.validate_dataset \
  --questions data/eval/questions.jsonl \
  --qrels data/eval/qrels.jsonl \
  --chunks data/processed/chunks/ntd_chunks.jsonl
```


## Документация

```text
docs/install_python_env.md
docs/install_llama_cpp.md
```

