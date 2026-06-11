# Rosatom RAG

RAG-система для поиска и генерации ответов по нормативно-технической документации.

Проект реализует полный пайплайн:

- извлечение текста из PDF/DOCX документов;
- разбиение корпуса на чанки;
- построение FAISS индекса;
- FAISS и BM25 поиск по чанкам;
- NER-ранжирование чанков;
- генерация ответа через локальную LLM;
- консольный интерфейс;

## Основной pipeline

```text
FAISS + BM25 + rank-based fusion + NER/entity reranking + LLM
```

Пользователь задаёт вопрос, система находит релевантные фрагменты документов, передаёт их в LLM и возвращает ответ с источниками.

### Возможности
- Поддержка PDF/DOCX документов.
- Чанкинг документов.
- FAISS dense retrieval.
- BM25 sparse retrieval.
- NER-переранжирование.
- Локальная LLM через llama.cpp.
- Консольное меню для запуска RAG.
- Сравнение пайплайнов с NER-переранжированием и без.
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
│   │   └── vectorstore/                 # FAISS индекс
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
- FAISS индекс;
- предсказания NER;
- embedding модель;
- NER модель;
- GGUF LLM модель;
- реальные рабочие NER-файлы.

В git оставлены только .gitkeep и примеры файлов для демонстрации структуры и форматов.

## Установка

```bash
python3 -m venv .env
source .env/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Настройка LLM
Проект ожидает OpenAI-совместимый API endpoint:

http://127.0.0.1:8080/v1

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

### 3. Разбить корпус на чанки

```bash
python -m rosatom_rag.ingest.make_chunks
```

Результат:

```text
data/processed/chunks/ntd_chunks.jsonl
```

Этот файл является основным корпусом для дальнейшего поиска: по нему строится FAISS-индекс, а BM25-индекс создаётся в памяти при запуске поиска.
### 4. Проверить наличие embedding-модели

Перед построением FAISS-индекса embedding-модель должна лежать в директории:

```text
models/emb/sbert_large_nlu_ru/
```


### 5. Построить FAISS индекс

```bash
python -m rosatom_rag.retrieval.build_faiss
```

Результат:

```text
data/processed/vectorstore/faiss_ntd/
```

После этого готовы основные артефакты корпуса:

```text
data/processed/chunks/ntd_chunks.jsonl
data/processed/vectorstore/faiss_ntd/
```

BM25 не требует отдельной команды построения: индекс создаётся автоматически из ntd_chunks.jsonl при первом BM25-поиске.

## Подготовка NER-сущностей для поиска

### 1. Проверить наличие NER-модели

NER-модель должна лежать в директории:

```text
models/ner/rubert_ner_manual_v3_full_finetune/
```

Модель не хранится в git. Её нужно либо обучить отдельно, либо положить готовую модель в эту директорию.

### 2. Получить NER-предсказания для чанков
```bash
python -m rosatom_rag.ner.predict_ner \
  --output data/ner/predictions/chunk_entities_full.jsonl

python -m rosatom_rag.retrieval.build_chunk_ner_metadata
```

Результат:
```text
data/ner/entity_index/chunk_ner_metadata.jsonl
```

## Запуск RAG

Перед запуском должны быть готовы:

```text
data/processed/chunks/ntd_chunks.jsonl
data/processed/vectorstore/faiss_ntd/
data/ner/entity_index/chunk_ner_metadata.jsonl
```

### Консольное меню

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


## Исследрвание вклада NER в пайплан

Сравниваются два пайплайна:

```text
baseline:   FAISS + BM25 + rank-based fusion
enhanced:   FAISS + BM25 + rank-based fusion + NER/entity reranking
```


### Формат вопросов

```jsonl
{"id": 1, "discipline": "Электротехника", "question": "Согласно СП 484.1311500.2020, какой радиус контроля теплового точечного пожарного извещателя при высоте перекрытия до 3,5 м?", "source_doc_hint": "СП 484.1311500.2020"}
```

### Формат релевантных чанков для вопросов

```jsonl
{"id": 1, "relevant_chunk_ids": ["chunk_id_from_ntd_chunks_jsonl"]}
```

Примеры файлов:

```text
data/eval/questions.example.jsonl
data/eval/qrels.example.jsonl
```

### Запустить сравнение пайплайнов

```bash
python -m rosatom_rag.cli eval \
  --questions data/eval/questions.jsonl \
  --qrels data/eval/qrels.jsonl \
  --output-dir data/eval/final_hybrid_ner_cli
```


## Документация

```text
docs/install_python_env.md
docs/install_llama_cpp.md
```

