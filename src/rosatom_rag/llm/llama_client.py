from openai import OpenAI

from rosatom_rag.config import LLAMA_BASE_URL, LLAMA_API_KEY


def build_client():
    client = OpenAI(
        base_url=LLAMA_BASE_URL,
        api_key=LLAMA_API_KEY,
    )
    return client


def format_docs(retrieved_docs) -> str:
    parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source_file = doc.metadata.get("source_file", "unknown_file")
        page_num = doc.metadata.get("page_num", "unknown_page")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")

        block = (
            f"[SOURCE {i}]\n"
            f"file: {source_file}\n"
            f"page: {page_num}\n"
            f"chunk_id: {chunk_id}\n"
            f"text:\n{doc.page_content}"
        )
        parts.append(block)

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)


def collect_sources(retrieved_docs):
    sources = []

    for doc in retrieved_docs:
        source_file = doc.metadata.get("source_file", "unknown_file")
        page_num = doc.metadata.get("page_num", "unknown_page")
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")

        sources.append({
            "source_file": source_file,
            "page_num": page_num,
            "chunk_id": chunk_id,
        })

    return sources


def ask_llm_with_context(question: str, context: str) -> str:
    client = build_client()

    system_prompt = (
        "Ты помощник по нормативно-технической документации. "
        "Отвечай только на основе переданного контекста. "
        "Если в контексте недостаточно данных, так и скажи. "
        "Ничего не придумывай от себя. "
        "Отвечай по-русски, понятно и по делу."
    )

    user_prompt = (
        f"Контекст:\n{context}\n\n"
        f"Вопрос:\n{question}\n\n"
        "Сформируй ответ в таком формате:\n"
        "1. Краткий ответ.\n"
        "2. Какие источники использованы (укажи file и page, если они есть в контексте).\n"
        "3. Если вопрос слишком широкий, скажи об этом и предложи уточнить формулировку.\n"
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )

    return resp.choices[0].message.content