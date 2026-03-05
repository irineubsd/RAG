from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    TOP_K,
    COLLECTION_NAME,
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um analista de suporte. Responda SOMENTE com base no CONTEXTO. "
            "Se faltar informação no contexto, diga exatamente o que falta. "
            "Sempre cite fontes usando row_id. "
            "Se houver múltiplas fontes, cite todas relevantes.",
        ),
        ("human", "PERGUNTA:\n{question}\n\nCONTEXTO:\n{context}\n"),
    ]
)


def format_context(docs) -> str:
    blocks = []
    for d in docs:
        row_id = d.metadata.get("row_id", "unknown")
        blocks.append(f"---\nrow_id: {row_id}\n{d.page_content}")
    return "\n".join(blocks)


def main() -> None:
    question = input("Pergunta> ").strip()
    if not question:
        raise ValueError("Pergunta vazia. Informe uma pergunta antes de consultar.")

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    docs = vs.similarity_search(question, k=TOP_K)
    if not docs:
        print("\nNenhum contexto encontrado para a pergunta.")
        return

    context = format_context(docs)

    llm = OllamaLLM(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)
    msg = PROMPT.format_messages(question=question, context=context)

    answer = llm.invoke(msg)

    print("\nResposta:\n")
    print(answer)


if __name__ == "__main__":
    main()
