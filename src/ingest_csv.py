import os
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import (
    CSV_PATH,
    CSV_ID_COLUMN,
    CSV_DELIMITER,
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    COLLECTION_NAME,
)

# Colunas que costumam ser mais úteis em QA (ajuste depois)
PREFERRED_COLS = [
    "ticket_number",
    "client_name",
    "contract_name",
    "desk_name",
    "requestor_name",
    "status_name",
    "stage_name",
    "appointment_user_name",
    "beginning",
    "ending",
    "date",
    "description",
    "duration",
    "value",
]


def row_to_text(row: pd.Series) -> str:
    # Mantém formato estável e “citável”
    parts = []
    for col in PREFERRED_COLS:
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")

    # fallback: se preferidas estiverem vazias, indexa tudo (evita doc vazio)
    if not parts:
        for col, val in row.items():
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")

    return "\n".join(parts)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH não encontrado: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, sep=CSV_DELIMITER, dtype=str, keep_default_na=False)

    if CSV_ID_COLUMN not in df.columns:
        raise ValueError(
            f"CSV_ID_COLUMN='{CSV_ID_COLUMN}' não existe. Colunas: {list(df.columns)}"
        )

    docs: list[Document] = []
    for _, row in df.iterrows():
        row_id = str(row[CSV_ID_COLUMN]).strip()
        text = row_to_text(row)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(CSV_PATH),
                    "row_id": row_id,
                },
            )
        )

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # MVP: adiciona tudo. Depois implementamos "reset" e "upsert" melhor.
    vs.add_documents(docs)

    print(f"Indexados {len(docs)} registros em {CHROMA_PERSIST_DIR} (collection={COLLECTION_NAME})")


if __name__ == "__main__":
    main()
