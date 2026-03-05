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
    parts = []

    for col in PREFERRED_COLS:
        if col in row.index:
            val = row[col]
            if pd.isna(val) or not str(val).strip():
                continue
            parts.append(f"{col}: {val}")

    if not parts:
        for col, val in row.items():
            if pd.isna(val) or not str(val).strip():
                continue
            parts.append(f"{col}: {val}")

    return "\n".join(parts)


def main() -> None:
    if not CSV_PATH:
        raise ValueError("CSV_PATH não definido. Configure no ambiente ou no arquivo .env.")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH não encontrado: {CSV_PATH}")

    df = pd.read_csv(
        CSV_PATH,
        sep=CSV_DELIMITER,
        dtype=str,
        keep_default_na=False,
        engine="c",
    )

    if CSV_ID_COLUMN not in df.columns:
        raise ValueError(
            f"CSV_ID_COLUMN='{CSV_ID_COLUMN}' não existe. Colunas: {list(df.columns)}"
        )

    docs = []
    doc_ids = []
    source_name = os.path.basename(CSV_PATH)

    for idx, row in df.iterrows():
        row_id = str(row[CSV_ID_COLUMN]).strip()
        if not row_id:
            raise ValueError(
                f"Linha {idx + 1} com '{CSV_ID_COLUMN}' vazio. "
                "IDs vazios quebram o upsert idempotente."
            )

        text = row_to_text(row)
        doc_id = f"{source_name}::{row_id}"

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "row_id": row_id,
                },
            )
        )
        doc_ids.append(doc_id)

    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    BATCH_SIZE = 2000
    total = len(docs)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        vs.add_documents(docs[start:end], ids=doc_ids[start:end])
        print(f" Upsert {end}/{total}")


if __name__ == "__main__":
    main()
