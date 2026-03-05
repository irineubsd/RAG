# RAG com Ollama + Chroma

Pipeline simples de Retrieval-Augmented Generation para tickets em CSV:

1. Ingestão do CSV para um índice vetorial (Chroma).
2. Consulta semântica com geração de resposta via LLM local (Ollama).

## Requisitos

- Python 3.10+
- Ollama em execução (`OLLAMA_BASE_URL`)
- Modelos baixados no Ollama:
  - embedding: `nomic-embed-text`
  - chat: `llama3.1:8b-instruct` (ou outro compatível)

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas python-dotenv langchain-core langchain-ollama langchain-chroma
```

## Configuração

Crie `.env` com base no exemplo:

```bash
cp .env.example .env
```

Variáveis principais:

- `CSV_PATH`: caminho do CSV de entrada.
- `CSV_ID_COLUMN`: coluna usada como identificador estável (`row_id`).
- `CSV_DELIMITER`: delimitador (`\t` para tab, `,` para vírgula).
- `CHROMA_PERSIST_DIR`: diretório de persistência do índice.
- `COLLECTION_NAME`: nome da coleção no Chroma.
- `TOP_K`: número de documentos recuperados por pergunta.

## Ingestão

```bash
python3 -m src.ingest_csv
```

Notas:

- A ingestão é idempotente por `row_id` (mesmo ID faz upsert, não duplica).
- IDs vazios em `CSV_ID_COLUMN` geram erro para proteger consistência.

## Consulta

```bash
python3 -m src.query
```

A resposta deve citar as fontes usando `row_id`.
