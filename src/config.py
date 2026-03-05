import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

CSV_PATH = os.getenv("CSV_PATH")
CSV_ID_COLUMN = os.getenv("CSV_ID_COLUMN", "ticket_number")
CSV_DELIMITER_RAW = os.getenv("CSV_DELIMITER", "\t")
# Permite escrever "\t" no .env e virar TAB real
CSV_DELIMITER = CSV_DELIMITER_RAW.encode("utf-8").decode("unicode_escape")

TOP_K = int(os.getenv("TOP_K", "6"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tickets")
