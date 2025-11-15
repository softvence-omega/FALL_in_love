import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URL = "https://jahidtestmysite.pythonanywhere.com"
BACKEND_HISTORY_URL = f"{BASE_URL}/ai/ChatHistory/"
BACKEND_TOKEN_COUNT_URL = f"{BASE_URL}/ai/TokenCount/"
BACKEND_DOC_READ_COUNT_URL = f"{BASE_URL}/documents/Count/"

GLOBAL_ORG = "GlobalLaw"
LAW_COLLECTION=["AgedCareAct", "HomeCareAct", "NDIS", "GeneralAct", "Others"]

RAG_CONFIG = {
    "initial_fetch": int(os.getenv("RAG_INITIAL_LIMIT", "20")),
    "rerank_top_k": int(os.getenv("RERANK_TOP_K", "3")),
    "min_tokens_required": int(os.getenv("MIN_TOKENS_REQUIRED", "100")),
    "cache_ttl": int(os.getenv("CACHE_TTL_SECONDS", "3600"))
}
 
