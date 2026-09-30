"""All settings for the app live here, in one place."""


import os 
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

## ENV VAR / SECRET - LLMS 
JINA_API_KEY = os.getenv("JINA_API_KEY")


# GATEWAY 
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
# These legacy-named variables hold Portkey Model Catalog provider aliases
# (for example, ``@hrpolicy``), not raw provider credentials.
PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY")
PORTKEY_VIRTUAL_BACKUP_KEY = os.getenv("PORTKEY_VIRTUAL_BACKUP_KEY")
PORTKEY_VIRTUAL_GUARD_KEY = os.getenv("PORTKEY_VIRTUAL_GUARD_KEY")
PORTKEY_VIRTUAL_JUDGE_KEY = os.getenv("PORTKEY_VIRTUAL_JUDGE_KEY")

# Saved Portkey gateway configs. This workspace blocks inline config JSON, so
# requests must reference a saved config ID (pc-...) instead.
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
PORTKEY_GUARD_CONFIG_ID = os.getenv("PORTKEY_GUARD_CONFIG_ID")
PORTKEY_JUDGE_CONFIG_ID = os.getenv("PORTKEY_JUDGE_CONFIG_ID")
PORTKEY_GATEWAY_TIMEOUT_SECONDS = float(
    os.getenv("PORTKEY_GATEWAY_TIMEOUT_SECONDS", "60")
)
# Retry transient provider failures such as 429 responses. Portkey may also
# retry according to a saved config, so keep this intentionally bounded.
PORTKEY_CLIENT_MAX_RETRIES = int(os.getenv("PORTKEY_CLIENT_MAX_RETRIES", "3"))

# GUARD MODEL 
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"
JUDGE_MODEL_NAME = "openai/gpt-oss-20b"
# EVALUATION MODEL
# Defaults to the main model-catalog alias unless a dedicated judge alias is
# supplied. This keeps evaluation usable with the existing .env setup.
# JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "openai/gpt-oss-20b")

# TRACING 
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")





## DEFINE PATH - DATA / VECTOR STORE 
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")
VECTOR_STORE_PATH = os.path.join("data", "faiss_index")


## VECTORE STORES 
    # IN MEMORY 
    # persistent memory - vectors # 100gb - ingestion 
    # cloud memory 


QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# colection name inside the database Cluster
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hr_policy")



## MODELS 
# LLM and EMBEDING MODEL 
LLM_MODEL_NAME = "openai/gpt-oss-20b"  # from Groq
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"



## CHUNK / TEXT SPLITTING CONFIG 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60


# RETRIVAL RESULTS 
TOP_K_RESULTS = 3


## SYSTEM INSTRUCTIONS 
SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. Answer only with facts explicitly supported by the search "
    "results. Do not infer, embellish, or promise an outcome (for example, approval, "
    "eligibility, or an exception) unless the search results state it. If the answer isn't "
    "in the search results, say you don't know instead of guessing. Keep answers concise "
    "and directly responsive to the employee's question."
)


def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing."""
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to your .env file.")
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL/QDRANT_API_KEY. Please add them to your .env file.")
    parsed_qdrant_url = urlparse(QDRANT_URL)
    if (
        parsed_qdrant_url.scheme not in {"http", "https"}
        or not parsed_qdrant_url.hostname
        or parsed_qdrant_url.hostname.endswith(".example.com")
    ):
        raise ValueError(
            "Invalid QDRANT_URL. Set it to the HTTPS endpoint shown for your "
            "cluster in the Qdrant Cloud dashboard (for example, "
            "https://<cluster-id>.<region>.cloud.qdrant.io)."
        )
    if not PORTKEY_API_KEY:
        raise ValueError("Missing PORTKEY_API_KEY. Please add it to your .env file.")
    if not PORTKEY_VIRTUAL_KEY:
        raise ValueError(
            "Missing PORTKEY_VIRTUAL_KEY. Add the Portkey provider alias "
            "(for example, @my-groq-provider) to your .env file."
        )
    if not PORTKEY_VIRTUAL_KEY.startswith("@"):
        raise ValueError(
            "Invalid PORTKEY_VIRTUAL_KEY. It must be a saved Portkey provider "
            "alias beginning with @ (for example, @my-groq-provider)."
        )
    if not PORTKEY_VIRTUAL_GUARD_KEY:
        raise ValueError(
            "Missing PORTKEY_VIRTUAL_GUARD_KEY. Add the Portkey virtual-key "
            "slug for the guard model (for example, @hrpolicy-guard) to your .env file."
        )
    if not PORTKEY_VIRTUAL_GUARD_KEY.startswith("@"):
        raise ValueError(
            "Invalid PORTKEY_VIRTUAL_GUARD_KEY. It must be a saved Portkey "
            "virtual-key alias beginning with @."
        )
    if PORTKEY_VIRTUAL_JUDGE_KEY and not PORTKEY_VIRTUAL_JUDGE_KEY.startswith("@"):
        raise ValueError(
            "Invalid PORTKEY_VIRTUAL_JUDGE_KEY. When set, it must be a saved "
            "Portkey virtual-key alias beginning with @."
        )
    for setting_name, config_id in {
        "PORTKEY_CONFIG_ID": PORTKEY_CONFIG_ID,
        "PORTKEY_GUARD_CONFIG_ID": PORTKEY_GUARD_CONFIG_ID,
        "PORTKEY_JUDGE_CONFIG_ID": PORTKEY_JUDGE_CONFIG_ID,
    }.items():
        if config_id and not config_id.startswith("pc-"):
            raise ValueError(
                f"Invalid {setting_name}. When set, it must be a saved Portkey "
                "gateway config ID beginning with pc-."
            )
    if PORTKEY_GATEWAY_TIMEOUT_SECONDS <= 0:
        raise ValueError("PORTKEY_GATEWAY_TIMEOUT_SECONDS must be greater than zero.")
    if PORTKEY_CLIENT_MAX_RETRIES < 0:
        raise ValueError("PORTKEY_CLIENT_MAX_RETRIES cannot be negative.")
