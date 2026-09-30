"""Connect to the LLM, the brain of the assistant, through Portkey."""


from hr_assistant.gateway import get_gateway_llm, get_gateway_judge_llm
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """Return the Portkey LLM, configured with primary/fallback routing."""
    logger.info("Initializing LLM via Portkey fallback gateway configuration")
    return get_gateway_llm()


def get_judge_llm():
    """Return the Portkey LLM, configured with primary/fallback routing."""
    logger.info("Initializing LLM via Portkey fallback gateway configuration")
    return get_gateway_judge_llm()


# def get_11m():
#     'Return a Groq chat model. Reads GROQ_API_KEY from the environ
#     logger. info("Initializing LLM '%s'", config. LLM_MODEL_NAME)
#     return ChatGroq(model=config.LLM_MODEL_NAME, temperature=0)
