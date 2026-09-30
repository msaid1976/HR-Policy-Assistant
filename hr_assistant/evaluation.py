"""
Step 9: evaluate answer quality against a fixed set 
of test questions.

Unlike tracing (which just records what happened), 
evaluation runs the
agent against a known set of question/reference-answer 
pairs and scores
each answer using a second LLM as a judge.
Results are uploaded to
LangSmith as a Dataset + Experiment, 
so quality can be compared across
runs (after a prompt change, a new model, a new guardrail, etc).

The judge model is routed through Portkey too,
using the same slug as
the main app's LLM (gateway.py's PRIMARY_PROVIDER) 
but a different
underlying model (JUDGE_MODEL_NAME) - 
so it isn't grading its own
output verbatim, without needing a second slug set up.
"""


import re
import time
from collections.abc import Callable
from typing import Any

from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

from hr_assistant.gateway import get_gateway_judge_llm
from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask, build_hr_assistant
from hr_assistant.vector_store import get_retriever, load_vector_store



logger = get_logger(__name__)

# question paper 
DATASET_NAME = "hr-policy-qna"
JUDGE_MAX_ATTEMPTS = 4

TEST_CASES = [
    {"question": "How many days of paid annual leave do I get per year?",
    "answer": "20 days"},
    {"question": "How many days of unused annual leave can be carried forward?", "answer": "Up to 5 days"},
    {"question": "How many paid sick days do I get per year?", "answer": "10 days"},
    {"question": "How many days per week can I work from home?", "answer": "Up to 2 days, with manager approval"},
    {"question": "How long is the probation period?",
    "answer": "3 months"},
    {"question": "What is the notice period during probation?", "answer": "15 days"},
    {"question": "What is the standard notice period for resignation?",
    "answer": "30 days"},
    {"question": "Within how many days must reimbursement claims be submitted?",
    "answer": "30 days of the expense"},
    {"question": "How many public holidays does the company observe each year?", "answer": "12"},
    {"question": "Within how many days is full and final settlement processed after the last working day?", "answer": "45 days"},
]


# if dataset is there reuse it , if not create a new dataset 
# question paper 
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", 
        DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
            "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset


def _is_rate_limit_error(error: Exception) -> bool:
    """Identify provider 429 failures without coupling to one SDK exception type."""
    message = str(error).lower()
    return "429" in message or "rate limit" in message


def _retry_delay_seconds(error: Exception) -> float:
    """Use the provider's suggested delay when it is included in the error text."""
    match = re.search(r"try again in\s+([0-9.]+)s", str(error), re.IGNORECASE)
    return max(1.0, float(match.group(1)) + 0.5) if match else 2.0


def _run_judge_with_retries(
    metric: str, judge: Callable[..., dict[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    """Retry only transient judge rate limits, then let other failures surface."""
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            return judge(**kwargs)
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt == JUDGE_MAX_ATTEMPTS:
                raise
            delay = _retry_delay_seconds(error)
            logger.warning(
                "%s judge was rate limited; retrying in %.1f seconds (%d/%d)",
                metric,
                delay,
                attempt,
                JUDGE_MAX_ATTEMPTS - 1,
            )
            time.sleep(delay)

# start the exam
def run_evaluation():
    """Upload the dataset (if needed) 
    and run the correctness evaluation."""
    client = Client()
    dataset = _ensure_dataset(client)

    # Built once and reused for every test case, instead of rebuilding
    # the whole agent (and reconnecting to Qdrant) 10 times over.
    agent = build_hr_assistant()
    retriever = get_retriever(load_vector_store())

    # write the answers 
    def target(inputs: dict) -> dict:
        """
        Run one test question through the real agent, 
        and also capture
        the retrieved chunks 
        so groundedness can check the answer against
        what was actually retrieved 
        (not just the reference answer).
        """
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
        return {"answer": answer, "context": context}

    # giving marks 
    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=get_gateway_judge_llm(),
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=get_gateway_judge_llm(),
    )

    coverage = {
        "correctness": {"scored": 0, "unavailable": 0},
        "groundedness": {"scored": 0, "unavailable": 0},
    }

    def unavailable(metric: str, reason: str) -> dict:
        coverage[metric]["unavailable"] += 1
        return {"key": metric, "score": None, "comment": reason}

    def correctness_evaluator(
        inputs: dict,
        outputs: dict | None,
        reference_outputs: dict | None,
        **kwargs: Any,
    ) -> dict:
        """Score answer correctness, recording unavailable scores explicitly."""
        if not outputs or "answer" not in outputs:
            return unavailable("correctness", "Skipped because the target run produced no answer.")
        try:
            result = _run_judge_with_retries(
                "correctness",
                correctness_judge,
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
            )
        except Exception as error:
            logger.exception("Correctness judge failed after retries: %s", error)
            return unavailable("correctness", f"Judge unavailable after retries: {error}")
        coverage["correctness"]["scored"] += 1
        return result

    def groundedness_evaluator(outputs: dict | None, **kwargs: Any) -> dict:
        """Check the answer is supported by the retrieved context, not invented."""
        if not outputs or "answer" not in outputs or "context" not in outputs:
            return unavailable(
                "groundedness",
                "Skipped because the target run did not produce an answer and context.",
            )
        try:
            result = _run_judge_with_retries(
                "groundedness",
                groundedness_judge,
                outputs={"answer": outputs["answer"]},
                context=outputs["context"],
            )
        except Exception as error:
            logger.exception("Groundedness judge failed after retries: %s", error)
            return unavailable("groundedness", f"Judge unavailable after retries: {error}")
        coverage["groundedness"]["scored"] += 1
        return result

    def evaluation_coverage(runs: list, examples: list) -> dict:
        """Expose score availability so averages cannot hide missing evaluations."""
        total = len(examples)
        results = []
        for metric, counts in coverage.items():
            scored = counts["scored"]
            unavailable_count = counts["unavailable"]
            results.append(
                {
                    "key": f"{metric}_coverage",
                    "score": scored / total if total else None,
                    "comment": f"{scored}/{total} scored; {unavailable_count} unavailable.",
                }
            )
        return {"results": results}

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        summary_evaluators=[evaluation_coverage],
        experiment_prefix="hr-policy-evalzz",
        description="HR policy assistant correctness + groundedness evaluation",
        # The Groq route is limited to 8,000 tokens/minute. Serial evaluation
        # prevents target and judge calls from creating a burst beyond that cap.
        max_concurrency=1,
    )
