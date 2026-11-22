"""Everything in one file, so you can read it top to bottom.

    python -m evals.inspect_demo
    deepeval inspect

Costs money: 2 graph runs + 3 metrics each.
"""

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.tracing import observe, update_current_trace
from ..graph import graph

# Local judge: free. Swap to a stronger model for numbers you publish.
JUDGE = OllamaModel(model="llama3.1:8b")

LIMIT = 5  # how many goldens to score


# ---- 1. Ask the graph one question ---------------------------------------
# Returns the answer text, and the chunks the answer was written from.
def ask_graph(question):
    state = graph.invoke(
        {"question": question},
        {"recursion_limit": 40},
    )

    answer = state.get("answer", "")

    # state["chunks"] is a list of dicts like
    #   {"blog_post": "diffusion-models", "text": "Sora uses DiT..."}
    # DeepEval wants plain strings, so take just the "text" of each one.
    chunks = []
    for chunk in state.get("chunks", []):
        chunks.append(chunk["text"])

    return answer, chunks


# ---- 2. Run the graph inside a trace -------------------------------------
# @observe turns this function into a trace that DeepEval can save and show.
@observe(type="agent")
def run_and_trace(golden):
    answer, chunks = ask_graph(golden.input)

    # Tell DeepEval what happened. These are the four things metrics judge.
    update_current_trace(
        input=golden.input,
        output=answer,
        retrieval_context=chunks,
        expected_output=golden.expected_output,
    )

    return answer


# ---- 3. Load the goldens -------------------------------------------------
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="evals/goldens.json")

goldens = dataset.goldens[:LIMIT]
dataset.goldens = goldens

# ---- 4. For each golden: run the graph, then score that run --------------
metrics = [
    FaithfulnessMetric(model=JUDGE),
    AnswerRelevancyMetric(model=JUDGE),
    ContextualRelevancyMetric(model=JUDGE),
]

for golden in dataset.evals_iterator(metrics=metrics):
    run_and_trace(golden)

print()
print("Saved to .deepeval/  ->  now run:  deepeval inspect")
