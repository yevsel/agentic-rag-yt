"""Score ONE golden with all five RAG metrics.

    python -m evals.smoke

Costs money: one graph run + five metrics (each metric is a few judge calls).
Run it yourself when you are ready.

Two kinds of metric live here:

    Generator metrics  -> judge the ANSWER your writer produced
        Faithfulness
        AnswerRelevancy

    Retriever metrics  -> judge the CHUNKS your search returned
        ContextualRelevancy
        ContextualPrecision
        ContextualRecall
"""

import json
import os

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel

from evals.run import make_test_case, run_graph

# Local judge: free. Swap to a stronger model for numbers you publish.
JUDGE = OllamaModel(model="llama3.1:8b")

# Which golden to score. goldens.json has 20, so 0-19 are all valid.
GOLDEN_INDEX = 0

# ---- 1. Load the goldens -------------------------------------------------
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="evals/goldens.json")
golden = dataset.goldens[GOLDEN_INDEX]

print("QUESTION:", golden.input)
print()

# ---- 2. Run the graph ----------------------------------------------------
answer, chunks = run_graph(golden.input)
print("ANSWER :", answer[:300])
print("CHUNKS :", len(chunks))
print()

# ---- 3. Combine into a test case -----------------------------------------
test_case = make_test_case(
    question=golden.input,
    answer=answer,
    chunks=chunks,
    expected_output=golden.expected_output,
)

# ---- 4. The five metrics -------------------------------------------------
# One list, five rulers. Same method on each: .measure(test_case)
metrics = [
    FaithfulnessMetric(model=JUDGE),         # is every claim supported by the chunks?
    AnswerRelevancyMetric(model=JUDGE),      # is the answer about the question?
    ContextualRelevancyMetric(model=JUDGE),  # how much of the context is useful?
    ContextualPrecisionMetric(model=JUDGE),  # are the best chunks ranked first?
    ContextualRecallMetric(model=JUDGE),     # did we retrieve everything needed?
]

# ---- 5. Score it ---------------------------------------------------------
print("=" * 70)

rows = []

for metric in metrics:
    metric.measure(test_case)

    name = type(metric).__name__
    print(name)
    print("  score:", metric.score)
    print("  pass :", metric.is_successful())
    print("  why  :", metric.reason)
    print()

    rows.append(
        {
            "metric": name,
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason,
        }
    )

print("=" * 70)
print("Default pass mark is 0.5 for every metric.")

# ---- 6. Save the results -------------------------------------------------
os.makedirs("evals/runs", exist_ok=True)

result = {
    "question": golden.input,
    "expected_output": golden.expected_output,
    "answer": answer,
    "chunk_count": len(chunks),
    "metrics": rows,
}

with open("evals/runs/smoke.json", "w") as file:
    json.dump(result, file, indent=2)

print("Saved to evals/runs/smoke.json")
