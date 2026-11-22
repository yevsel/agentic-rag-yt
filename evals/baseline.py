"""Loop over the goldens and score each one with Faithfulness.

    python -m evals.baseline

This is the same steps as smoke.py, just wrapped in a for-loop.
Set LIMIT = None to run all 20.
"""

import json
import os

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel

from evals.run import make_test_case, run_graph

# Local judge: free. Swap to a stronger model for numbers you publish.
JUDGE = OllamaModel(model="llama3.1:8b")

LIMIT = None  # None = all 20 goldens

# ---- 1. Load the goldens -------------------------------------------------
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="evals/goldens.json")

goldens = dataset.goldens
if LIMIT is not None:
    goldens = goldens[:LIMIT]

# ---- 2. Loop over them ---------------------------------------------------
rows = []
os.makedirs("evals/runs", exist_ok=True)

for number, golden in enumerate(goldens, start=1):
    print(f"{number}. {golden.input[:70]}")

    try:
        answer, chunks = run_graph(golden.input)
        test_case = make_test_case(golden.input, answer, chunks, golden.expected_output)

        metric = FaithfulnessMetric(model=JUDGE)
        metric.measure(test_case)

        passed = metric.is_successful()
        print(f"   score={metric.score:.2f}  pass={passed}")

        rows.append(
            {
                "input": golden.input,
                "score": metric.score,
                "passed": passed,
                "reason": metric.reason,
            }
        )

    # One bad question should not kill the whole run.
    except Exception as error:
        print(f"   FAILED: {error}")
        rows.append(
            {
                "input": golden.input,
                "score": None,
                "passed": False,
                "reason": f"error: {error}",
            }
        )

    # Save after every golden, so nothing is lost if it stops halfway.
    with open("evals/runs/baseline_faithfulness.json", "w") as file:
        json.dump({"rows": rows}, file, indent=2)

# ---- 3. Summarise --------------------------------------------------------
scores = [row["score"] for row in rows if row["score"] is not None]
passes = sum(1 for row in rows if row["passed"])

print()
print(f"SCORED: {len(scores)} of {len(rows)}")

if scores:
    average = sum(scores) / len(scores)
    print(f"AVERAGE FAITHFULNESS: {average:.2f}")
    print(f"PASSED              : {passes}/{len(scores)}")

print("Saved to evals/runs/baseline_faithfulness.json")
