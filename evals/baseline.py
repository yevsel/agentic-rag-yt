"""Loop over the goldens and score each one with Faithfulness.

    python -m evals.baseline

This is the same 5 steps as smoke.py, just wrapped in a for-loop.
Set LIMIT = None to run all 20.
"""

import json
import os

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import FaithfulnessMetric

from evals.run import make_test_case, run_graph

LIMIT = 5  # start small (costs money + time); change to None for all 20

# ---- 1. Load the goldens -------------------------------------------------
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="evals/goldens.json")

goldens = dataset.goldens
if LIMIT is not None:
    goldens = goldens[:LIMIT]

# ---- 2. Loop over them ---------------------------------------------------
rows = []

for number, golden in enumerate(goldens, start=1):
    print(f"{number}. {golden.input[:70]}")

    answer, chunks = run_graph(golden.input)
    test_case = make_test_case(golden.input, answer, chunks, golden.expected_output)

    metric = FaithfulnessMetric()
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

# ---- 3. Summarise --------------------------------------------------------
if rows:
    average = sum(row["score"] for row in rows) / len(rows)
    passes = sum(1 for row in rows if row["passed"])
    print()
    print(f"AVERAGE FAITHFULNESS: {average:.2f}")
    print(f"PASSED              : {passes}/{len(rows)}")

    # Save the run so you can compare baselines later
    os.makedirs("evals/runs", exist_ok=True)
    with open("evals/runs/baseline_faithfulness.json", "w") as file:
        json.dump({"average": average, "rows": rows}, file, indent=2)
    print("Saved to evals/runs/baseline_faithfulness.json")
