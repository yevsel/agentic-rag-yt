# DeepEval Glossary — Plain English

Every term you will meet while working through these evals, explained the simple way.

**How to read this:** each term gets a one-line meaning, then why it matters for *this* project.
If a term is just a name, it still gets a line — you will see it in the docs.

---

## 1. The big ideas

| Term | Plain meaning |
|---|---|
| **LLM-as-a-judge** | Using an AI model to grade another AI's output. Instead of writing exact-match rules, you ask a model "is this answer supported by these chunks?" and it gives a score. |
| **Metric** | A ruler. It looks at a test case and returns a number between 0 and 1, plus a written reason. |
| **Threshold** | The pass mark for a metric. Default is `0.5`. Score ≥ threshold means pass. |
| **Eval / evaluation** | A test that measures quality instead of checking exact output. |
| **Regression** | Something that used to pass and now fails. The main thing evals are for. |

---

## 2. The four objects you touch

### **Golden**
A test case you wrote **before** running your app. It holds the question and the ideal answer.
It has **no** `actual_output` and **no** `retrieval_context` yet — those come from your app.

> In this project: `evals/goldens.json` holds 30 of them.

### **LLMTestCase**
A golden **plus** what your app actually produced. This is the object metrics score.
Only one field is required: `input`. Everything else is optional.

> In this project: `make_test_case()` in `evals/run.py` builds one.

### **EvaluationDataset**
A box that holds a list of goldens. It can load them from JSON, JSONL or CSV.

> In this project: `dataset.add_goldens_from_json_file("evals/goldens.json")`.

### **Metric**
The ruler. See section 4 for what it returns.

### **Test run**
One pass of metrics over one or more test cases. Saved locally, and optionally uploaded to Confident AI.

---

## 3. The fields inside a test case

| Field | Plain meaning | Example here |
|---|---|---|
| `input` | The question the user asked. | `"What are the three components of ACE?"` |
| `actual_output` | The answer **your app** produced. | `state["answer"]` |
| `expected_output` | The answer you wish it had produced. | From the golden. Needed only by some metrics. |
| `retrieval_context` | The list of text chunks your retriever found **for this question**. | `state["chunks"]` → list of strings |
| `context` | Ground-truth source text, used when generating datasets. **Not** for scoring. | The whole blog post |
| `metadata` | Any extra info you want to attach. Not scored. | `{"intent": "in_scope", "searches": 2}` |

**The one mistake to never make:** using `context` (whole document) where `retrieval_context` (the 5 chunks returned) belongs. It makes recall look perfect and relevancy look terrible, and both numbers become meaningless.

---

## 4. What a metric gives back, and the knobs

### **score**
A number from 0 to 1. Higher is better, for every metric.

### **reason**
A written explanation of the score. **This is the useful part.** The number tells you something is wrong; the reason tells you what.

### **is_successful()**
Returns `True` if `score >= threshold`. This is what fails a test in CI.

### **threshold**
The pass mark. Default `0.5`. Set it from a measured baseline, not a guess.

### **strict_mode**
`True` gives a hard 0 or 1 instead of a score, and forces the threshold to 1. Use it when there is no middle ground.

### **verbose_mode**
`True` prints the judge's intermediate steps. Your debugger when a score surprises you.

### **flaky**
`True` keeps the score and reports it, but its failure never fails the test. Use it for a metric you want to watch but do not trust yet.

### **reference-based vs referenceless**
- **Reference-based** needs `expected_output`. Here: ContextualPrecision, ContextualRecall.
- **Referenceless** judges without an ideal answer. Here: Faithfulness, AnswerRelevancy, ContextualRelevancy.

Referenceless metrics can also run on real production traffic, where no ideal answer exists.

### **judge model**
The LLM doing the grading. OpenAI by default. Change it per metric with `model=`.
**Never compare two runs scored by different judges.**

---

## 5. The five RAG metrics

### **Faithfulness**
*"Is every claim in the answer supported by the retrieved chunks?"*
It pulls the claims out of the answer, pulls the facts out of the chunks, and compares them.
Low score means the **writer made something up**.

### **AnswerRelevancy**
*"Is the answer actually about the question?"*
It splits the answer into statements and asks whether each one is relevant to the input.
Low score means **padding, waffle, or answering a different question**.

### **ContextualRelevancy**
*"How much of the retrieved context is useful?"*
It judges the chunks themselves for relevance to the question.
Low score means **too much noise in the context**.

### **ContextualPrecision**
*"Are the good chunks ranked above the bad ones?"*
It rewards relevant chunks for appearing **early**.
Low score means **relevant chunks are buried**. This is the metric that reacts to reranking.

### **ContextualRecall**
*"Did retrieval bring back everything the ideal answer needed?"*
It breaks the ideal answer into statements and checks whether the chunks can support them.
Low score means **retrieval missed information** — small `k`, bad chunking, weak embeddings.

### Quick diagnosis table

```
Faithfulness low      → writer invented claims
AnswerRelevancy low   → answer is padded or off-topic
Recall low            → retrieval is missing information
Precision low         → good chunks are buried
Relevancy low         → too much noise in context
All high but wrong    → blame your goldens or your judge
```

---

## 6. Custom metrics

### **G-Eval**
The general-purpose judge. You describe **in plain English** what "good" means, and it scores against that.

### **evaluation_steps**
A fixed list of steps for G-Eval. Write them once and reuse them, instead of describing criteria and letting the model invent steps each time. Fixed steps mean stable scores.

### **criteria**
A free-text description of what good means. Use `criteria` **or** `evaluation_steps`, never both.

### **Rubric**
Score bands for G-Eval, e.g. "0–2 = wrong, 3–5 = partial, 6–8 = good". Stops the judge clustering everything around the middle.

### **DAGMetric**
A decision tree of small yes/no questions that map to fixed scores. Use it when the rule is objective ("did it include a citation? then judge tone"). More predictable than G-Eval.

### **Expectations**
Plain-language rules attached to a golden, checked without an LLM:

```python
Expectations(
    must=["State that the post does not give Sora's parameter count."],
    must_not=["State a specific parameter count."],
)
```

> In this project: all 10 DECLINE goldens use these.

---

## 7. Tracing — finding *where* quality dropped

| Term | Plain meaning |
|---|---|
| **trace** | The full record of one app run, from question to answer. |
| **span** | One step inside a trace — one node, one model call, one tool call. |
| **@observe** | A decorator that records a function as a span, and can attach a metric to it. |
| **update_current_span()** | Inside a span, say what its input and output were, so a metric can judge it. |
| **update_current_trace()** | Same idea, but for the whole run at once. |
| **CallbackHandler** | A LangChain/LangGraph hook that records every node and model call automatically — no code changes inside your nodes. |

### Trajectory metrics (judge the *path*, not the answer)

| Metric | Asks |
|---|---|
| **TaskCompletion** | Did the agent finish the task? |
| **StepEfficiency** | Did it take more steps than needed? (a cost signal) |
| **PlanAdherence** | Did it follow its own plan? |
| **PlanQuality** | Was the plan a good plan? |

These were built for free-roaming agents. This graph is a fixed workflow, so treat them as secondary.

---

## 8. Ways to run a metric

| Term | Plain meaning |
|---|---|
| **measure(tc)** | Score one test case, right now. Best for debugging. |
| **a_measure(tc)** | The async version. Use with `asyncio.gather` to run several at once. |
| **evaluate(test_cases, metrics)** | Score a whole list. Good for scripts and notebooks. Adds caching, parallelism and cost tracking. |
| **assert_test(test_case, metrics)** | Like `evaluate`, but raises an error on failure. This is the CI gate. |
| **deepeval test run file.py** | Runs your pytest file through DeepEval's runner. |
| **caching** | Reuses scores for identical test cases. `deepeval test run -c`. Saves money on re-runs. |
| **Confident AI** | DeepEval's optional cloud dashboard. `deepeval login`. Everything works locally without it. |
| **AsyncConfig** | A small settings object that controls whether a run is async. `AsyncConfig(run_async=False)` for sync apps. |

---

## 9. RAG words you will keep meeting

| Term | Plain meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation. Look things up first, then answer using what you found. |
| **chunk** | A small piece of a document. We cut posts into ~1000-character pieces. |
| **embedding** | A list of numbers that represents meaning. Similar text gets similar numbers. |
| **retrieval** | Finding the chunks whose embeddings are closest to the question. |
| **vector store** | The database that holds embeddings. Here: Pinecone. |
| **reranking** | Taking the top 20 results and reordering them so the best are first. Improves precision. |
| **grounding** | Keeping the answer tied to the source text. Ungrounded = made up. |
| **hallucination** | A confident answer with no support in the sources. |
| **abstention** | Correctly refusing to answer because the sources do not contain it. |
| **precision** | Of the things we returned, how many were useful? |
| **recall** | Of the useful things that exist, how many did we return? |

---

## 10. Words specific to this project

| Term | Plain meaning |
|---|---|
| **gate** | A checkpoint where an LLM decides whether to continue. This graph has three. |
| **Adaptive RAG** | Deciding **whether** to retrieve at all. Here: `classify_node` may skip search and answer directly. |
| **Corrective RAG** | Checking whether the retrieved chunks are good enough, and searching again with a better query if not. Here: `grade_node` → `refine_node`. |
| **Self-RAG** | Checking the final answer against the chunks, fixing it, or refusing. Here: `check_answer_node` → `fix_answer_node` / `give_up_node`. |
| **DECLINE** | Our marker in `goldens.json` meaning "the correct behaviour is to refuse". `expected_output == "DECLINE"`. |
| **verified** | A state field. `True` if the Self-RAG check passed. |
| **refused** | A state field. `True` if the graph decided not to answer. |
| **searches / attempts / repairs** | Counters in state. They enforce the loop limits (4, 2, 2). Deterministic — assert them, do not judge them. |
| **recursion_limit** | The maximum number of steps LangGraph will take before it stops. Stops infinite loops. |

---

## The 30-second summary

> A **golden** is a question plus its ideal answer.
> Running the app fills in the **actual_output** and **retrieval_context**, and together they make an **LLMTestCase**.
> A **metric** scores that test case from 0 to 1 and explains why.
> A **threshold** turns the score into pass or fail.
> **Faithfulness** asks "did it stick to the sources?", **Recall** asks "did we find everything?", **Precision** asks "were the best chunks first?".
