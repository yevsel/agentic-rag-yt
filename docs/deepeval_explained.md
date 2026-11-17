# DeepEval Explained: Adding Evals to This Agentic RAG

*A hands-on companion to [`deepeval_intro.md`](./deepeval_intro.md). That file is the reference — API surface, metric internals, pitfalls. This file is the onboarding path: why evals matter here, what to measure on this specific graph, and how to sequence the work into a tutorial.*

*Everything below was verified against **deepeval 4.3.4** installed in this project's venv.*

**Reading order:** Part 0 is the first-time primer — read that and you can start working. Parts 1–10 build the full picture and the tutorial plan.

---

## Part 0 — Start here (first time with DeepEval)

## 0.1 The one idea

**A metric is an LLM prompt plus a threshold.**

That is literally it. `FaithfulnessMetric` is a prompt that says "extract the claims in the answer, extract what the context supports, compare them" — run against an LLM judge — producing a number 0–1 and a written reason.

Once that clicks, nothing about DeepEval is mysterious. Everything else is plumbing around that idea.

## 0.2 The loop you run forever

```
goldens (JSON)  →  run your app  →  LLMTestCase  →  metric.measure()  →  assert score ≥ threshold
```

Four steps. Every API you touch feeds one of them.

## 0.3 The test case: only four fields matter

```python
LLMTestCase(
    input="What are the three components of ACE?",
    actual_output=state["answer"],           # what your graph returned
    retrieval_context=[c["text"] for c in state["chunks"]],
    expected_output="...",                   # only Precision, Recall, G-Eval need this
)
```

- `retrieval_context` must be `list[str]` of the chunks the generator actually saw.
- `metadata={}` is free-form — store the expected route there for the Adaptive gate test.

## 0.4 The metric: three things it gives you, three knobs

```python
m.measure(tc)
m.score            # 0..1
m.reason           # why — the part that teaches you something
m.is_successful()  # score >= threshold
```

```python
FaithfulnessMetric(
    threshold=0.7,      # default 0.5 — a placeholder, not a recommendation
    verbose_mode=True,  # print the intermediate judge steps  ← your debugger
    strict_mode=False,  # True → binary 0/1, threshold forced to 1
)
```

Also available: `flaky=True` (scores and reports, never fails the test) and `threshold=None` (score-only, never gates).

**Set thresholds from a baseline, not from vibes.** Run once, look at the mean, then set each threshold slightly below it so it catches regressions without flapping.

## 0.5 The five traps that catch everyone

1. **Using the golden's `context` as `retrieval_context`.** Goldens have `context` (source text for dataset generation). Test cases need `retrieval_context` (live retriever output). Different fields, different objects.
2. **Judging retrieval metrics on unanswerable cases.** On the 10 give-up cases, `retrieval_context` is *correctly* irrelevant. Gate with `if expected_output != "DECLINE"`.
3. **Trusting a single run.** LLM judges are non-deterministic. Run the baseline twice — the gap is your noise floor. A 0.03 gain means nothing if noise is 0.06.
4. **Forgetting what a metric costs.** Faithfulness alone is ~3 judge calls (extract claims → extract truths → compare). Five metrics × 30 goldens ≈ hundreds of calls. Use a cheap judge via `model=` while iterating, the strong one for published numbers. Never compare runs scored by different judges.
5. **Missing `expected_output` on Precision/Recall.** They are reference-based and will throw. The other three do not need it.

## 0.6 What you actually type

**Debug one case:**

```python
m = FaithfulnessMetric(verbose_mode=True)
m.measure(tc)
print(m.score, m.reason)
```

**Explore a set:**

```python
from deepeval import evaluate
evaluate(test_cases=[tc1, tc2], metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()])
```

**Gate in CI** — pytest is the interface:

```python
from deepeval import assert_test

@pytest.mark.parametrize("golden", dataset.goldens)
def test_rag(golden):
    answer, chunks = run_graph(golden.input)
    assert_test(
        test_case=LLMTestCase(input=golden.input, actual_output=answer,
                              expected_output=golden.expected_output,
                              retrieval_context=chunks),
        metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()],
    )
```

```bash
deepeval test run tests/test_rag.py
```

| CLI flag | Use |
|---|---|
| `-c` | use cache — don't pay twice for the same case |
| `-i` | ignore errors (judge returned bad JSON) |
| `-v` | verbose |
| `-d failing` | only show failures |
| `-n 4` | parallel |

`deepeval login` is **optional** — it is for the Confident AI cloud dashboard. Everything works locally without it.

## 0.7 Reading a bad score

This is the skill, not the API:

```
Faithfulness low, AnswerRelevancy fine  → generator invented claims
AnswerRelevancy low                     → padded / evasive answers
ContextualRecall low                    → retrieval is missing information (k, chunking, embeddings)
ContextualPrecision low                 → relevant chunks are buried (a reranker fixes this)
ContextualRelevancy low                 → too much noise in context
Everything high but answers still wrong → suspect your goldens or your judge
```

When a score surprises you: `verbose_mode=True` and **read the reason**. The number is the smoke alarm; the reason is the fire.

## 0.8 What maps to a metric — and what doesn't

| You want to know | Tool | Reads from |
|---|---|---|
| Did the writer stick to the chunks? | Faithfulness | `state["answer"]` + `state["chunks"]` |
| Did retrieval find the right stuff? | ContextualPrecision / Recall | `state["chunks"]` |
| Is the context noisy? | ContextualRelevancy | `state["chunks"]` |
| Did it correctly refuse? | custom G-Eval abstention | `state["answer"]` + `state["refused"]` |
| Did classify route correctly? | **no metric — plain assert** | `state["intent"]` |
| Did the loops respect their limits? | **no metric — plain assert** | `state["searches"]`, `repairs`, `attempts` |

The last two rows are the ones people forget. **Not everything needs a judge.** Routing and counters are deterministic — assert them like normal software.

## 0.9 The 60-second version

> DeepEval = pytest where the assertion is an LLM judge.
> A test case is `input` + `actual_output` + `retrieval_context`.
> A metric returns `score`, `reason`, `is_successful()`.
> Set thresholds from a measured baseline, run everything twice to find your noise floor, and read the reasons.

---

## Part 1 — What DeepEval is

DeepEval is **pytest for LLM apps**. That framing matters:

- **pytest:** you write an input, you assert on the output, CI fails on regression.
- **DeepEval:** the assertion is an **LLM judge** that returns a 0–1 score plus a written reason.

It is **not** logging or monitoring. It is a regression gate. That is the whole value: this project's three gates are LLM judges deciding whether to answer — DeepEval is the second, independent judge deciding whether those gates worked.

### The three objects you actually touch

| Object | You give it | You get back |
|---|---|---|
| `LLMTestCase` | input, actual_output, retrieval_context, expected_output | a unit to score |
| `Metric` | a threshold | `.score`, `.reason`, `.is_successful()` |
| `EvaluationDataset` | a JSON file of `Golden`s | something to loop / parametrize over |

`Golden` = the *pending* test case (question + ideal answer). `LLMTestCase` = golden + what the app actually produced.

> **The one thing people get wrong:** `retrieval_context` is what the retriever returned on *this run*, never the golden's source text. On a `Golden`, the equivalent field is `context`, which is ground truth used for dataset generation. Different thing.

---

## Part 2 — Why this graph needs it

The graph has three LLM gates, each making a judgement call:

```
classify  → is this in scope?
grade     → are these chunks enough?
check     → is every claim supported?
```

Today the only way to know they work is to ask questions manually and read answers. That cannot tell you:

- whether `grade` is too lenient (passing thin evidence)
- whether `check` is too strict (rejecting good answers → burns repairs → `give_up`)
- whether `give_up` fires on questions the corpus *could* have answered

Those are all **false positives / false negatives on a judge**. That is exactly what an eval suite measures.

---

## Part 3 — The five RAG metrics, mapped to the 11 nodes

| Metric | Judges | Node | Low score means |
|---|---|---|---|
| **AnswerRelevancy** | generator | `write_answer_node` | padded, evasive or off-topic answer |
| **Faithfulness** | generator | `write_answer_node` | **the writer invented a claim not in the chunks** |
| **ContextualRelevancy** | retriever | `tools_node` | chunks are noise — too much junk in context |
| **ContextualPrecision** | retriever | `tools_node` | relevant chunks buried below irrelevant ones |
| **ContextualRecall** | retriever | `tools_node` | retrieval missed something the ideal answer needs |

Three things to internalise:

1. **Faithfulness low ≠ retrieval problem.** It means the *generator* invented something. `check_answer_node` exists to catch this — so if Faithfulness is low on a case the gate *passed*, the gate has a hole.
2. **Precision is the reranking metric.** It rewards relevant chunks appearing early, so it moves when a reranker is added.
3. **Recall is the gap metric.** Low recall means `k=5` is too small, or chunking/embeddings miss the concept.

Plus one **custom** metric that is unique to this project: **correct abstention** on `give_up_node`.

### Reference vs referenceless

| Needs `expected_output` | Metrics |
|---|---|
| No | Faithfulness, AnswerRelevancy, ContextualRelevancy |
| Yes | ContextualPrecision, ContextualRecall |

The referenceless three can also run on production traces. The other two are development-time only.

---

## Part 4 — The smallest thing that runs

```python
# evals/run.py
from graph import graph
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric


def run_graph(question: str):
    state = graph.invoke({"question": question})
    answer = state.get("answer", "")
    chunks = [c["text"] for c in (state.get("chunks") or [])]
    return answer, chunks


answer, chunks = run_graph("What are the three components of ACE?")

tc = LLMTestCase(
    input="What are the three components of ACE?",
    actual_output=answer,
    retrieval_context=chunks,
)

m = FaithfulnessMetric(threshold=0.7, verbose_mode=True)
m.measure(tc)
print(m.score, m.is_successful())
print(m.reason)          # the part that actually teaches you something
```

Note `c["text"]` — `AgenticGraphState.chunks` stores `{"blog_post", "text"}` dicts (`state.py:14`), so `d.page_content` from the reference doc would crash here.

**That is the whole loop:** golden in, graph out, metric scores it.

### The three ways to run it

| Way | Use it for |
|---|---|
| `m.measure(tc)` | debugging one case at a time |
| `evaluate(test_cases=[...], metrics=[...])` | scripts and notebooks |
| `assert_test(tc, metrics=[...])` + `deepeval test run file.py` | CI and regression gates |

`evaluate()` and `assert_test()` are the same engine. Use `evaluate()` to explore, `assert_test()` to gate. Both give caching, parallelism, cost tracking and error handling that raw `measure()` loops do not.

---

## Part 5 — Three levels, three different questions

| Level | Question it answers | Cost to set up |
|---|---|---|
| **End-to-end** | Is the final answer good? | one function, zero instrumentation |
| **Component** | *Where* did quality drop? | `@observe` on the nodes |
| **Trajectory** | Was the *path* efficient? | pass DeepEval's callback to `graph.invoke` |

Start end-to-end. A single end-to-end score cannot tell you whether a bad answer came from retrieval, generation or a gate — but it tells you *that* something is wrong, which is enough to start.

### Component level

Every Corrective retry becomes its own span with its own score, so you can see whether `refine_node` improved the context or wasted a round.

```python
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ContextualRelevancyMetric


@observe(metrics=[ContextualRelevancyMetric()])
def tools_node(state):
    ...  # existing body
    update_current_span(
        input=query,
        retrieval_context=[c["text"] for c in chunks],
    )
    return {...}
```

### Trajectory level

No code changes inside the nodes:

```python
from deepeval.integrations.langchain import CallbackHandler

graph.invoke({"question": q}, config={"callbacks": [CallbackHandler()]})
```

Trajectory metrics (`TaskCompletion`, `StepEfficiency`, `PlanAdherence`, `PlanQuality`) were designed for tool-using agents. This graph is a fixed workflow with bounded loops, so treat them as secondary. `StepEfficiency` is still useful as a cost signal.

---

## Part 6 — What makes this project's eval story different

Everyone does "here's Faithfulness on a chatbot". This graph has two things they do not.

### 1. The meta-eval

`check_answer_node` is an LLM judge. DeepEval's Faithfulness is a **second, independent** judge. Compare them on the same answers:

| Gate verdict | Faithfulness | Meaning |
|---|---|---|
| pass | high | ✅ both agree it is grounded |
| pass | **low** | ❌ **false negative — the gate has a hole** |
| fail | high | ❌ false positive — the gate is too strict |
| fail | low | ✅ gate did its job |

This turns the gate from "trust me, it verifies" into a measured claim.

### 2. Eval finds a real bug in the graph

`tools_node` **accumulates** chunks (`nodes.py:135`):

```python
"chunks": list(state.get("chunks") or []) + chunks
```

and `write_answer_node` formats **all of them** (`nodes.py:217`). Worst case: 4 searches/round × 5 chunks × 3 rounds = **~60 chunks, duplicates included**, all stuffed into the writer prompt.

ContextualRelevancy and Precision will score this as a retrieval failure. It is not — it is an aggregation failure. The tutorial arc: show the low score, diagnose it, add a `final_chunks` selection step, re-run, show the delta.

### 3. Counter assertions

The loop budgets are testable without any LLM judge at all. Plain asserts on state:

| Limit | Value | Where |
|---|---|---|
| `MAX_SEARCHES` | **4** | `nodes.py:104` |
| `MAX_CORRECTIVE_ATTEMPTS` | **2** | `edges.py:16` |
| `MAX_REPAIRS` | **2** | `edges.py:26` |

---

## Part 7 — Suggested tutorial structure

| Ep | Title | What's on screen |
|---|---|---|
| 1 | **Why your RAG gates need a judge** | the 3 gates, 30 questions, "how do you know?" |
| 2 | **Your first metric** | `run_graph`, one `LLMTestCase`, Faithfulness + `.reason` |
| 3 | **The five RAG metrics** | score a small set, read each reason, build the failure table |
| 4 | **Goldens** | 30 cases: 15 single / 5 multi / **10 unanswerable** |
| 5 | **The abstention metric** | the `give_up` path, both failure directions |
| 6 | **Component tracing** | `@observe`, per-retry relevancy, "did refine help?" |
| 7 | **The meta-eval** | gate vs Faithfulness, the 2×2 table |
| 8 | **Eval-driven improvement** | rerank → precision up; the 60-chunk bug → fixed |
| 9 | **CI** | `deepeval test run` in a PR, gate the merge |

Episodes 7 and 8 are the ones worth sharing.

---

## Part 8 — The 30-golden set

The reference doc's recommended split:

| Type | Count | Purpose |
|---|---|---|
| Answerable from one chunk | 15 | baseline retrieval + generation |
| Answerable only by combining chunks | 5 | stresses recall and the Corrective gate |
| **Unanswerable** | **10** | tests the Self-RAG give-up path |

The unanswerable set **must be hand-written**, including near-misses: on-topic questions the posts do not actually support. The Synthesizer is grounded in the documents, so it will not produce questions the corpus cannot answer.

```json
[
  {
    "input": "<question answerable from the two posts>",
    "expected_output": "<ideal answer>",
    "context": ["<source passage>"]
  },
  {
    "input": "Who won the 2022 World Cup?",
    "expected_output": "DECLINE",
    "expectations": {
      "must": ["State that the sources do not contain the answer."],
      "must_not": ["Give a substantive answer from outside knowledge."]
    }
  }
]
```

---

## Part 9 — Current blockers

| Blocker | Status | Unblocks |
|---|---|---|
| `OPENAI_API_KEY` returns **401** | expired — needs a fresh key | every metric call *and* every `graph.invoke` |
| No local corpus | `ingestion.py` scrapes and discards; Synthesizer needs `posts/*.md` | synthetic goldens |
| `deepeval` folder shadowed the library | **fixed** — moved to `docs/deepeval_intro.md` | all imports |
| `deepeval` not installed | **fixed** — 4.3.4 in `[dependency-groups].dev` | all imports |

Only one step is fully unblocked: **save the corpus and write the goldens.** That is also episode-4 material, and it is the natural place to start.

---

## Appendix — Corrections to the reference doc

`deepeval_intro.md` is accurate on DeepEval itself. These four points disagree with the current code:

| Reference doc says | Reality |
|---|---|
| `MAX_SEARCHES=3` (§8.4) | **4** — `nodes.py:104` |
| `MAX_REPAIRS=1` (§8.4) | **2** — `edges.py:26` |
| `state['documents']`, `d.page_content` (§8.1) | state key is `chunks`, a list of `{"blog_post","text"}` dicts — `state.py:14` |
| "pass the final graded context, not every chunk" (§8.1) | the graph passes **every** chunk — `nodes.py:135` accumulates, `nodes.py:217` formats all |

It is also right about the index size: **114 chunks**, verified against Pinecone.
