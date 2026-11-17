# DeepEval Deep Dive: Evaluating Your Agentic RAG

*Written against the DeepEval 4.3.x docs (November 2026). Examples assume your LangGraph + Pinecone graph: 11 nodes, three gates (Adaptive, Corrective, Self-RAG).*

## 1. The mental model

DeepEval is pytest for LLM apps. Five objects cover almost everything:

| Object | What it is |
| --- | --- |
| Golden | A pending test case: an input plus expected results. No app output yet. |
| LLMTestCase | A golden plus what your app actually produced (actual\_output, retrieval\_context). |
| EvaluationDataset | A list of goldens. Single-turn or multi-turn, never both. |
| Metric | A ruler. Scores 0 to 1, returns a reason, passes when score ≥ threshold (default 0.5). |
| Test run | One execution of metrics over test cases. Saved locally; optionally pushed to Confident AI. |

Three levels you can evaluate at:

- **End-to-end:** treat the app as a black box (input → output). Simplest. Start here.
- **Component-level:** attach metrics to individual spans (retriever, generator) with @observe. This is how you find *where* quality drops.
- **Trajectory:** score the full ordered trace of an agent (for example TaskCompletion, StepEfficiency).

Your graph is a multi-step agent, so one end-to-end score can’t tell you whether a bad answer came from retrieval, generation or a gate. Start end-to-end, then add component-level once you begin tuning.

## 2. Setup

```bash
pip install -U deepeval
export OPENAI_API_KEY=...   # default judge
```

- Python 3.9+. DeepEval auto-loads `.env.local`, then `.env`, from the current directory at import time (real environment variables win). Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.
- `deepeval login` is optional (cloud reports on Confident AI). Everything works locally without it.
- `pip install -U deepeval[inspect]` adds `deepeval inspect`, a terminal UI for browsing test runs as trace trees. The docs recommend it for dev environments only.
- **Version gotcha:** current docs import `SingleTurnParams` from `deepeval.test_case`. Many older tutorials use `LLMTestCaseParams`. If an import fails, check your version first.
- **Judge model:** metrics call an LLM judge, OpenAI by default. Override per metric with `model=`. Azure, Ollama, Gemini, LiteLLM and any custom `DeepEvalBaseLLM` subclass are supported.

## 3. Test case anatomy for RAG

```python
from deepeval.test_case import LLMTestCase

tc = LLMTestCase(
    input='What does the post say about X?',
    actual_output=answer,                # what YOUR graph returned
    expected_output=gold_answer,         # ideal answer, from the golden
    retrieval_context=retrieved_chunks,  # list[str], chunks retrieved on THIS run
)
```

- `retrieval_context` is what retrieval and faithfulness metrics judge. It must be the live output of your retriever (post-rerank, exactly what the generator saw), never the golden’s context.
- `context` (on goldens) is ground-truth source text used for dataset generation. Different thing.
- `expected_output` is required only by reference-based metrics (Contextual Precision, Contextual Recall, G-Eval correctness).

## 4. Running metrics three ways

**4.1 Standalone** (best for debugging one case):

```python
from deepeval.metrics import FaithfulnessMetric

m = FaithfulnessMetric(threshold=0.7, verbose_mode=True)
m.measure(tc)
print(m.score, m.reason, m.is_successful())
```

**4.2 evaluate()** (scripts and notebooks, no instrumentation needed):

```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

evaluate(test_cases=[tc1, tc2], metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()])
```

**4.3 pytest + CLI** (CI and regression gates):

```python
# test_rag.py
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path='evals/goldens.json')

@pytest.mark.parametrize('golden', dataset.goldens)
def test_rag(golden):
    answer, chunks = run_graph(golden.input)   # your function, see section 8
    tc = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        retrieval_context=chunks,
    )
    assert_test(test_case=tc, metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()])
```

```bash
deepeval test run test_rag.py
```

`evaluate()` and `assert_test()` are the same engine. Use `evaluate()` to explore, `assert_test()` to gate. Both give you caching, parallelism, cost tracking and error handling that raw `measure()` loops don’t.

## 5. The five RAG metrics

| Metric | Judges | Needs expected\_output | Question it answers |
| --- | --- | --- | --- |
| AnswerRelevancy | Generator | No | Is the answer on-topic for the question? |
| Faithfulness | Generator | No | Is every claim in the answer supported by retrieval\_context? |
| ContextualRelevancy | Retriever | No | How much of the retrieved context is relevant to the question? |
| ContextualPrecision | Retriever | Yes | Are relevant chunks ranked above irrelevant ones? |
| ContextualRecall | Retriever | Yes | Did retrieval contain everything the ideal answer needs? |

**How each is computed** (all are LLM-as-a-judge and all return a reason):

- **Faithfulness:** extract the factual claims in actual\_output, extract the truths in retrieval\_context, compare. Score is roughly the share of claims the context supports.
- **AnswerRelevancy:** split the answer into statements, judge each as relevant or not to the input.
- **ContextualRelevancy:** judge the statements in retrieval\_context for relevance to the input. Measures noise.
- **ContextualPrecision:** for each chunk, judge whether it helps produce expected\_output, then compute a weighted cumulative precision that rewards relevant chunks appearing early. Order matters. This is the metric that reacts to reranking.
- **ContextualRecall:** break expected\_output into statements and check how many can be attributed to the retrieval context. Measures coverage.

**Reading the scores:**

- Faithfulness low, relevancy fine → the generator is inventing claims (prompt or model problem). The docs are explicit that this is a generation signal, not a retrieval one.
- AnswerRelevancy low → padded or off-topic answers.
- Recall low → the retriever misses needed info (k too small, chunking, embeddings, no keyword signal).
- Precision low → relevant chunks are buried. Reranking fixes this.
- ContextualRelevancy low → too much noise in context. Lower k, improve chunking, add filters.
- All metrics high but answers still wrong → suspect your expected\_output or the judge. Turn on `verbose_mode` and read the reasons.

**Choosing metrics:** the docs advise no more than five, split roughly 2–3 generic system metrics and 1–2 custom ones. For your project: Faithfulness, AnswerRelevancy, ContextualPrecision, ContextualRecall, plus one custom abstention metric (section 7).

**Thresholds:** the default is 0.5 for every metric. Treat it as a starting point. Record a baseline first, then set each threshold slightly below the baseline so it catches regressions without flapping.

**Reference vs referenceless:** Precision and Recall need ground truth, so they are development-time metrics. Faithfulness, AnswerRelevancy and ContextualRelevancy need no labels and can also run on production traces.

## 6. Datasets and goldens

A golden carries the input and expected results; your app fills in the rest at eval time. Key fields on `Golden`: `input`, `expected_output`, `context`, `expectations`, plus free-form `additional_metadata` and `comments`. Leave `actual_output` and `retrieval_context` empty, since they are produced per run.

**JSON file** (the format your loader reads):

```json
[
  {
    "input": "<question answerable from your two posts>",
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

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path='evals/goldens.json')
# also: add_goldens_from_jsonl_file, add_goldens_from_csv_file
dataset.save_as(file_type='json', directory='./evals', file_name='goldens_v2')
```

If your keys differ (for example `question` instead of `input`), the loaders accept custom key names such as `input_key_name`.

**What a good set looks like for your graph** (roughly 30 goldens to start):

- Answerable from a single chunk (about 15).
- Answerable only by combining chunks or comparing (about 5). These stress recall and the Corrective gate.
- Unanswerable (about 10), including near-misses: on-topic questions the posts don’t actually support. These test your Self-RAG give-up path, which is your project’s whole hook.

**Synthetic goldens with the Synthesizer:**

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['posts/post1.md', 'posts/post2.md'],
    include_expected_output=True,
)
synthesizer.save_as(file_type='json', directory='./synthetic_data', file_name='weng_goldens')
```

- Pipeline: generate inputs → filter by quality (threshold 0.5, up to 3 retries) → evolve to be harder → style.
- It never generates `actual_output`. That’s your app’s job.
- Control complexity with `EvolutionConfig`. Seven evolutions exist, but only four stay answerable from the context: MULTICONTEXT, CONCRETIZING, CONSTRAINED and COMPARATIVE. For RAG, restrict to those so you don’t get questions the corpus can’t answer by accident.
- `generate_goldens_from_contexts()` gives you control over which chunks seed the questions, handy when you want goldens tied to your own 114 chunks.
- The docs themselves say to manually inspect and edit synthetic data. Do it. Also write the unanswerable questions by hand, because generation is grounded in your documents.

```python
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import EvolutionConfig

cfg = EvolutionConfig(
    evolutions={Evolution.MULTICONTEXT: 0.4, Evolution.CONCRETIZING: 0.2,
                Evolution.CONSTRAINED: 0.2, Evolution.COMPARATIVE: 0.2},
    num_evolutions=2,
)
synthesizer = Synthesizer(evolution_config=cfg)
```

## 7. Custom metrics

### 7.1 G-Eval

G-Eval is the general-purpose LLM judge. You describe the criteria in plain language.

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

correctness = GEval(
    name='Correctness',
    evaluation_steps=[
        'Check whether facts in actual output contradict facts in expected output',
        'Heavily penalize omission of detail',
        'Vague language or differing opinions are fine',
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
)
```

Rules that matter:

- Give `criteria` **or** `evaluation_steps`, never both. With only `criteria`, G-Eval regenerates its steps on every run, which causes score drift. Workflow: write a short criteria, inspect the generated steps, then paste them in as `evaluation_steps`.
- Include in `evaluation_params` only the fields your steps actually mention. Extra fields feed the judge irrelevant context and degrade scores.
- `rubric=[Rubric(score_range=(0,2), expected_outcome='...'), ...]` pins the judge to score bands. Ranges run 0–10 inclusive and must not overlap. This stops scores clustering mid-range.
- `strict_mode=True` forces a binary 1 or 0 (and sets the threshold to 1).
- With OpenAI models, G-Eval normalizes the score using output token probabilities. A custom judge model needs `a_generate_raw_response` returning logprobs, otherwise it falls back to the raw integer.
- G-Eval is not deterministic. If you need fixed decision logic, use DAG.

### 7.2 A correct-abstention metric for your project

```python
abstention = GEval(
    name='Correct Abstention',
    evaluation_steps=[
        'If expected output is DECLINE, pass only if actual output clearly says the sources cannot answer and gives no substantive answer',
        'If expected output is a reference answer, penalize heavily if actual output declines to answer',
        'If actual output answers when it should decline, score very low',
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.8,
)
```

Run it on **all** goldens. It catches both failure directions: answering when it should decline, and declining when it shouldn’t.

### 7.3 DAG, for deterministic judging

DAGMetric builds a decision tree of narrow LLM questions whose branches map to fixed scores. Use it when the rule is objective (for example: did the answer include a citation, and only then judge tone). Start with G-Eval, move to DAG when you need control.

### 7.4 Expectations: must and must-not rules on a golden

```python
from deepeval.dataset import Golden, Expectations
from deepeval.test_case import LLMTestCase

golden = Golden(
    input='Who won the 2022 World Cup?',
    expectations=Expectations(
        must=['State that the sources do not contain the answer.'],
        must_not=['Give a substantive answer from outside knowledge.'],
    ),
)
tc = LLMTestCase(input=golden.input, actual_output=answer, expectations=golden.expectations)
```

Per the docs, `evaluate()` or `assert_test()` can check a test case’s expectations without an explicit metric, and `evals_iterator()` checks them automatically. If your version complains, add a metric alongside. Expectations are plain-language and survive JSON, JSONL and CSV save/load, so they’re a lighter alternative to the abstention metric above.

## 8. Wiring DeepEval into your LangGraph

### 8.1 Black box first

```python
def run_graph(question: str):
    state = graph.invoke({'question': question})
    # adapt these keys to your state schema
    answer = state['answer']
    chunks = [d.page_content for d in state['documents']]
    return answer, chunks
```

- With up to 3 searches and Corrective refinement, pass the **final graded context the generator used**, not every chunk ever retrieved. Otherwise Precision, Recall and Relevancy get scored on noise your gates already filtered out.
- On give-up answers the context may be empty or irrelevant. Run generator and retriever metrics only on answerable goldens (the suite in section 9 does this).

### 8.2 Component-level with @observe

A single end-to-end score hides where quality drops. Trace the pieces and attach metrics to them:

```python
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.tracing import observe, update_current_span

@observe(metrics=[ContextualRelevancyMetric()])
def retrieve(query: str):
    chunks = pinecone_search(query)          # your retrieval (+ rerank)
    update_current_span(input=query, retrieval_context=chunks)
    return chunks

@observe(metrics=[AnswerRelevancyMetric()])
def generate(query: str, chunks):
    answer = llm_answer(query, chunks)
    update_current_span(input=query, output=answer)
    return answer

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path='evals/goldens.json')

for golden in dataset.evals_iterator():
    graph.invoke({'question': golden.input})
```

- Your graph nodes are plain functions, so you can put `@observe` on the retrieve and generate nodes directly.
- Every call is a span, so a Corrective retry produces its own relevancy score. You can see whether refinement actually improved the context.
- For end-to-end metrics from the same trace, decorate the top-level function and call `update_current_trace(input=..., output=..., retrieval_context=...)`, then pass `metrics=[...]` to `evals_iterator()`. Trace-level and span-level scores land in one test run.
- In CI, swap the loop for `assert_test(golden=golden)` inside a pytest test; the metrics stay on the spans.
- The docs use `AsyncConfig(run_async=False)` (from `deepeval.evaluate`) for synchronous apps. For async graphs, wrap each call in `asyncio.create_task(...)` and call `dataset.evaluate(task)`.

### 8.3 Trajectory metrics via the LangGraph callback

No code changes inside nodes. Pass DeepEval’s callback handler at invoke time:

```python
from deepeval.evaluate import AsyncConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

for golden in dataset.evals_iterator(
    metrics=[TaskCompletionMetric()],
    async_config=AsyncConfig(run_async=False),
):
    graph.invoke(
        {'question': golden.input},
        config={'callbacks': [CallbackHandler()]},
    )
```

Every graph run, node, model call and tool call becomes a span. Trajectory metrics are TaskCompletion, StepEfficiency, PlanAdherence and PlanQuality. They were designed for tool-using agents. Your graph is a fixed workflow with bounded loops, so treat them as secondary. StepEfficiency is still useful as a cost signal (does reranking reduce wasted steps?). For RAG metrics on spans, use the @observe pattern in 8.2.

### 8.4 Testing each gate

| Gate | What to test | How |
| --- | --- | --- |
| Adaptive (classify) | Does it route each question correctly? | Store the expected route in the golden’s `additional_metadata` and assert equality in pytest. No LLM judge needed. |
| Corrective (grade/refine) | Does refinement improve context, within MAX\_CORRECTIVE\_ATTEMPTS=2? | ContextualRelevancy on each retrieve span, compare attempt 1 vs attempt 2. Assert attempt counts from state. |
| Self-RAG (check/fix/give\_up) | Does it catch unsupported answers and decline correctly? | Faithfulness on final answers as an independent audit, plus the abstention metric on unanswerable goldens. |
| Limits | MAX\_SEARCHES=3, MAX\_REPAIRS=1 never exceeded | Plain asserts on state counters. |

**Meta-eval of your gate.** Your Self-RAG gate is itself an LLM judge. DeepEval’s Faithfulness is a second, independent judge. Compare them: if the gate passed an answer that Faithfulness scores low, that’s a gate false negative; if the gate rejected an answer Faithfulness scores high, that’s a false positive. This is a strong, original section for your tutorial.

## 9. Experiments: reranking, hybrid search, semantic cache

The workflow is: freeze the golden set, label each run (baseline, +rerank, +hybrid, +cache), change one thing per run, compare.

```python
import json, os, statistics, time
from deepeval.metrics import (FaithfulnessMetric, AnswerRelevancyMetric,
                              ContextualPrecisionMetric, ContextualRecallMetric)
from deepeval.test_case import LLMTestCase

NAMES = ['abstention', 'faithfulness', 'answer_relevancy', 'ctx_precision', 'ctx_recall']

def run_suite(label, goldens, pipeline):
    rows, t0 = [], time.time()
    for g in goldens:
        answer, chunks = pipeline(g.input)
        tc = LLMTestCase(input=g.input, actual_output=answer,
                         expected_output=g.expected_output, retrieval_context=chunks)
        metrics = {'abstention': abstention}          # from section 7.2
        if g.expected_output != 'DECLINE':             # retrieval/generator metrics: answerable only
            metrics.update({
                'faithfulness': FaithfulnessMetric(),
                'answer_relevancy': AnswerRelevancyMetric(),
                'ctx_precision': ContextualPrecisionMetric(),
                'ctx_recall': ContextualRecallMetric(),
            })
        row = {'input': g.input}
        for name, m in metrics.items():
            m.measure(tc)
            row[name] = {'score': m.score, 'reason': m.reason}
        rows.append(row)
    summary = {n: round(statistics.mean(r[n]['score'] for r in rows if n in r), 3) for n in NAMES}
    summary['seconds'] = round(time.time() - t0)
    os.makedirs('evals/runs', exist_ok=True)
    with open(f'evals/runs/{label}.json', 'w') as f:
        json.dump({'label': label, 'summary': summary, 'rows': rows}, f, indent=2)
    return summary
```

Also track what DeepEval doesn’t: average LLM calls per query and latency (count them in your graph state or wrap your LLM client). Cost per question is half the story for reranking and caching.

**What to expect (hypotheses to test, not guarantees):**

- **Reranking** (retrieve 20–30, keep 4–5): ContextualPrecision should rise because relevant chunks move up. If you shrink the final k, Recall can dip, so watch both. Fewer Corrective retries is the payoff you want to see in the call counts.
- **Hybrid search:** targets Recall, since keyword matching surfaces chunks dense embeddings miss.
- **Semantic cache:** it bypasses your gates on a hit, so judge it differently (below).

**Evaluating a semantic cache.** LLM-judge metrics are the wrong tool for the cache itself. Use plain pytest assertions on three pair sets:

- Paraphrase pairs must hit.
- Near-opposite pairs (enable vs disable, before vs after) must miss.
- After the index version changes, previously cached entries must miss.

Then run Faithfulness on cache-hit answers against the retrieval\_context stored with the cached entry, to confirm stale or wrong answers aren’t served.

**Judge noise.** LLM judges vary run to run. Run the baseline twice first to learn your noise floor. With around 30 goldens, small differences in a mean score are probably noise, so also look at per-question deltas and the reasons for the questions that changed.

## 10. CI/CD

```yaml
name: rag-evals
on: [pull_request]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12'}
      - run: pip install -r requirements.txt
      - run: deepeval test run tests/test_rag.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
```

- `deepeval test run file.py -i -c`: `-i` ignores errors (useful when a judge returns bad JSON) and `-c` uses the local cache, so a partially failed run doesn’t redo finished cases. See the Flags and Configs docs page for the full flag list, including parallelism.
- Each `assert_test()` / `evaluate()` call needs at least one non-flaky metric with a threshold.
- `flaky=True` on a metric keeps scoring and reporting it, but its failure never fails the test case. Good for a noisy metric you want to watch.
- `threshold=None` is score-only mode: recorded, never gating.
- Keep the PR suite small (about 10 goldens covering each gate) and run the full set nightly.

## 11. Judge reliability, cost and speed

- Each LLM metric makes several judge calls (claim extraction, verdicts, reason). Five metrics over 30 goldens is hundreds of calls. Use a cheaper judge via `model=` while iterating and the stronger one for numbers you report. Never compare runs scored by different judges.
- Prefer a judge from a different model family than your generator, to reduce self-preference bias (a practical rule of thumb, not a DeepEval requirement).
- Metrics run async internally by default (`async_mode=True`). To run several without blocking, use `await asyncio.gather(m1.a_measure(tc), m2.a_measure(tc))`.
- `verbose_mode=True` prints a metric’s intermediate steps. Use it whenever a score surprises you.
- Custom judge: subclass `DeepEvalBaseLLM` and implement `load_model`, `generate`, `a_generate` and `get_model_name`. The docs warn results can degrade because weaker models struggle with valid JSON; pass a custom `evaluation_template` to tighten prompts.
- There is also an optional eval-mode setting (`deepeval set-eval-mode llm|hybrid|system_one`) that changes who does the judging; `llm` is the default.

## 12. Pitfalls checklist

- Using the golden’s `context` as `retrieval_context`. Use what the retriever actually returned.
- Forgetting `expected_output` on Precision and Recall cases.
- Running Faithfulness and the retrieval metrics on unanswerable goldens. Score abstention there instead.
- Putting params in G-Eval `evaluation_params` that your steps never mention.
- Giving G-Eval only `criteria` and wondering why scores drift. Lock in `evaluation_steps`.
- Trusting a synthetic dataset without reading it.
- Changing the golden set between runs you plan to compare.
- Mixing single-turn and multi-turn goldens in one dataset. DeepEval raises an error.
- Using the default 0.5 threshold everywhere without checking it against a baseline.
- Claiming an improvement smaller than your measured run-to-run noise.

## 13. Suggested build order for your tutorial

1. Write 30 goldens (15 single-chunk, 5 multi-chunk, 10 unanswerable). Save as JSON.
2. Add `run_graph()` and run the black-box suite. Record the baseline.
3. Add the abstention metric and show your three gates earning their keep.
4. Add @observe spans and show per-component scores, including Corrective retries.
5. Add reranking. Re-run. Show the precision and call-count deltas.
6. Add hybrid search. Re-run.
7. Add the semantic cache with its own pair tests, and the Self-RAG meta-eval.
8. Wire `deepeval test run` into CI with a small PR subset.

## 14. Sources

- DeepEval docs: https://deepeval.com/docs/getting-started-rag
- Metrics introduction (custom judges, thresholds, flaky metrics): https://deepeval.com/docs/metrics-introduction
- G-Eval: https://deepeval.com/docs/metrics-llm-evals
- Datasets and goldens: https://deepeval.com/docs/evaluation-datasets
- Golden Synthesizer: https://deepeval.com/docs/golden-synthesizer
- LangGraph integration: https://deepeval.com/integrations/frameworks/langgraph
- Component-level evals: https://www.deepeval.com/docs/evaluation-component-level-llm-evals
- Faithfulness: https://www.deepeval.com/docs/metrics-faithfulness
- Package and release history: https://pypi.org/project/deepeval/
