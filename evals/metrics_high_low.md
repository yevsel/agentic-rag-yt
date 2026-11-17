# The Five RAG Metrics — What High and Low Mean

Each metric gives a score from 0 to 1. Here is what that number is telling you.

---

## Faithfulness
**Asks:** did the answer stick to the retrieved chunks?

- **High:** every claim in the answer came from the chunks.
- **Low:** it made something up.

**Blames:** the writer (`write_answer_node`).

---

## AnswerRelevancy
**Asks:** did the answer actually answer the question?

- **High:** on-topic, no waffle.
- **Low:** rambling, padded, or answering something else.

**Blames:** the writer (`write_answer_node`).

---

## ContextualRelevancy
**Asks:** was the retrieved context full of junk?

- **High:** nearly every chunk was worth reading.
- **Low:** lots of irrelevant chunks mixed in.

**Blames:** the retriever (`tools_node`).

---

## ContextualPrecision
**Asks:** are the best chunks ranked first?

- **High:** useful chunks at the top, junk at the bottom.
- **Low:** useful chunks buried below junk.

**Blames:** the retriever (`tools_node`). Reranking fixes this one.

---

## ContextualRecall
**Asks:** did retrieval find everything the good answer needed?

- **High:** the chunks contain everything the ideal answer needs.
- **Low:** something required was never retrieved.

**Blames:** the retriever (`tools_node`). Causes: `k` too small, bad chunking, weak embeddings.

---

## Shortcut

```
Faithfulness, AnswerRelevancy          -> the writer
ContextualRelevancy, Precision, Recall -> the retriever
```

A low score tells you *what* kind of failure it is and *where* to look.
It does not tell you the fix — read the `reason` string for that.
