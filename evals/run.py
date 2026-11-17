"""Run the agentic RAG graph, then turn the result into a DeepEval test case.

This file is the bridge between your app and DeepEval. It has two jobs:

    1. run_graph(question)        -> ask the graph, get (answer, chunks)
    2. make_test_case(...)        -> put one run into the shape DeepEval scores

Nothing else. Read these two functions, and you understand the whole eval loop.
"""

from deepeval.test_case import LLMTestCase
from graph import graph


def run_graph(question):
    """Ask the graph one question. Return (answer, chunks).

    answer = the text the graph finally produced
    chunks = the pieces of the blog posts the answer was written from
    """

    # graph.invoke() runs the whole graph and hands back the final state.
    # The state is a dict with keys like "answer", "chunks", "intent", ...
    state = graph.invoke(
        {"question": question},
        {"recursion_limit": 40},
    )

    # The final answer. .get() returns "" instead of crashing if it is missing.
    answer = state.get("answer", "")

    # The chunks the writer was shown.
    # state["chunks"] looks like:
    #   [{"blog_post": "diffusion-models", "text": "Sora uses DiT..."}, ...]
    # DeepEval wants a plain list of strings, so pull out just the "text".
    chunks = []
    for chunk in state.get("chunks", []):
        chunks.append(chunk["text"])

    return answer, chunks


def make_test_case(question, answer, chunks, expected_output=None):
    """Package one graph run into the object DeepEval can score.

    input             = the question the user asked
    actual_output     = what your graph answered
    retrieval_context = the chunks the answer was based on
    expected_output   = the ideal answer (only some metrics need this)
    """

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=chunks,
        expected_output=expected_output,
    )

    return test_case


# This block only runs if you execute the file directly:
#   python -m evals.run
# It is a quick way to see what the graph returns, before scoring anything.
if __name__ == "__main__":
    answer, chunks = run_graph("What are the three components of ACE?")

    print("ANSWER:")
    print(answer)
    print()
    print("NUMBER OF CHUNKS:", len(chunks))
    print("FIRST CHUNK:")
    print(chunks[0] if chunks else "(none)")
