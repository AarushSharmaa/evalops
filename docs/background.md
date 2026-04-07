# background

## what this is

ragcheck is a lightweight Python library for evaluating RAG (Retrieval-Augmented Generation) systems. It answers one question: given a question, an answer, and the documents your system retrieved — how good is this answer?

It returns three scores: faithfulness, answer relevance, and context precision. It needs no ground truth. It works with any LLM the developer already has.

---

## where this came from

Built at Incedo, a GenAI QA system for a US wealth management firm. The system used RAG to generate test cases from user stories. We had no good way to evaluate whether the RAG pipeline was actually working — whether the answers were faithful to the retrieved documents, or whether we were retrieving the right context in the first place.

RAGAS existed but required committing to their framework and was OpenAI-heavy. BLEU/ROUGE required reference answers we didn't have in production. So we eyeballed outputs.

ragcheck is the library that should have existed then.

---

## the problem with existing tools

| tool | problem |
|------|---------|
| RAGAS | heavy framework, opinionated, OpenAI-first, breaking changes constantly |
| BLEU / ROUGE | requires ground truth reference answers — useless in prod |
| DeepEval | another framework to adopt, paid tiers for real use |
| manual review | doesn't scale past a handful of queries |

The gap: a reference-free RAG eval tool that installs in one line, works with any LLM, and has no framework lock-in.

---

## design philosophy

**llm_fn pattern** — the developer passes any callable that takes a string and returns a string. The library handles prompting and parsing. This makes ragcheck work with every LLM ever built, without a single line of integration code from us.

```python
# works with anything
def my_llm(prompt: str) -> str:
    return your_llm_client.generate(prompt)

result = evaluate(question, answer, contexts, llm_fn=my_llm)
```

**reference-free** — no ground truth answers needed. The metrics are computed purely from the question, answer, and retrieved contexts. This is what makes it useful in production.

**zero mandatory dependencies** — the library itself only uses the Python standard library. The user's LLM client is their own.

**one function** — `evaluate()` is the entire public API for v1. Keep it that way.

---

## the three metrics

**Faithfulness** — Is every claim in the answer supported by the retrieved documents? Catches hallucination. Score: 0–1.

**Answer relevance** — Does the answer actually address the question? Catches off-topic responses. Score: 0–1.

**Context precision** — Was the retrieved context useful, or mostly noise? Evaluates retrieval quality. Score: 0–1.

Each metric is computed via an LLM prompt. The prompt is structured, asks for a score with reasoning, and parses the numeric result.

---

## how it works under the hood

Each metric sends a carefully structured prompt to `llm_fn`. For example, faithfulness:

> "Here is an answer. Here are the source documents it was supposed to be based on. Rate from 0 to 1 how well the answer is supported by the documents. Return only a JSON with keys: score (float), reasoning (string)."

The library formats the prompt, calls `llm_fn`, parses the JSON response, and returns a `EvalResult` dataclass.

---

## who uses this

**Prototype phase**: developer building a RAG system who wants to know if it's working before going to prod. Drops ragcheck in, runs it on 20 sample queries, spots the failure modes.

**Production phase**: engineering team running ragcheck on a sample of live queries daily. Alerts if faithfulness drops below a threshold. Catches regressions after retrieval or prompt changes.

---

## positioning

RAGAS is a gym membership. ragcheck is a pushup.

You don't adopt ragcheck. You call it.

---

## what this signals (portfolio context)

This project demonstrates:
- Deep understanding of RAG failure modes (not just how to build RAG, but how to know when it's failing)
- Production thinking: reference-free because prod doesn't have ground truth
- API design judgment: llm_fn pattern instead of locking to one provider
- PyPI packaging and open source distribution

Origin story ties directly to the Incedo work on the wealth management QA system.