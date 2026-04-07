# roadmap

## current phase: phase 1

---

## phase 1 — core library ✦ current

**goal**: working `evaluate()` function, locally installable, tested.

**deliverables**:
- `ragcheck/core.py` — the entire library
  - `evaluate(question, answer, contexts, llm_fn)` → `EvalResult`
  - `EvalResult` dataclass with `.faithfulness`, `.answer_relevance`, `.context_precision`
  - Three internal prompt functions, one per metric
  - JSON response parsing with error handling
- `ragcheck/__init__.py` — exports `evaluate` and `EvalResult`
- `pyproject.toml` — package metadata, hatchling build backend, no mandatory deps
- `tests/test_core.py` — unit tests with a mock `llm_fn`, no real LLM needed
- `examples/quickstart.py` — working example with Gemini (uses `GEMINI_API_KEY` from `.env`)
- `README.md` — what, why, 10-line quickstart, four LLM examples, comparison to RAGAS

**done when**: `pip install -e .` works, `evaluate()` returns correct scores on mock inputs, all tests pass.

---

## phase 2 — pypi publish

**goal**: `pip install ragcheck` works for anyone in the world.

**deliverables**:
- Clean package name claimed on PyPI (`ragcheck`)
- Version `0.1.0` published
- README renders correctly on PyPI
- GitHub repo public with proper license (MIT)

**done when**: a fresh `pip install ragcheck` in a clean virtualenv works.

---

## phase 3 — batch evaluation

**goal**: evaluate an entire dataset, not just one query at a time.

**deliverables**:
- `evaluate_batch(samples: list[dict], llm_fn) → BatchResult`
- `BatchResult` with per-sample scores + aggregate stats (mean, min, max per metric)
- CSV export: `result.to_csv("eval_results.csv")`
- Updated README with batch example

**rationale**: single-query eval is useful for debugging. Batch eval is what you'd run in CI or overnight on a test set.

---

## phase 4 — cli

**goal**: run ragcheck without writing any Python.

**deliverables**:
- `ragcheck eval --input queries.json --output results.csv --model gemini`
- Built-in LLM options: `--model gemini`, `--model openai`, `--model ollama`
- Optional deps: `ragcheck[gemini]`, `ragcheck[openai]` in pyproject extras

**rationale**: makes it accessible to ML engineers who aren't writing Python scripts for every eval run.

---

## phase 5 — streamlit demo (portfolio)

**goal**: a live demo that hiring managers can click on and immediately understand.

**deliverables**:
- Streamlit app: paste a question, answer, and context chunks → see scores with explanations
- Deployed on Streamlit Cloud
- Link in README and GitHub profile

**rationale**: PyPI is a signal. A live demo makes it real and clickable.

---

## not building (explicitly out of scope)

- Custom embedding models or retrieval — this is eval only, not a RAG framework
- A dashboard or persistent storage — that's a different product
- Async support — add only if there's a real request for it
- Support for non-text modalities — out of scope for v1