"""
External developer simulation tests.

These tests are written from the perspective of a developer who just ran
"pip install ragcheck" and is trying it out for the first time. They follow
the README exactly and test the things a real user would actually try.

Run with a real LLM:
    GROQ_API_KEY=... pytest tests/test_external_developer.py -v

Run with mock LLM (no API key needed):
    pytest tests/test_external_developer.py -v -k "not groq"
"""

import json
import time
import os

import pytest

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

requires_groq = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY not set",
)


def make_groq_llm_fn():
    import urllib.request

    def llm_fn(prompt: str) -> str:
        payload = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "ragcheck-external-test/0.2.0",
            },
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        time.sleep(2)
        return body["choices"][0]["message"]["content"]

    return llm_fn


@pytest.fixture(autouse=True)
def rate_limit_pause():
    yield
    time.sleep(2)


# ---------------------------------------------------------------------------
# Scenario 1: Developer copies the README quickstart verbatim
# ---------------------------------------------------------------------------

@requires_groq
def test_readme_quickstart_works_exactly_as_shown():
    """The exact code block from the README should run without modification."""
    import ragcheck

    llm_fn = make_groq_llm_fn()

    result = ragcheck.evaluate(
        question="What causes the northern lights?",
        answer="Charged particles from the sun collide with gases in Earth's atmosphere.",
        contexts=["Aurora borealis occurs when solar particles interact with the upper atmosphere."],
        llm_fn=llm_fn,
    )

    # These are all the attributes the README shows
    assert isinstance(result.faithfulness, float)
    assert isinstance(result.answer_relevance, float)
    assert isinstance(result.context_precision, float)
    assert isinstance(result.reasoning, dict)
    assert isinstance(result.parse_errors, list)
    assert isinstance(result.passed(), bool)
    assert isinstance(result.passed(threshold=0.7), bool)
    assert isinstance(result.to_json(indent=2), str)
    assert isinstance(str(result), str)


# ---------------------------------------------------------------------------
# Scenario 2: Developer uses their own OpenAI-style lambda
# ---------------------------------------------------------------------------

def test_lambda_llm_fn_works():
    """Developer passes a lambda instead of a named function -- README shows this pattern."""
    import ragcheck

    result = ragcheck.evaluate(
        question="What is the capital of France?",
        answer="Paris.",
        contexts=["Paris is the capital of France."],
        llm_fn=lambda prompt: json.dumps({"score": 0.9, "reasoning": "correct"}),
    )

    assert result.faithfulness == 0.9


# ---------------------------------------------------------------------------
# Scenario 3: Developer uses context_recall after reading the optional section
# ---------------------------------------------------------------------------

@requires_groq
def test_context_recall_section_of_readme():
    """Developer enables context_recall exactly as shown in the README."""
    import ragcheck

    llm_fn = make_groq_llm_fn()

    result = ragcheck.evaluate(
        question="What is the boiling point of water?",
        answer="Water boils at 100 degrees Celsius.",
        contexts=["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
        llm_fn=llm_fn,
        include_context_recall=True,
    )

    assert result.context_recall is not None
    assert 0.0 <= result.context_recall <= 1.0
    assert "context_recall" in result.reasoning
    # context_recall should appear in str() output
    assert "context_recall" in str(result)
    # context_recall should appear in to_dict()
    assert "context_recall" in result.to_dict()


# ---------------------------------------------------------------------------
# Scenario 4: Developer writes a custom metric from the README example
# ---------------------------------------------------------------------------

@requires_groq
def test_custom_metric_section_of_readme():
    """Developer defines a conciseness metric exactly as shown in the README."""
    import ragcheck

    llm_fn = make_groq_llm_fn()

    def conciseness_prompt(question: str, answer: str, contexts: list) -> str:
        return (
            f"Rate how concise this answer is. 1.0 = very concise, 0.0 = very verbose.\n\n"
            f"Answer: {answer}\n\n"
            f'Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}'
        )

    result = ragcheck.evaluate(
        question="What is the capital of France?",
        answer="Paris.",
        contexts=["Paris is the capital of France."],
        llm_fn=llm_fn,
        extra_metrics={"conciseness": conciseness_prompt},
    )

    assert "conciseness" in result.extra_metrics
    assert 0.0 <= result.extra_metrics["conciseness"] <= 1.0
    assert "conciseness" in result.reasoning
    assert result.parse_errors == []


# ---------------------------------------------------------------------------
# Scenario 5: Developer uses evaluate_batch for a test suite
# ---------------------------------------------------------------------------

@requires_groq
def test_batch_eval_filter_pattern_from_readme():
    """Developer runs a batch and filters passing results -- the README pattern."""
    import ragcheck

    llm_fn = make_groq_llm_fn()

    items = [
        {
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Tokyo.",
            "contexts": ["Tokyo is the capital city of Japan."],
        },
        {
            "question": "Who painted the Mona Lisa?",
            "answer": "The Mona Lisa was painted by Pablo Picasso.",
            "contexts": ["The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."],
        },
    ]

    results = ragcheck.evaluate_batch(items, llm_fn)
    passing = [r for r in results if r.passed(threshold=0.7)]

    assert len(results) == 2
    # First item is grounded and correct -- should pass
    assert results[0].passed(threshold=0.7) is True
    # Second item is a hallucination -- faithfulness should be low
    assert results[1].faithfulness <= 0.4
    assert len(passing) == 1


# ---------------------------------------------------------------------------
# Scenario 6: Developer exports results to JSON for logging
# ---------------------------------------------------------------------------

@requires_groq
def test_export_to_json_and_reload():
    """Developer serialises a result and reads it back -- common logging pattern."""
    import ragcheck

    llm_fn = make_groq_llm_fn()

    result = ragcheck.evaluate(
        question="What is photosynthesis?",
        answer="Photosynthesis converts sunlight into food for plants.",
        contexts=["Plants use sunlight, water, and CO2 to produce glucose through photosynthesis."],
        llm_fn=llm_fn,
    )

    raw = result.to_json()
    parsed = json.loads(raw)

    # All fields round-trip correctly
    assert parsed["faithfulness"] == result.faithfulness
    assert parsed["answer_relevance"] == result.answer_relevance
    assert parsed["context_precision"] == result.context_precision
    assert parsed["reasoning"] == result.reasoning
    assert parsed["parse_errors"] == result.parse_errors


# ---------------------------------------------------------------------------
# Scenario 7: Developer makes common mistakes -- library should give clear errors
# ---------------------------------------------------------------------------

def test_mistake_passes_string_instead_of_list_for_contexts():
    """Developer accidentally passes a string for contexts instead of a list."""
    import ragcheck

    with pytest.raises(TypeError, match="contexts must be a list"):
        ragcheck.evaluate(
            question="What is photosynthesis?",
            answer="Plants convert sunlight to food.",
            contexts="This should be a list",
            llm_fn=lambda p: "{}",
        )


def test_mistake_passes_non_callable_llm():
    """Developer passes a string API key instead of a function by accident."""
    import ragcheck

    with pytest.raises(TypeError, match="llm_fn must be callable"):
        ragcheck.evaluate(
            question="What is photosynthesis?",
            answer="Plants convert sunlight to food.",
            contexts=["Some context."],
            llm_fn="gpt-4o",
        )


def test_mistake_empty_question():
    """Developer forgets to fill in the question field."""
    import ragcheck

    with pytest.raises(ValueError, match="question"):
        ragcheck.evaluate(
            question="",
            answer="Some answer.",
            contexts=["Some context."],
            llm_fn=lambda p: "{}",
        )


# ---------------------------------------------------------------------------
# Scenario 8: Developer uses passed() as a CI gate
# ---------------------------------------------------------------------------

def test_ci_gate_pattern():
    """Developer uses passed() to fail a CI pipeline -- the exact README pattern."""
    import ragcheck

    # Simulate a bad RAG response
    call_count = 0
    def bad_llm(prompt):
        nonlocal call_count
        call_count += 1
        score = 0.3 if call_count == 1 else 0.9
        return json.dumps({"score": score, "reasoning": "ok"})

    result = ragcheck.evaluate(
        question="What is the capital of France?",
        answer="London is the capital of France.",
        contexts=["Paris is the capital of France."],
        llm_fn=bad_llm,
    )

    assert result.passed(threshold=0.7) is False

    # Simulate how a developer would use this in CI
    ci_would_fail = not result.passed(threshold=0.7)
    assert ci_would_fail is True
