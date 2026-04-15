import json
import os
import tempfile
from unittest.mock import Mock

import pytest

import ragcheck
from ragcheck import EvalResult, evaluate, evaluate_batch, assert_no_regression


def make_mock_llm(score: float, reasoning: str = "ok"):
    """Returns an llm_fn that always responds with the given score and reasoning."""
    def mock_llm(prompt: str) -> str:
        return json.dumps({"score": score, "reasoning": reasoning})
    return mock_llm


QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


# ---------------------------------------------------------------------------
# Return type and shape
# ---------------------------------------------------------------------------

def test_evaluate_returns_evalresult():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert isinstance(result, EvalResult)
    assert hasattr(result, "faithfulness")
    assert hasattr(result, "answer_relevance")
    assert hasattr(result, "context_precision")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "parse_errors")


def test_reasoning_has_all_keys():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert set(result.reasoning.keys()) == {"faithfulness", "answer_relevance", "context_precision"}


def test_parse_errors_empty_on_clean_response():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.parse_errors == []


# ---------------------------------------------------------------------------
# Score values and clamping
# ---------------------------------------------------------------------------

def test_perfect_scores():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(1.0))
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_precision == 1.0


def test_zero_scores():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.0))
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.0
    assert result.context_precision == 0.0


def test_scores_clamped_above_one():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(1.5))
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_precision == 1.0


def test_scores_clamped_below_zero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(-0.2))
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.0
    assert result.context_precision == 0.0


def test_score_as_string_is_coerced():
    """LLMs sometimes return {"score": "0.8"} instead of {"score": 0.8}."""
    llm = lambda prompt: '{"score": "0.8", "reasoning": "string score"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.8


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def test_reasoning_values_match_llm_output():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.5, "partial match"))
    assert result.reasoning["faithfulness"] == "partial match"
    assert result.reasoning["answer_relevance"] == "partial match"
    assert result.reasoning["context_precision"] == "partial match"


def test_missing_reasoning_key_returns_empty_string():
    """If LLM omits 'reasoning', field should be empty string, not an error."""
    llm = lambda prompt: '{"score": 0.7}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.7
    assert result.reasoning["faithfulness"] == ""


def test_missing_score_key_defaults_to_zero():
    """If LLM omits 'score', default to 0.0."""
    llm = lambda prompt: '{"reasoning": "no score field here"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert result.reasoning["faithfulness"] == "no score field here"


# ---------------------------------------------------------------------------
# JSON parse resilience
# ---------------------------------------------------------------------------

def test_json_parse_failure_returns_zero():
    llm = lambda prompt: "this is not json at all"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0


def test_json_parse_failure_populates_parse_errors():
    """A parse failure should be visible in parse_errors, not just silently zero."""
    llm = lambda prompt: "this is not json at all"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert len(result.parse_errors) == 3  # all three metrics failed


def test_partial_parse_failure_recorded():
    """If only one metric fails, parse_errors has exactly one entry."""
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "not json"
        return json.dumps({"score": 0.8, "reasoning": "ok"})

    result = evaluate(QUESTION, ANSWER, CONTEXTS, mixed_llm)
    assert len(result.parse_errors) == 1
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.8


def test_embedded_json_extracted_from_prose():
    """LLMs often wrap JSON in explanation text — regex fallback should handle this."""
    llm = lambda prompt: 'Here is my evaluation: {"score": 0.8, "reasoning": "well supported"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.8
    assert result.reasoning["faithfulness"] == "well supported"
    assert result.parse_errors == []


def test_empty_response_returns_zero_with_parse_error():
    llm = lambda prompt: ""
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert len(result.parse_errors) == 3


def test_bare_number_response_falls_back():
    """json.loads("0.9") is valid JSON but not a dict — should not crash."""
    llm = lambda prompt: "0.9"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert len(result.parse_errors) == 3


# ---------------------------------------------------------------------------
# EvalResult.passed()
# ---------------------------------------------------------------------------

def test_passed_all_above_threshold():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.passed(threshold=0.7) is True


def test_passed_all_below_threshold():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.5))
    assert result.passed(threshold=0.7) is False


def test_passed_default_threshold_is_0_7():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.7))
    assert result.passed() is True


def test_passed_one_metric_below_threshold():
    """passed() should fail if even one metric misses the threshold."""
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        score = 0.9 if call_count != 2 else 0.5
        return json.dumps({"score": score, "reasoning": "ok"})

    result = evaluate(QUESTION, ANSWER, CONTEXTS, mixed_llm)
    assert result.passed(threshold=0.7) is False


def test_passed_exact_threshold_boundary():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.7))
    assert result.passed(threshold=0.7) is True

    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.699))
    assert result.passed(threshold=0.7) is False


# ---------------------------------------------------------------------------
# EvalResult.__str__()
# ---------------------------------------------------------------------------

def test_str_contains_all_metrics():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.85, "good"))
    s = str(result)
    assert "faithfulness" in s
    assert "answer_relevance" in s
    assert "context_precision" in s


def test_str_contains_score_values():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.85, "good"))
    s = str(result)
    assert "0.85" in s


def test_str_contains_parse_errors_when_present():
    llm = lambda prompt: "not json"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    s = str(result)
    assert "parse_errors" in s


def test_str_no_parse_errors_section_when_clean():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    s = str(result)
    assert "parse_errors" not in s


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_non_callable_llm_fn_raises_typeerror():
    with pytest.raises(TypeError, match="llm_fn must be callable"):
        evaluate(QUESTION, ANSWER, CONTEXTS, "not a function")


def test_empty_question_raises_valueerror():
    with pytest.raises(ValueError, match="question"):
        evaluate("", ANSWER, CONTEXTS, make_mock_llm(0.9))


def test_whitespace_only_question_raises_valueerror():
    with pytest.raises(ValueError, match="question"):
        evaluate("   ", ANSWER, CONTEXTS, make_mock_llm(0.9))


def test_empty_answer_raises_valueerror():
    with pytest.raises(ValueError, match="answer"):
        evaluate(QUESTION, "", CONTEXTS, make_mock_llm(0.9))


def test_non_list_contexts_raises_typeerror():
    with pytest.raises(TypeError, match="contexts must be a list"):
        evaluate(QUESTION, ANSWER, "not a list", make_mock_llm(0.9))


# ---------------------------------------------------------------------------
# llm_fn call behaviour
# ---------------------------------------------------------------------------

def test_llm_called_exactly_three_times():
    mock = Mock(return_value=json.dumps({"score": 0.7, "reasoning": "ok"}))
    evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert mock.call_count == 3


def test_llm_fn_exception_propagates():
    """Errors from llm_fn should not be silently swallowed."""
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("API call failed")

    with pytest.raises(RuntimeError, match="API call failed"):
        evaluate(QUESTION, ANSWER, CONTEXTS, failing_llm)


# ---------------------------------------------------------------------------
# Context variations
# ---------------------------------------------------------------------------

def test_multiple_contexts():
    contexts = [
        "Paris is the capital of France.",
        "France is known for the Eiffel Tower.",
        "The Seine river flows through Paris.",
    ]
    result = evaluate(QUESTION, ANSWER, contexts, make_mock_llm(0.9))
    assert isinstance(result, EvalResult)


def test_single_context():
    result = evaluate(QUESTION, ANSWER, ["Paris is in France."], make_mock_llm(0.9))
    assert isinstance(result, EvalResult)


def test_empty_contexts_does_not_raise():
    result = evaluate(QUESTION, ANSWER, [], make_mock_llm(0.5))
    assert isinstance(result, EvalResult)


# ---------------------------------------------------------------------------
# context_recall (optional metric)
# ---------------------------------------------------------------------------

def test_context_recall_absent_by_default():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.context_recall is None
    assert "context_recall" not in result.reasoning


def test_context_recall_present_when_requested():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), include_context_recall=True)
    assert result.context_recall == 0.8
    assert "context_recall" in result.reasoning


def test_context_recall_calls_llm_four_times():
    mock = Mock(return_value=json.dumps({"score": 0.7, "reasoning": "ok"}))
    evaluate(QUESTION, ANSWER, CONTEXTS, mock, include_context_recall=True)
    assert mock.call_count == 4


def test_context_recall_included_in_passed():
    """passed() must gate on context_recall when it is present."""
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # First 3 calls: core metrics at 0.9; 4th call: recall at 0.3
        score = 0.9 if call_count < 4 else 0.3
        return json.dumps({"score": score, "reasoning": "ok"})

    result = evaluate(QUESTION, ANSWER, CONTEXTS, mixed_llm, include_context_recall=True)
    assert result.context_recall == 0.3
    assert result.passed(threshold=0.7) is False  # recall drags it down


def test_context_recall_parse_error_recorded():
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            return "not json"
        return json.dumps({"score": 0.8, "reasoning": "ok"})

    result = evaluate(QUESTION, ANSWER, CONTEXTS, mixed_llm, include_context_recall=True)
    assert result.context_recall == 0.0
    assert len(result.parse_errors) == 1


# ---------------------------------------------------------------------------
# extra_metrics (custom metrics)
# ---------------------------------------------------------------------------

def test_extra_metric_score_returned():
    def conciseness_prompt(q, a, c):
        return f"Rate conciseness of: {a}"

    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.75),
        extra_metrics={"conciseness": conciseness_prompt},
    )
    assert result.extra_metrics["conciseness"] == 0.75


def test_extra_metric_reasoning_returned():
    def my_prompt(q, a, c):
        return "some prompt"

    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.6, "brief and clear"),
        extra_metrics={"conciseness": my_prompt},
    )
    assert result.reasoning["conciseness"] == "brief and clear"


def test_extra_metric_included_in_passed():
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        score = 0.9 if call_count <= 3 else 0.2
        return json.dumps({"score": score, "reasoning": "ok"})

    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, mixed_llm,
        extra_metrics={"conciseness": lambda q, a, c: "rate this"},
    )
    assert result.extra_metrics["conciseness"] == 0.2
    assert result.passed(threshold=0.7) is False


def test_multiple_extra_metrics():
    mock = Mock(return_value=json.dumps({"score": 0.8, "reasoning": "ok"}))
    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, mock,
        extra_metrics={
            "conciseness": lambda q, a, c: "prompt1",
            "tone": lambda q, a, c: "prompt2",
        },
    )
    assert "conciseness" in result.extra_metrics
    assert "tone" in result.extra_metrics
    assert mock.call_count == 5  # 3 core + 2 extra


def test_no_extra_metrics_by_default():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.extra_metrics == {}


def test_extra_metric_parse_error_recorded():
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            return "not json"
        return json.dumps({"score": 0.8, "reasoning": "ok"})

    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, mixed_llm,
        extra_metrics={"bad_metric": lambda q, a, c: "prompt"},
    )
    assert len(result.parse_errors) == 1
    assert result.extra_metrics["bad_metric"] == 0.0


# ---------------------------------------------------------------------------
# EvalResult.to_dict() and to_json()
# ---------------------------------------------------------------------------

def test_to_dict_contains_core_fields():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8, "ok"))
    d = result.to_dict()
    assert "faithfulness" in d
    assert "answer_relevance" in d
    assert "context_precision" in d
    assert "reasoning" in d
    assert "parse_errors" in d


def test_to_dict_excludes_context_recall_when_absent():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert "context_recall" not in result.to_dict()


def test_to_dict_includes_context_recall_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), include_context_recall=True)
    assert "context_recall" in result.to_dict()
    assert result.to_dict()["context_recall"] == 0.8


def test_to_dict_excludes_extra_metrics_when_empty():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert "extra_metrics" not in result.to_dict()


def test_to_dict_includes_extra_metrics_when_present():
    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9),
        extra_metrics={"conciseness": lambda q, a, c: "p"},
    )
    assert result.to_dict()["extra_metrics"]["conciseness"] == 0.9


def test_to_json_is_valid_json():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.7))
    raw = result.to_json()
    parsed = json.loads(raw)
    assert parsed["faithfulness"] == 0.7


def test_to_json_kwargs_forwarded():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.7))
    raw = result.to_json(indent=2)
    assert "\n" in raw  # indented output has newlines


def test_to_dict_scores_match_result_fields():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.65))
    d = result.to_dict()
    assert d["faithfulness"] == result.faithfulness
    assert d["answer_relevance"] == result.answer_relevance
    assert d["context_precision"] == result.context_precision


# ---------------------------------------------------------------------------
# evaluate_batch()
# ---------------------------------------------------------------------------

BATCH = [
    {"question": "What is the capital of France?", "answer": "Paris.", "contexts": ["Paris is the capital."]},
    {"question": "What is 2+2?", "answer": "4.", "contexts": ["Basic arithmetic."]},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare.", "contexts": ["Shakespeare wrote Hamlet."]},
]


def test_evaluate_batch_returns_list_of_evalresults():
    results = evaluate_batch(BATCH, make_mock_llm(0.8))
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, EvalResult) for r in results)


def test_evaluate_batch_preserves_order():
    """Results should correspond to items in the same order."""
    scores = [0.9, 0.5, 0.7]
    idx = 0
    def ordered_llm(prompt: str) -> str:
        nonlocal idx
        score = scores[idx // 3]
        idx += 1
        return json.dumps({"score": score, "reasoning": "ok"})

    results = evaluate_batch(BATCH, ordered_llm)
    assert results[0].faithfulness == 0.9
    assert results[1].faithfulness == 0.5
    assert results[2].faithfulness == 0.7


def test_evaluate_batch_empty_list():
    results = evaluate_batch([], make_mock_llm(0.8))
    assert results == []


def test_evaluate_batch_kwargs_forwarded():
    """include_context_recall should be forwarded to each evaluate() call."""
    mock = Mock(return_value=json.dumps({"score": 0.7, "reasoning": "ok"}))
    results = evaluate_batch(BATCH, mock, include_context_recall=True)
    assert all(r.context_recall is not None for r in results)
    # 3 items × 4 calls each (3 core + 1 recall) = 12
    assert mock.call_count == 12


def test_evaluate_batch_filter_passing():
    """Typical usage: filter batch results for those that pass."""
    call_count = 0
    def mixed_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # Item 1 (calls 1-3): 0.9, Item 2 (calls 4-6): 0.4, Item 3 (calls 7-9): 0.9
        score = 0.4 if 4 <= call_count <= 6 else 0.9
        return json.dumps({"score": score, "reasoning": "ok"})

    results = evaluate_batch(BATCH, mixed_llm)
    passing = [r for r in results if r.passed(threshold=0.7)]
    assert len(passing) == 2


def test_evaluate_batch_propagates_llm_exception():
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("API down")

    with pytest.raises(RuntimeError, match="API down"):
        evaluate_batch(BATCH, failing_llm)


# ---------------------------------------------------------------------------
# __str__ with new fields
# ---------------------------------------------------------------------------

def test_str_includes_context_recall_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), include_context_recall=True)
    assert "context_recall" in str(result)


def test_str_excludes_context_recall_when_absent():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert "context_recall" not in str(result)


def test_str_includes_extra_metrics():
    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9),
        extra_metrics={"conciseness": lambda q, a, c: "p"},
    )
    assert "conciseness" in str(result)


# ---------------------------------------------------------------------------
# Helper for v1.0.0 tests — build an EvalResult directly from field values
# ---------------------------------------------------------------------------

def make_result(faithfulness=0.9, answer_relevance=0.9, context_precision=0.9,
                context_recall=None, failure_modes=None, tokens_used=0,
                estimated_cost_usd=0.0, extra_metrics=None):
    return EvalResult(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        reasoning={
            "faithfulness": "ok",
            "answer_relevance": "ok",
            "context_precision": "ok",
        },
        context_recall=context_recall,
        failure_modes=failure_modes or [],
        tokens_used=tokens_used,
        estimated_cost_usd=estimated_cost_usd,
        extra_metrics=extra_metrics or {},
    )


# ---------------------------------------------------------------------------
# Failure mode classification (v1.0.0)
# ---------------------------------------------------------------------------

def test_hallucination_label_when_faithfulness_low():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.3))
    assert "hallucination" in result.failure_modes


def test_off_topic_label_when_relevance_low():
    """Low answer_relevance → off_topic label."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        scores = [0.9, 0.3, 0.9]
        idx = min(call_count[0] - 1, 2)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert "off_topic" in result.failure_modes


def test_retrieval_miss_label_when_precision_low():
    """Low context_precision → retrieval_miss label."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        scores = [0.9, 0.9, 0.3]
        idx = min(call_count[0] - 1, 2)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert "retrieval_miss" in result.failure_modes


def test_context_noise_label_when_recall_low():
    """Low context_recall → context_noise label (only when include_context_recall=True)."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        scores = [0.9, 0.9, 0.9, 0.3]
        idx = min(call_count[0] - 1, 3)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock, include_context_recall=True)
    assert "context_noise" in result.failure_modes


def test_multiple_failure_labels_when_multiple_scores_low():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    assert "hallucination" in result.failure_modes
    assert "off_topic" in result.failure_modes
    assert "retrieval_miss" in result.failure_modes


def test_no_failure_modes_when_all_scores_high():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.failure_modes == []


def test_failure_modes_is_list():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert isinstance(result.failure_modes, list)


def test_failure_modes_in_to_dict_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    d = result.to_dict()
    assert "failure_modes" in d
    assert isinstance(d["failure_modes"], list)


def test_failure_modes_in_to_markdown():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    md = result.to_markdown()
    assert len(result.failure_modes) > 0
    assert any(m in md for m in result.failure_modes)


def test_failure_modes_absent_from_to_dict_when_empty():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert "failure_modes" not in result.to_dict()


# ---------------------------------------------------------------------------
# Token and cost tracking (v1.0.0)
# ---------------------------------------------------------------------------

def test_tokens_used_positive_after_eval():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert result.tokens_used > 0


def test_estimated_cost_zero_when_no_model():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert result.estimated_cost_usd == 0.0


def test_estimated_cost_nonzero_for_known_model():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    assert result.estimated_cost_usd > 0.0


def test_custom_pricing_overrides_builtin():
    r1 = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o")
    r2 = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
                  pricing={"input": 0.000001, "output": 0.000002})
    assert r2.estimated_cost_usd != r1.estimated_cost_usd


def test_cost_scales_with_more_llm_calls():
    r_no_recall = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    r_with_recall = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
                             model="gpt-4o-mini", include_context_recall=True)
    assert r_with_recall.tokens_used > r_no_recall.tokens_used


def test_tokens_in_to_dict_when_nonzero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    d = result.to_dict()
    assert "tokens_used" in d
    assert "estimated_cost_usd" in d


def test_cost_in_to_markdown_when_nonzero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    md = result.to_markdown()
    assert "Cost" in md or "$" in md


def test_unknown_model_name_does_not_crash():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="my-custom-model-v99")
    assert result.estimated_cost_usd == 0.0
    assert result.tokens_used > 0


# ---------------------------------------------------------------------------
# to_markdown() (v1.0.0)
# ---------------------------------------------------------------------------

def test_to_markdown_contains_all_core_metric_names():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    md = result.to_markdown()
    assert "faithfulness" in md
    assert "answer_relevance" in md
    assert "context_precision" in md


def test_to_markdown_contains_score_values():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.75))
    md = result.to_markdown()
    assert "0.75" in md


def test_to_markdown_contains_reasoning():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8, "looks good"))
    md = result.to_markdown()
    assert "looks good" in md


def test_to_markdown_contains_failure_modes_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    md = result.to_markdown()
    assert any(m in md for m in result.failure_modes)


def test_to_markdown_omits_failure_modes_when_empty():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    md = result.to_markdown()
    assert "hallucination" not in md
    assert "off_topic" not in md


def test_to_markdown_contains_cost_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    md = result.to_markdown()
    assert "Cost" in md or "$" in md


def test_to_markdown_includes_context_recall_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), include_context_recall=True)
    md = result.to_markdown()
    assert "context_recall" in md


def test_to_markdown_includes_extra_metrics():
    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
        extra_metrics={"conciseness": lambda q, a, c: "rate conciseness"},
    )
    md = result.to_markdown()
    assert "conciseness" in md


def test_to_markdown_is_valid_markdown_table():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    md = result.to_markdown()
    assert "| Metric |" in md
    assert "|---|" in md


# ---------------------------------------------------------------------------
# save_baseline() and assert_no_regression() (v1.0.0)
# ---------------------------------------------------------------------------

def test_save_baseline_writes_json_file():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_save_baseline_file_is_valid_json():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "faithfulness" in data
    finally:
        os.unlink(path)


def test_assert_no_regression_passes_when_equal():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert_no_regression(result.to_dict(), result)


def test_assert_no_regression_passes_within_tolerance():
    baseline = make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8)
    new_result = make_result(faithfulness=0.76, answer_relevance=0.76, context_precision=0.76)
    assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_assert_no_regression_raises_when_beyond_tolerance():
    baseline = make_result(faithfulness=0.9)
    new_result = make_result(faithfulness=0.7)
    with pytest.raises(AssertionError):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_assert_no_regression_error_message_names_metric():
    baseline = make_result(faithfulness=0.9)
    new_result = make_result(faithfulness=0.7)
    with pytest.raises(AssertionError, match="faithfulness"):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_assert_no_regression_accepts_dict_baseline():
    baseline_dict = {"faithfulness": 0.8, "answer_relevance": 0.8, "context_precision": 0.8}
    new_result = make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8)
    assert_no_regression(baseline_dict, new_result)


def test_assert_no_regression_checks_context_recall_if_in_baseline():
    baseline = make_result(context_recall=0.9)
    new_result = make_result(context_recall=0.7)
    with pytest.raises(AssertionError, match="context_recall"):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_save_and_reload_round_trips_correctly():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        assert_no_regression(path, result)  # load from file path — should not raise
    finally:
        os.unlink(path)
