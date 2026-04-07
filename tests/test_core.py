import json
from unittest.mock import Mock

import pytest

from ragcheck import EvalResult, evaluate


def make_mock_llm(score: float, reasoning: str = "ok"):
    def mock_llm(prompt: str) -> str:
        return json.dumps({"score": score, "reasoning": reasoning})
    return mock_llm


QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


# --- Return type ---

def test_evaluate_returns_evalresult():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert isinstance(result, EvalResult)


# --- Score values ---

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


def test_scores_clamped_high():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(1.5))
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_precision == 1.0


def test_scores_clamped_low():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(-0.2))
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.0
    assert result.context_precision == 0.0


# --- Reasoning ---

def test_reasoning_populated():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8, "good answer"))
    assert "faithfulness" in result.reasoning
    assert "answer_relevance" in result.reasoning
    assert "context_precision" in result.reasoning


def test_reasoning_values():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.5, "partial match"))
    assert result.reasoning["faithfulness"] == "partial match"
    assert result.reasoning["answer_relevance"] == "partial match"
    assert result.reasoning["context_precision"] == "partial match"


# --- JSON parse error handling ---

def test_json_parse_failure_returns_zero():
    bad_llm = lambda prompt: "this is not json at all"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, bad_llm)
    assert result.faithfulness == 0.0
    assert "JSON parse error" in result.reasoning["faithfulness"]


def test_embedded_json_extraction():
    def llm_with_prose(prompt: str) -> str:
        return 'Here is my evaluation: {"score": 0.8, "reasoning": "well supported"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm_with_prose)
    assert result.faithfulness == 0.8
    assert result.reasoning["faithfulness"] == "well supported"


# --- llm_fn call count ---

def test_llm_called_three_times():
    mock = Mock(return_value=json.dumps({"score": 0.7, "reasoning": "ok"}))
    evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert mock.call_count == 3


# --- Context variations ---

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


def test_empty_contexts():
    result = evaluate(QUESTION, ANSWER, [], make_mock_llm(0.5))
    assert isinstance(result, EvalResult)


# --- EvalResult fields ---

def test_evalresult_fields_exist():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert hasattr(result, "faithfulness")
    assert hasattr(result, "answer_relevance")
    assert hasattr(result, "context_precision")
    assert hasattr(result, "reasoning")
