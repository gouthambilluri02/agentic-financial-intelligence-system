"""
Tests for FinancialAgent.process_question().

This file currently covers:

- Invalid and empty question handling
- Retrieved chunk normalization
- Retrieval plan normalization
- Retry metadata normalization
- Invalid ToolExecutor result normalization
- No-evidence response behavior
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.app.agents.financial_agent import FinancialAgent


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def build_confidence_result() -> dict[str, Any]:
    """Return a predictable confidence result for mocked responses."""

    return {
        "score": 0.9,
        "percentage": 90.0,
        "level": "high",
        "reasons": ["Test confidence result."],
        "components": {},
        "method": "Test confidence method.",
    }


def build_execution_trace() -> list[dict[str, Any]]:
    """Return a predictable execution trace."""

    return [
        {
            "step": 1,
            "component": "retrieval",
            "status": "success",
            "details": "Retrieval completed.",
        },
        {
            "step": 2,
            "component": "response_generation",
            "status": "success",
            "details": "Response generated.",
        },
    ]


def build_valid_execution_result() -> dict[str, Any]:
    """Return a valid ToolExecutor response."""

    return {
        "success": True,
        "execution_plan": ["financial_calculator"],
        "executed_tools": ["financial_calculator"],
        "successful_tools": ["financial_calculator"],
        "failed_tools": [],
        "tool_outputs": [],
        "prompt_context": "",
        "duration_ms": 12.5,
        "error": None,
    }


def build_chunk() -> dict[str, Any]:
    """Return one valid retrieved financial-report chunk."""

    return {
        "content": "Apple reported total net sales of $100 billion.",
        "distance": 0.15,
        "metadata": {
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "document_type": "10-K",
            "source_file": "apple_2025_10k.pdf",
            "page": 10,
        },
    }


def build_agent() -> FinancialAgent:
    """
    Create a FinancialAgent whose external collaborators are mocked.

    The real FinancialAgent orchestration code is still executed.
    """

    agent = FinancialAgent()

    agent.company_detector = MagicMock()
    agent.intent_detector = MagicMock()
    agent.metric_detector = MagicMock()

    agent.tool_router = MagicMock()
    agent.execution_planner = MagicMock()
    agent.tool_executor = MagicMock()

    agent.retrieval_orchestrator = MagicMock()
    agent.prompt_builder = MagicMock()
    agent.deterministic_response_builder = MagicMock()

    agent.source_ranking_service = MagicMock()
    agent.confidence_scoring_service = MagicMock()
    agent.execution_trace_service = MagicMock()

    agent.memory = MagicMock()

    agent.company_detector.detect_companies.return_value = ["Apple"]
    agent.intent_detector.detect_intent.return_value = (
        "financial_calculation"
    )
    agent.metric_detector.detect_metric.return_value = "revenue"

    agent.memory.get_last_companies.return_value = []
    agent.memory.get_last_intent.return_value = None
    agent.memory.get_last_question.return_value = None

    agent.tool_router.select_tool.return_value = (
        "financial_calculator"
    )

    agent.execution_planner.create_plan.return_value = [
        "financial_calculator"
    ]

    agent.tool_executor.execute_plan.return_value = (
        build_valid_execution_result()
    )

    agent.source_ranking_service.rank_sources.return_value = [
        {
            "company": "Apple",
            "ticker": "AAPL",
            "source_file": "apple_2025_10k.pdf",
            "fiscal_year": 2025,
            "document_type": "10-K",
            "page": 11,
            "relevance_score": 0.95,
            "retrieval_distance": 0.15,
            "rank": 1,
        }
    ]

    agent.deterministic_response_builder.build_answer.return_value = (
        "Apple's calculated revenue result is available."
    )

    agent.confidence_scoring_service.calculate.return_value = (
        build_confidence_result()
    )

    agent.execution_trace_service.build_trace.return_value = (
        build_execution_trace()
    )

    return agent


def configure_retrieval(
    agent: FinancialAgent,
    *,
    chunks: Any,
    plan: Any = None,
    retry_performed: Any = False,
    retry_count: Any = 0,
    retrieval_sufficient: Any = True,
) -> None:
    """Configure the mocked RetrievalOrchestrator response."""

    result = {
        "chunks": chunks,
        "retry_performed": retry_performed,
        "retry_count": retry_count,
        "retrieval_sufficient": retrieval_sufficient,
    }

    if plan is not None:
        result["plan"] = plan

    agent.retrieval_orchestrator.retrieve.return_value = result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invalid and empty user questions."""

    @pytest.mark.parametrize(
        "question",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
            object(),
        ],
    )
    def test_non_string_question_returns_empty_response(
        self,
        question: Any,
    ) -> None:
        agent = build_agent()

        response = agent.process_question(question)

        assert response["answer"] == (
            "Please enter a financial question."
        )
        assert response["confidence"]["score"] == 0.0
        assert response["confidence"]["percentage"] == 0.0
        assert response["confidence"]["level"] == "low"
        assert response["sources"] == []
        assert response["execution_trace"] == []
        assert response["calculation"] is None
        assert response["comparison"] is None
        assert response["calculated_comparison"] is None
        assert response["risk_analysis"] is None
        assert response["tool_execution_success"] is False
        assert response["detected_companies"] == []
        assert response["detected_intent"] == "general_question"
        assert response["detected_metric"] is None
        assert response["retrieval_sufficient"] is False
        assert response["deterministic_answer_used"] is False

    @pytest.mark.parametrize(
        "question",
        [
            "",
            " ",
            "     ",
            "\n",
            "\n\n",
            "\t",
            "\t\t",
            " \n\t ",
        ],
    )
    def test_blank_string_returns_empty_response(
        self,
        question: str,
    ) -> None:
        agent = build_agent()

        response = agent.process_question(question)

        assert response["answer"] == (
            "Please enter a financial question."
        )
        assert response["execution_plan"] == []
        assert response["executed_tools"] == []
        assert response["successful_tools"] == []
        assert response["failed_tools"] == []
        assert response["tool_outputs"] == []
        assert response["retry_performed"] is False
        assert response["retry_count"] == 0

    def test_invalid_question_does_not_run_detection_pipeline(
        self,
    ) -> None:
        agent = build_agent()

        agent.process_question(None)

        agent.company_detector.detect_companies.assert_not_called()
        agent.intent_detector.detect_intent.assert_not_called()
        agent.metric_detector.detect_metric.assert_not_called()
        agent.tool_router.select_tool.assert_not_called()
        agent.retrieval_orchestrator.retrieve.assert_not_called()
        agent.execution_planner.create_plan.assert_not_called()
        agent.tool_executor.execute_plan.assert_not_called()

    def test_valid_question_is_trimmed_before_detection(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retrieval_sufficient=False,
        )

        agent.process_question("   Calculate Apple revenue.   ")

        agent.company_detector.detect_companies.assert_called_once_with(
            "Calculate Apple revenue."
        )
        agent.intent_detector.detect_intent.assert_called_once_with(
            "Calculate Apple revenue."
        )
        agent.metric_detector.detect_metric.assert_called_once_with(
            "Calculate Apple revenue."
        )


# ---------------------------------------------------------------------------
# Retrieved chunk normalization
# ---------------------------------------------------------------------------


class TestRetrievedChunkNormalization:
    """Tests for retrieval chunk normalization."""

    @pytest.mark.parametrize(
        "invalid_chunks",
        [
            None,
            "financial data",
            100,
            3.14,
            {},
            (),
            set(),
        ],
    )
    def test_invalid_chunks_are_normalized_to_empty_list(
        self,
        invalid_chunks: Any,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=invalid_chunks,
            plan={"strategy": "semantic"},
            retrieval_sufficient=True,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["answer"] == (
            "I could not find relevant information in "
            "the available financial reports."
        )
        assert response["sources"] == []
        assert response["retrieval_sufficient"] is False
        assert response["deterministic_answer_used"] is False

        agent.source_ranking_service.rank_sources.assert_called_once_with(
            []
        )

        agent.confidence_scoring_service.calculate.assert_called_once()

        confidence_call = (
            agent.confidence_scoring_service.calculate.call_args.kwargs
        )

        assert confidence_call["retrieved_chunks"] == []
        assert confidence_call["retrieval_sufficient"] is False

    def test_missing_chunks_defaults_to_empty_list(
        self,
    ) -> None:
        agent = build_agent()

        agent.retrieval_orchestrator.retrieve.return_value = {
            "plan": {"strategy": "semantic"},
            "retry_performed": False,
            "retry_count": 0,
            "retrieval_sufficient": True,
        }

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["answer"] == (
            "I could not find relevant information in "
            "the available financial reports."
        )
        assert response["retrieval_sufficient"] is False

    def test_valid_chunk_list_is_passed_to_tool_executor(
        self,
    ) -> None:
        agent = build_agent()
        chunks = [build_chunk()]

        configure_retrieval(
            agent,
            chunks=chunks,
            plan={"strategy": "semantic"},
            retrieval_sufficient=True,
        )

        agent.process_question(
            "Calculate Apple revenue."
        )

        executor_call = (
            agent.tool_executor.execute_plan.call_args.kwargs
        )

        assert executor_call["retrieved_chunks"] == chunks


# ---------------------------------------------------------------------------
# Retrieval plan normalization
# ---------------------------------------------------------------------------


class TestRetrievalPlanNormalization:
    """Tests for invalid retrieval plan values."""

    @pytest.mark.parametrize(
        "invalid_plan",
        [
            None,
            "semantic",
            100,
            2.5,
            [],
            (),
            set(),
        ],
    )
    def test_invalid_retrieval_plan_is_normalized_to_empty_dict(
        self,
        invalid_plan: Any,
    ) -> None:
        agent = build_agent()

        result = {
            "chunks": [],
            "plan": invalid_plan,
            "retry_performed": False,
            "retry_count": 0,
            "retrieval_sufficient": False,
        }

        agent.retrieval_orchestrator.retrieve.return_value = result

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["plan"] == {}

        trace_call = (
            agent.execution_trace_service.build_trace.call_args.kwargs
        )

        assert trace_call["retrieval_plan"] == {}

    def test_missing_retrieval_plan_defaults_to_empty_dict(
        self,
    ) -> None:
        agent = build_agent()

        agent.retrieval_orchestrator.retrieve.return_value = {
            "chunks": [],
            "retry_performed": False,
            "retry_count": 0,
            "retrieval_sufficient": False,
        }

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["plan"] == {}

    def test_valid_retrieval_plan_is_preserved(
        self,
    ) -> None:
        agent = build_agent()

        retrieval_plan = {
            "strategy": "company_metric_search",
            "queries": ["Apple revenue"],
        }

        configure_retrieval(
            agent,
            chunks=[],
            plan=retrieval_plan,
            retrieval_sufficient=False,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["plan"] == retrieval_plan


# ---------------------------------------------------------------------------
# Retry metadata normalization
# ---------------------------------------------------------------------------


class TestRetryMetadataNormalization:
    """Tests for retry metadata returned by retrieval."""

    @pytest.mark.parametrize(
        "invalid_retry_count",
        [
            None,
            "1",
            1.5,
            [],
            {},
            (),
        ],
    )
    def test_invalid_retry_count_defaults_to_zero(
        self,
        invalid_retry_count: Any,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retry_performed=True,
            retry_count=invalid_retry_count,
            retrieval_sufficient=False,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["retry_count"] == 0
        assert response["retry_performed"] is True

        confidence_call = (
            agent.confidence_scoring_service.calculate.call_args.kwargs
        )

        assert confidence_call["retry_count"] == 0

    def test_missing_retry_count_defaults_to_zero(
        self,
    ) -> None:
        agent = build_agent()

        agent.retrieval_orchestrator.retrieve.return_value = {
            "chunks": [],
            "plan": {},
            "retry_performed": False,
            "retrieval_sufficient": False,
        }

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["retry_count"] == 0

    def test_valid_integer_retry_count_is_preserved(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retry_performed=True,
            retry_count=2,
            retrieval_sufficient=False,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["retry_count"] == 2
        assert response["retry_performed"] is True

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            ("yes", True),
            ("", False),
            (None, False),
        ],
    )
    def test_retry_performed_is_converted_to_boolean(
        self,
        raw_value: Any,
        expected: bool,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retry_performed=raw_value,
            retry_count=0,
            retrieval_sufficient=False,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["retry_performed"] is expected


# ---------------------------------------------------------------------------
# Invalid ToolExecutor result normalization
# ---------------------------------------------------------------------------


class TestExecutionResultNormalization:
    """Tests for invalid ToolExecutor return values."""

    @pytest.mark.parametrize(
        "invalid_execution_result",
        [
            None,
            "invalid result",
            100,
            3.14,
            [],
            (),
            set(),
        ],
    )
    def test_invalid_execution_result_uses_fallback_dictionary(
        self,
        invalid_execution_result: Any,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[build_chunk()],
            plan={"strategy": "semantic"},
            retrieval_sufficient=True,
        )

        agent.tool_executor.execute_plan.return_value = (
            invalid_execution_result
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["tool_execution_success"] is False
        assert response["executed_tools"] == []
        assert response["successful_tools"] == []
        assert response["failed_tools"] == [
            "financial_calculator"
        ]
        assert response["tool_outputs"] == []
        assert response["tool_execution_duration_ms"] == 0.0
        assert response["deterministic_answer_used"] is True
        assert response["answer"] == (
            "Apple's calculated revenue result is available."
        )

        confidence_call = (
            agent.confidence_scoring_service.calculate.call_args.kwargs
        )

        normalized_result = confidence_call["execution_result"]

        assert normalized_result["success"] is False
        assert normalized_result["execution_plan"] == [
            "financial_calculator"
        ]
        assert normalized_result["executed_tools"] == []
        assert normalized_result["successful_tools"] == []
        assert normalized_result["failed_tools"] == [
            "financial_calculator"
        ]
        assert normalized_result["tool_outputs"] == []
        assert normalized_result["duration_ms"] == 0.0
        assert normalized_result["error"] == (
            "ToolExecutor must return a dictionary."
        )
        assert normalized_result["prompt_context"] == (
            "Tool execution returned an invalid result."
        )


# ---------------------------------------------------------------------------
# No-evidence response
# ---------------------------------------------------------------------------


class TestNoEvidenceResponse:
    """Tests for the early return when retrieval finds no chunks."""

    def test_empty_chunks_returns_limited_answer(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={"strategy": "semantic"},
            retrieval_sufficient=True,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        assert response["answer"] == (
            "I could not find relevant information in "
            "the available financial reports."
        )
        assert response["sources"] == []
        assert response["retrieval_sufficient"] is False
        assert response["deterministic_answer_used"] is False

    def test_no_evidence_path_updates_memory(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retrieval_sufficient=False,
        )

        agent.process_question(
            "Calculate Apple revenue."
        )

        agent.memory.update.assert_called_once_with(
            question="Calculate Apple revenue.",
            companies=["Apple"],
            intent="financial_calculation",
        )

    def test_no_evidence_path_skips_deterministic_builder(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retrieval_sufficient=False,
        )

        agent.process_question(
            "Calculate Apple revenue."
        )

        (
            agent.deterministic_response_builder
            .build_answer
            .assert_not_called()
        )

    def test_no_evidence_path_marks_trace_as_limited(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retrieval_sufficient=False,
        )

        response = agent.process_question(
            "Calculate Apple revenue."
        )

        final_trace_step = response["execution_trace"][-1]

        assert (
            final_trace_step["component"]
            == "response_generation"
        )
        assert final_trace_step["status"] == "limited"
        assert final_trace_step["details"] == (
            "The system returned a limited response because "
            "relevant report evidence was unavailable."
        )

    def test_no_evidence_path_calculates_confidence_as_insufficient(
        self,
    ) -> None:
        agent = build_agent()

        configure_retrieval(
            agent,
            chunks=[],
            plan={},
            retry_performed=True,
            retry_count=1,
            retrieval_sufficient=True,
        )

        agent.process_question(
            "Calculate Apple revenue."
        )

        confidence_call = (
            agent.confidence_scoring_service.calculate.call_args.kwargs
        )

        assert confidence_call["retrieved_chunks"] == []
        assert confidence_call["retrieval_sufficient"] is False
        assert confidence_call["retry_count"] == 1
        assert confidence_call["deterministic_answer_used"] is False