"""
Tests for ExecutionTraceService.

This test suite is matched to the exact implementation in:

backend/app/services/execution_trace_service.py
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.execution_trace_service import ExecutionTraceService


_UNSET = object()


@pytest.fixture
def service() -> ExecutionTraceService:
    return ExecutionTraceService()


def make_trace(
    service: ExecutionTraceService,
    *,
    selected_tool: Any = "document_retrieval",
    retrieval_plan: Any = _UNSET,
    retrieved_chunks: Any = _UNSET,
    retry_performed: Any = False,
    retry_count: Any = 0,
    retrieval_sufficient: Any = True,
    execution_result: Any = _UNSET,
    deterministic_answer_used: Any = True,
) -> list[dict[str, Any]]:
    """
    Call build_trace() with defaults while still allowing tests to pass
    explicit None, empty lists, empty dictionaries, and invalid values.
    """

    actual_plan = (
        {"top_k": 5}
        if retrieval_plan is _UNSET
        else retrieval_plan
    )

    actual_chunks = (
        [{"content": "default chunk"}]
        if retrieved_chunks is _UNSET
        else retrieved_chunks
    )

    actual_execution_result = (
        {"tool_outputs": []}
        if execution_result is _UNSET
        else execution_result
    )

    return service.build_trace(
        selected_tool=selected_tool,
        retrieval_plan=actual_plan,
        retrieved_chunks=actual_chunks,
        retry_performed=retry_performed,
        retry_count=retry_count,
        retrieval_sufficient=retrieval_sufficient,
        execution_result=actual_execution_result,
        deterministic_answer_used=deterministic_answer_used,
    )


class TestBaseTrace:
    def test_returns_four_steps_without_tool_outputs(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(service)

        assert len(result) == 4
        assert [item["step"] for item in result] == [1, 2, 3, 4]
        assert [item["component"] for item in result] == [
            "query_analysis",
            "document_retrieval",
            "retrieval_evaluation",
            "response_generation",
        ]

    def test_every_step_has_expected_keys(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(service)

        for item in result:
            assert set(item) == {
                "step",
                "component",
                "status",
                "duration_ms",
                "details",
            }

    def test_default_trace_values(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(service)

        assert result == [
            {
                "step": 1,
                "component": "query_analysis",
                "status": "success",
                "duration_ms": None,
                "details": "Primary tool selected: document_retrieval",
            },
            {
                "step": 2,
                "component": "document_retrieval",
                "status": "success",
                "duration_ms": None,
                "details": "Retrieved 1 chunks; top_k=5",
            },
            {
                "step": 3,
                "component": "retrieval_evaluation",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "Sufficient=True; "
                    "retry_performed=False; "
                    "retry_count=0"
                ),
            },
            {
                "step": 4,
                "component": "response_generation",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "A deterministic response builder produced "
                    "the final answer."
                ),
            },
        ]


class TestQueryAnalysisStep:
    @pytest.mark.parametrize(
        "selected_tool",
        [
            "financial_calculator",
            "company_comparison",
            "calculated_comparison",
            "",
            " ",
            None,
            123,
            True,
            [],
            {},
        ],
    )
    def test_selected_tool_is_interpolated_directly(
        self,
        service: ExecutionTraceService,
        selected_tool: Any,
    ) -> None:
        result = make_trace(
            service,
            selected_tool=selected_tool,
        )

        assert result[0] == {
            "step": 1,
            "component": "query_analysis",
            "status": "success",
            "duration_ms": None,
            "details": f"Primary tool selected: {selected_tool}",
        }


class TestDocumentRetrievalStep:
    @pytest.mark.parametrize(
        ("retrieved_chunks", "expected_status", "expected_count"),
        [
            ([], "failed", 0),
            (None, "failed", 0),
            ("invalid", "failed", 0),
            (123, "failed", 0),
            ({}, "failed", 0),
            ((), "failed", 0),
            ([{}], "success", 1),
            ([{}, {}], "success", 2),
        ],
    )
    def test_chunk_normalization_and_status(
        self,
        service: ExecutionTraceService,
        retrieved_chunks: Any,
        expected_status: str,
        expected_count: int,
    ) -> None:
        result = make_trace(
            service,
            retrieved_chunks=retrieved_chunks,
        )

        assert result[1]["status"] == expected_status
        assert result[1]["details"] == (
            f"Retrieved {expected_count} chunks; top_k=5"
        )

    @pytest.mark.parametrize(
        "retrieval_plan",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_invalid_plan_is_normalized_to_empty_dictionary(
        self,
        service: ExecutionTraceService,
        retrieval_plan: Any,
    ) -> None:
        result = make_trace(
            service,
            retrieval_plan=retrieval_plan,
        )

        assert result[1]["details"] == (
            "Retrieved 1 chunks; top_k=Unavailable"
        )

    def test_empty_plan_uses_unavailable_top_k(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            retrieval_plan={},
        )

        assert result[1]["details"] == (
            "Retrieved 1 chunks; top_k=Unavailable"
        )

    @pytest.mark.parametrize(
        "top_k",
        [
            0,
            -1,
            10,
            "5",
            None,
            True,
            [],
            {},
        ],
    )
    def test_top_k_value_is_interpolated_as_is(
        self,
        service: ExecutionTraceService,
        top_k: Any,
    ) -> None:
        result = make_trace(
            service,
            retrieval_plan={"top_k": top_k},
        )

        assert result[1]["details"] == (
            f"Retrieved 1 chunks; top_k={top_k}"
        )


class TestRetrievalEvaluationStep:
    @pytest.mark.parametrize(
        ("retrieval_sufficient", "expected_status"),
        [
            (True, "success"),
            (1, "success"),
            ("yes", "success"),
            ([1], "success"),
            ({"yes": True}, "success"),
            (False, "limited"),
            (0, "limited"),
            ("", "limited"),
            (None, "limited"),
            ([], "limited"),
            ({}, "limited"),
        ],
    )
    def test_status_uses_truthiness(
        self,
        service: ExecutionTraceService,
        retrieval_sufficient: Any,
        expected_status: str,
    ) -> None:
        result = make_trace(
            service,
            retrieval_sufficient=retrieval_sufficient,
        )

        assert result[2]["status"] == expected_status

    @pytest.mark.parametrize(
        ("sufficient", "retry_performed", "retry_count"),
        [
            (True, False, 0),
            (False, True, 1),
            (None, None, None),
            ("yes", "yes", "two"),
            ([], {}, []),
        ],
    )
    def test_details_interpolate_values_as_is(
        self,
        service: ExecutionTraceService,
        sufficient: Any,
        retry_performed: Any,
        retry_count: Any,
    ) -> None:
        result = make_trace(
            service,
            retrieval_sufficient=sufficient,
            retry_performed=retry_performed,
            retry_count=retry_count,
        )

        assert result[2]["details"] == (
            f"Sufficient={sufficient}; "
            f"retry_performed={retry_performed}; "
            f"retry_count={retry_count}"
        )


class TestExecutionResultNormalization:
    @pytest.mark.parametrize(
        "execution_result",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_invalid_execution_result_adds_no_tool_steps(
        self,
        service: ExecutionTraceService,
        execution_result: Any,
    ) -> None:
        result = make_trace(
            service,
            execution_result=execution_result,
        )

        assert len(result) == 4
        assert result[-1]["component"] == "response_generation"

    @pytest.mark.parametrize(
        "tool_outputs",
        [
            None,
            "invalid",
            123,
            {},
            (),
        ],
    )
    def test_invalid_tool_outputs_add_no_tool_steps(
        self,
        service: ExecutionTraceService,
        tool_outputs: Any,
    ) -> None:
        result = make_trace(
            service,
            execution_result={"tool_outputs": tool_outputs},
        )

        assert len(result) == 4


class TestToolOutputSteps:
    def test_successful_tool_output(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "financial_calculator",
                        "success": True,
                        "duration_ms": 12,
                    }
                ]
            },
        )

        assert result[3] == {
            "step": 4,
            "component": "financial_calculator",
            "status": "success",
            "duration_ms": 12.0,
            "details": "Tool completed successfully.",
        }

    def test_failed_tool_with_error(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "risk_analysis",
                        "success": False,
                        "error": "No evidence found.",
                        "duration_ms": 8.5,
                    }
                ]
            },
        )

        assert result[3] == {
            "step": 4,
            "component": "risk_analysis",
            "status": "failed",
            "duration_ms": 8.5,
            "details": "Tool failed: No evidence found.",
        }

    @pytest.mark.parametrize(
        "error",
        [
            None,
            "",
            0,
            False,
            [],
            {},
        ],
    )
    def test_failed_tool_without_truthy_error_uses_default_message(
        self,
        service: ExecutionTraceService,
        error: Any,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "tool",
                        "success": False,
                        "error": error,
                    }
                ]
            },
        )

        assert result[3]["details"] == "Tool execution failed."

    @pytest.mark.parametrize(
        ("success", "expected_status"),
        [
            (True, "success"),
            (1, "success"),
            ("yes", "success"),
            ([1], "success"),
            ({"ok": True}, "success"),
            (False, "failed"),
            (0, "failed"),
            ("", "failed"),
            (None, "failed"),
            ([], "failed"),
            ({}, "failed"),
        ],
    )
    def test_success_uses_boolean_conversion(
        self,
        service: ExecutionTraceService,
        success: Any,
        expected_status: str,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "tool",
                        "success": success,
                    }
                ]
            },
        )

        assert result[3]["status"] == expected_status

    @pytest.mark.parametrize(
        ("duration_ms", "expected"),
        [
            (0, 0.0),
            (12, 12.0),
            (-5, -5.0),
            (3.25, 3.25),
            (True, 1.0),
            (False, 0.0),
            (None, None),
            ("12", None),
            ([], None),
            ({}, None),
            ((), None),
        ],
    )
    def test_duration_handling(
        self,
        service: ExecutionTraceService,
        duration_ms: Any,
        expected: float | None,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "tool",
                        "success": True,
                        "duration_ms": duration_ms,
                    }
                ]
            },
        )

        assert result[3]["duration_ms"] == expected

    @pytest.mark.parametrize(
        ("tool", "expected_component"),
        [
            ("financial_calculator", "financial_calculator"),
            ("", ""),
            (None, "None"),
            (123, "123"),
            (True, "True"),
            ([], "[]"),
            ({}, "{}"),
        ],
    )
    def test_tool_name_is_stringified(
        self,
        service: ExecutionTraceService,
        tool: Any,
        expected_component: str,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": tool,
                        "success": True,
                    }
                ]
            },
        )

        assert result[3]["component"] == expected_component

    def test_missing_tool_name_uses_unknown_tool(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {"success": True}
                ]
            },
        )

        assert result[3]["component"] == "unknown_tool"

    @pytest.mark.parametrize(
        "invalid_item",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_invalid_tool_output_items_are_skipped(
        self,
        service: ExecutionTraceService,
        invalid_item: Any,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    invalid_item,
                    {
                        "tool": "valid_tool",
                        "success": True,
                    },
                ]
            },
        )

        assert len(result) == 5
        assert result[3]["step"] == 4
        assert result[3]["component"] == "valid_tool"
        assert result[4]["step"] == 5

    def test_valid_tool_order_is_preserved(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            execution_result={
                "tool_outputs": [
                    {"tool": "first", "success": True},
                    {"tool": "second", "success": False},
                    {"tool": "third", "success": True},
                ]
            },
        )

        assert [
            item["component"]
            for item in result[3:-1]
        ] == [
            "first",
            "second",
            "third",
        ]

        assert [
            item["step"]
            for item in result
        ] == [1, 2, 3, 4, 5, 6, 7]


class TestResponseGenerationStep:
    def test_deterministic_response_message(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            deterministic_answer_used=True,
        )

        assert result[-1] == {
            "step": 4,
            "component": "response_generation",
            "status": "success",
            "duration_ms": None,
            "details": (
                "A deterministic response builder produced "
                "the final answer."
            ),
        }

    @pytest.mark.parametrize(
        "value",
        [
            False,
            None,
            0,
            "",
            [],
            {},
        ],
    )
    def test_falsy_flag_uses_language_model_message(
        self,
        service: ExecutionTraceService,
        value: Any,
    ) -> None:
        result = make_trace(
            service,
            deterministic_answer_used=value,
        )

        assert result[-1]["details"] == (
            "The language model produced the final "
            "answer using verified context."
        )

    @pytest.mark.parametrize(
        "value",
        [
            True,
            1,
            "yes",
            [1],
            {"used": True},
        ],
    )
    def test_truthy_flag_uses_deterministic_message(
        self,
        service: ExecutionTraceService,
        value: Any,
    ) -> None:
        result = make_trace(
            service,
            deterministic_answer_used=value,
        )

        assert result[-1]["details"] == (
            "A deterministic response builder produced "
            "the final answer."
        )

    def test_response_generation_is_always_successful(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = make_trace(
            service,
            retrieved_chunks=[],
            retrieval_sufficient=False,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "failed_tool",
                        "success": False,
                    }
                ]
            },
            deterministic_answer_used=False,
        )

        assert result[-1]["status"] == "success"


class TestCompleteTrace:
    def test_complete_realistic_trace(
        self,
        service: ExecutionTraceService,
    ) -> None:
        result = service.build_trace(
            selected_tool="financial_calculator",
            retrieval_plan={"top_k": 8},
            retrieved_chunks=[
                {"content": "Apple revenue"},
                {"content": "More Apple revenue"},
            ],
            retry_performed=True,
            retry_count=1,
            retrieval_sufficient=True,
            execution_result={
                "tool_outputs": [
                    {
                        "tool": "document_retrieval",
                        "success": True,
                        "duration_ms": 20,
                    },
                    {
                        "tool": "financial_calculator",
                        "success": True,
                        "duration_ms": 5.5,
                    },
                ]
            },
            deterministic_answer_used=True,
        )

        assert result == [
            {
                "step": 1,
                "component": "query_analysis",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "Primary tool selected: financial_calculator"
                ),
            },
            {
                "step": 2,
                "component": "document_retrieval",
                "status": "success",
                "duration_ms": None,
                "details": "Retrieved 2 chunks; top_k=8",
            },
            {
                "step": 3,
                "component": "retrieval_evaluation",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "Sufficient=True; "
                    "retry_performed=True; "
                    "retry_count=1"
                ),
            },
            {
                "step": 4,
                "component": "document_retrieval",
                "status": "success",
                "duration_ms": 20.0,
                "details": "Tool completed successfully.",
            },
            {
                "step": 5,
                "component": "financial_calculator",
                "status": "success",
                "duration_ms": 5.5,
                "details": "Tool completed successfully.",
            },
            {
                "step": 6,
                "component": "response_generation",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "A deterministic response builder produced "
                    "the final answer."
                ),
            },
        ]
