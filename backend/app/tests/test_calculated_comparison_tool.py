from __future__ import annotations

from typing import Any

import pytest

from backend.app.tools.calculated_comparison_tool import (
    CalculatedComparisonTool,
)


class FakeDecompositionService:
    """
    Configurable fake query decomposition service.
    """

    def __init__(
        self,
        result: Any,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def decompose(
        self,
        question: str,
        companies: list[str],
        intent: str,
        metric: str | None,
    ) -> Any:
        self.calls.append(
            {
                "question": question,
                "companies": companies,
                "intent": intent,
                "metric": metric,
            }
        )

        return self.result


class FakeCalculationService:
    """
    Configurable fake financial calculation service.
    """

    def __init__(
        self,
        result: Any,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def calculate(
        self,
        question: str,
        metric: str,
        companies: list[str],
        retrieved_chunks: list[dict[str, Any]],
    ) -> Any:
        self.calls.append(
            {
                "question": question,
                "metric": metric,
                "companies": companies,
                "retrieved_chunks": retrieved_chunks,
            }
        )

        return self.result


class FakeReasoningService:
    """
    Configurable fake multi-step reasoning service.
    """

    def __init__(
        self,
        result: Any,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def reason(
        self,
        decomposition: dict[str, Any],
        calculation: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "decomposition": decomposition,
                "calculation": calculation,
            }
        )

        return self.result


@pytest.fixture
def valid_chunks() -> list[dict[str, Any]]:
    return [
        {
            "content": (
                "Year Ended September 28, 2024 2023 "
                "Total net sales 391,035 383,285"
            ),
            "metadata": {
                "company": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "document_type": "10K",
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
            "distance": 0.21,
        },
        {
            "content": (
                "Year Ended June 30, 2024 2023 "
                "Revenue 245,122 211,915"
            ),
            "metadata": {
                "company": "Microsoft",
                "ticker": "MSFT",
                "fiscal_year": 2024,
                "document_type": "10K",
                "source_file": "microsoft_2024_10k.pdf",
                "page": 79,
            },
            "distance": 0.23,
        },
    ]


@pytest.fixture
def valid_decomposition() -> dict[str, Any]:
    return {
        "is_complex": True,
        "reasoning_type": "calculated_comparison",
        "calculation_requested": True,
        "comparison_requested": True,
        "conclusion_requested": True,
        "calculated_comparison": True,
        "companies": [
            "Apple",
            "Microsoft",
        ],
        "metric": "revenue",
        "subtasks": [
            {
                "step": 1,
                "type": "calculation",
                "company": "Apple",
                "metric": "revenue",
                "description": (
                    "Calculate Apple's revenue growth."
                ),
            },
            {
                "step": 2,
                "type": "calculation",
                "company": "Microsoft",
                "metric": "revenue",
                "description": (
                    "Calculate Microsoft's revenue growth."
                ),
            },
            {
                "step": 3,
                "type": "calculated_comparison",
                "company": None,
                "metric": "revenue",
                "description": (
                    "Compare the verified calculated "
                    "revenue growth rates."
                ),
            },
            {
                "step": 4,
                "type": "conclusion",
                "company": None,
                "metric": "revenue",
                "description": (
                    "Identify the stronger performer using "
                    "only verified deterministic results."
                ),
            },
        ],
        "required_capabilities": [
            "document_retrieval",
            "financial_calculator",
            "multi_step_reasoning",
        ],
    }


@pytest.fixture
def valid_calculation() -> dict[str, Any]:
    return {
        "success": True,
        "calculation_type": "growth_rate",
        "metric": "revenue",
        "results": [
            {
                "company": "Apple",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 391035.0,
                "previous_value": 383285.0,
                "result": 2.02,
                "unit": "percent",
                "formula": (
                    "((current - previous) / previous) * 100"
                ),
            },
            {
                "company": "Microsoft",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 245122.0,
                "previous_value": 211915.0,
                "result": 15.67,
                "unit": "percent",
                "formula": (
                    "((current - previous) / previous) * 100"
                ),
            },
        ],
        "error": None,
    }


@pytest.fixture
def valid_reasoning() -> dict[str, Any]:
    return {
        "success": True,
        "reasoning_type": "calculated_comparison",
        "comparison_type": "growth_rate",
        "metric": "revenue",
        "ranked_results": [
            {
                "company": "Microsoft",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 245122.0,
                "previous_value": 211915.0,
                "result": 15.67,
                "unit": "percent",
            },
            {
                "company": "Apple",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 391035.0,
                "previous_value": 383285.0,
                "result": 2.02,
                "unit": "percent",
            },
        ],
        "strongest_company": "Microsoft",
        "strongest_result": 15.67,
        "weakest_company": "Apple",
        "weakest_result": 2.02,
        "difference_percentage_points": 13.65,
        "conclusion": (
            "Microsoft was the stronger performer based on "
            "revenue growth. Microsoft reported 15.67% growth "
            "compared with Apple's 2.02%, a difference of "
            "13.65 percentage points."
        ),
        "error": None,
    }


def build_tool(
    decomposition_result: Any,
    calculation_result: Any,
    reasoning_result: Any,
) -> tuple[
    CalculatedComparisonTool,
    FakeDecompositionService,
    FakeCalculationService,
    FakeReasoningService,
]:
    decomposition_service = FakeDecompositionService(
        result=decomposition_result,
    )

    calculation_service = FakeCalculationService(
        result=calculation_result,
    )

    reasoning_service = FakeReasoningService(
        result=reasoning_result,
    )

    tool = CalculatedComparisonTool(
        calculation_service=calculation_service,
        decomposition_service=decomposition_service,
        reasoning_service=reasoning_service,
    )

    return (
        tool,
        decomposition_service,
        calculation_service,
        reasoning_service,
    )


def test_successful_calculated_comparison(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    (
        tool,
        decomposition_service,
        calculation_service,
        reasoning_service,
    ) = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question=(
            "Compare Apple and Microsoft revenue growth and "
            "identify the stronger performer."
        ),
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is True
    assert result["tool"] == "calculated_comparison"
    assert result["decomposition"] == valid_decomposition
    assert result["calculation"] == valid_calculation
    assert result["reasoning"] == valid_reasoning
    assert result["error"] is None

    assert (
        "VERIFIED CALCULATED COMPARISON"
        in result["prompt_context"]
    )

    assert (
        "Strongest performer: Microsoft"
        in result["prompt_context"]
    )

    assert (
        "Difference: 13.65 percentage points"
        in result["prompt_context"]
    )

    assert len(decomposition_service.calls) == 1
    assert len(calculation_service.calls) == 1
    assert len(reasoning_service.calls) == 1


def test_services_receive_normalized_inputs(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    (
        tool,
        decomposition_service,
        calculation_service,
        _,
    ) = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    tool.run(
        question=(
            "  Compare Apple and Microsoft revenue growth.  "
        ),
        metric="revenue",
        companies=[
            " Apple ",
            "Microsoft",
            "apple",
            "",
            123,
        ],
        retrieved_chunks=valid_chunks,
    )

    decomposition_call = decomposition_service.calls[0]
    calculation_call = calculation_service.calls[0]

    assert decomposition_call["question"] == (
        "Compare Apple and Microsoft revenue growth."
    )

    assert decomposition_call["companies"] == [
        "Apple",
        "Microsoft",
    ]

    assert decomposition_call["intent"] == "comparison"
    assert decomposition_call["metric"] == "revenue"

    assert calculation_call["question"] == (
        "Compare Apple and Microsoft revenue growth."
    )

    assert calculation_call["companies"] == [
        "Apple",
        "Microsoft",
    ]

    assert calculation_call["metric"] == "revenue"
    assert calculation_call["retrieved_chunks"] == valid_chunks


@pytest.mark.parametrize(
    ("question", "expected_error"),
    [
        (
            None,
            (
                "A non-empty financial comparison question "
                "is required."
            ),
        ),
        (
            "",
            (
                "A non-empty financial comparison question "
                "is required."
            ),
        ),
        (
            "   ",
            (
                "A non-empty financial comparison question "
                "is required."
            ),
        ),
    ],
)
def test_invalid_question_returns_failure(
    question: Any,
    expected_error: str,
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    (
        tool,
        decomposition_service,
        calculation_service,
        reasoning_service,
    ) = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question=question,
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["tool"] == "calculated_comparison"
    assert result["error"] == expected_error
    assert result["decomposition"] is None
    assert result["calculation"] is None
    assert result["reasoning"] is None
    assert expected_error in result["prompt_context"]

    assert decomposition_service.calls == []
    assert calculation_service.calls == []
    assert reasoning_service.calls == []


@pytest.mark.parametrize(
    "metric",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_invalid_metric_returns_failure(
    metric: Any,
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare revenue growth.",
        metric=metric,
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "A supported financial metric is required for "
        "a calculated comparison."
    )


def test_companies_must_be_a_list(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare revenue growth.",
        metric="revenue",
        companies="Apple, Microsoft",
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "companies must be provided as a list."
    )


@pytest.mark.parametrize(
    "companies",
    [
        [],
        ["Apple"],
        ["Apple", " apple "],
        ["", None, 123],
    ],
)
def test_at_least_two_unique_companies_are_required(
    companies: list[Any],
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare revenue growth.",
        metric="revenue",
        companies=companies,
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "At least two companies are required for "
        "a calculated comparison."
    )


def test_retrieved_chunks_must_be_a_list(
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks="invalid",
    )

    assert result["success"] is False
    assert result["error"] == (
        "retrieved_chunks must be provided as a list."
    )


def test_empty_retrieved_chunks_returns_failure(
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=[],
    )

    assert result["success"] is False
    assert result["error"] == (
        "No retrieved financial-report chunks were "
        "provided for the calculated comparison."
    )


@pytest.mark.parametrize(
    "decomposition_result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_invalid_decomposition_result_returns_failure(
    decomposition_result: Any,
    valid_chunks: list[dict[str, Any]],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    (
        tool,
        _,
        calculation_service,
        reasoning_service,
    ) = build_tool(
        decomposition_result=decomposition_result,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "Query decomposition returned an invalid result."
    )

    assert calculation_service.calls == []
    assert reasoning_service.calls == []


@pytest.mark.parametrize(
    "calculation_result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_invalid_calculation_result_returns_failure(
    calculation_result: Any,
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    (
        tool,
        _,
        _,
        reasoning_service,
    ) = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=calculation_result,
        reasoning_result=valid_reasoning,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "Financial calculation returned an invalid result."
    )

    assert result["decomposition"] == valid_decomposition
    assert result["calculation"] is None
    assert reasoning_service.calls == []


@pytest.mark.parametrize(
    "reasoning_result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_invalid_reasoning_result_returns_failure(
    reasoning_result: Any,
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=reasoning_result,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "Multi-step reasoning returned an invalid result."
    )

    assert result["decomposition"] == valid_decomposition
    assert result["calculation"] == valid_calculation
    assert result["reasoning"] is None


def test_calculation_failure_is_propagated(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
) -> None:
    calculation_failure = {
        "success": False,
        "calculation_type": "growth_rate",
        "metric": "revenue",
        "results": [],
        "error": "Revenue values were unavailable.",
    }

    reasoning_failure = {
        "success": False,
        "reasoning_type": "calculated_comparison",
        "comparison_type": None,
        "metric": None,
        "ranked_results": [],
        "strongest_company": None,
        "strongest_result": None,
        "weakest_company": None,
        "weakest_result": None,
        "difference_percentage_points": None,
        "conclusion": None,
        "error": None,
    }

    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=calculation_failure,
        reasoning_result=reasoning_failure,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "Revenue values were unavailable."
    )

    assert result["calculation"] == calculation_failure
    assert result["reasoning"] == reasoning_failure
    assert (
        "VERIFIED CALCULATED COMPARISON"
        in result["prompt_context"]
    )


def test_reasoning_failure_takes_priority(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
) -> None:
    reasoning_failure = {
        "success": False,
        "reasoning_type": "calculated_comparison",
        "comparison_type": None,
        "metric": None,
        "ranked_results": [],
        "strongest_company": None,
        "strongest_result": None,
        "weakest_company": None,
        "weakest_result": None,
        "difference_percentage_points": None,
        "conclusion": None,
        "error": (
            "At least two verified calculations are required."
        ),
    }

    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=reasoning_failure,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "At least two verified calculations are required."
    )


def test_default_failure_message_is_used_when_errors_are_missing(
    valid_chunks: list[dict[str, Any]],
    valid_decomposition: dict[str, Any],
) -> None:
    calculation_failure = {
        "success": False,
        "metric": "revenue",
        "results": [],
        "error": None,
    }

    reasoning_failure = {
        "success": False,
        "error": None,
    }

    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=calculation_failure,
        reasoning_result=reasoning_failure,
    )

    result = tool.run(
        question="Compare Apple and Microsoft revenue growth.",
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=valid_chunks,
    )

    assert result["success"] is False
    assert result["error"] == (
        "The calculated comparison could not be completed."
    )


def test_prompt_context_contains_verified_calculations(
    valid_decomposition: dict[str, Any],
    valid_calculation: dict[str, Any],
    valid_reasoning: dict[str, Any],
) -> None:
    tool, _, _, _ = build_tool(
        decomposition_result=valid_decomposition,
        calculation_result=valid_calculation,
        reasoning_result=valid_reasoning,
    )

    prompt_context = tool._build_prompt_context(
        decomposition=valid_decomposition,
        calculation=valid_calculation,
        reasoning=valid_reasoning,
    )

    assert "Metric: Revenue" in prompt_context
    assert "Company: Apple" in prompt_context
    assert "Company: Microsoft" in prompt_context
    assert "Verified result: 2.02%" in prompt_context
    assert "Verified result: 15.67%" in prompt_context
    assert "1. Microsoft: 15.67%" in prompt_context
    assert "2. Apple: 2.02%" in prompt_context
    assert "Strongest performer: Microsoft" in prompt_context
    assert "Weakest performer: Apple" in prompt_context
    assert (
        "Difference: 13.65 percentage points"
        in prompt_context
    )
    assert "Verified conclusion:" in prompt_context
    assert "Completed reasoning steps:" in prompt_context


@pytest.mark.parametrize(
    ("companies", "expected"),
    [
        (
            [
                " Apple ",
                "Microsoft",
                "apple",
                "",
                None,
                123,
            ],
            [
                "Apple",
                "Microsoft",
            ],
        ),
        (
            "Apple",
            [],
        ),
        (
            [],
            [],
        ),
    ],
)
def test_normalize_companies(
    companies: Any,
    expected: list[str],
) -> None:
    assert (
        CalculatedComparisonTool._normalize_companies(
            companies
        )
        == expected
    )


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (
            15.674,
            "percent",
            "15.67%",
        ),
        (
            10,
            "million",
            "10.00 million",
        ),
        (
            2.5,
            "",
            "2.50",
        ),
        (
            None,
            "percent",
            "Unavailable",
        ),
        (
            "15.67",
            "percent",
            "Unavailable",
        ),
    ],
)
def test_format_result_value(
    value: Any,
    unit: Any,
    expected: str,
) -> None:
    assert (
        CalculatedComparisonTool._format_result_value(
            value=value,
            unit=unit,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (
            "revenue_growth",
            "Revenue Growth",
        ),
        (
            "net_income",
            "Net Income",
        ),
        (
            "",
            "Financial Metric",
        ),
        (
            None,
            "Financial Metric",
        ),
        (
            123,
            "Financial Metric",
        ),
    ],
)
def test_format_metric_name(
    metric: Any,
    expected: str,
) -> None:
    assert (
        CalculatedComparisonTool._format_metric_name(
            metric
        )
        == expected
    )


def test_failure_response_schema() -> None:
    tool = CalculatedComparisonTool()

    result = tool._failure(
        message="Test failure.",
        decomposition={
            "calculated_comparison": True,
        },
        calculation={
            "success": False,
        },
        reasoning={
            "success": False,
        },
    )

    assert result == {
        "success": False,
        "tool": "calculated_comparison",
        "decomposition": {
            "calculated_comparison": True,
        },
        "calculation": {
            "success": False,
        },
        "reasoning": {
            "success": False,
        },
        "prompt_context": (
            "Verified calculated comparison was unavailable.\n"
            "Reason: Test failure."
        ),
        "error": "Test failure.",
    }