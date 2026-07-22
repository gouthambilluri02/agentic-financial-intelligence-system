from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.app.services.financial_calculation_service import (
    FinancialCalculationService,
)
from backend.app.tools.financial_calculator import (
    FinancialCalculatorError,
)


@pytest.fixture
def service() -> FinancialCalculationService:
    calculation_service = FinancialCalculationService()
    calculation_service.value_extractor = MagicMock()
    calculation_service.calculator = MagicMock()
    return calculation_service


@pytest.fixture
def apple_year_values() -> list[dict[str, Any]]:
    return [
        {
            "company": "Apple",
            "metric": "revenue",
            "year": 2024,
            "value": 391035.0,
            "source_file": "apple_2024_10k.pdf",
            "page": 37,
        },
        {
            "company": "Apple",
            "metric": "revenue",
            "year": 2023,
            "value": 383285.0,
            "source_file": "apple_2024_10k.pdf",
            "page": 37,
        },
    ]


@pytest.fixture
def microsoft_year_values() -> list[dict[str, Any]]:
    return [
        {
            "company": "Microsoft",
            "metric": "revenue",
            "year": 2024,
            "value": 245122.0,
            "source_file": "microsoft_2024_10k.pdf",
            "page": 84,
        },
        {
            "company": "Microsoft",
            "metric": "revenue",
            "year": 2023,
            "value": 211915.0,
            "source_file": "microsoft_2024_10k.pdf",
            "page": 84,
        },
    ]


@pytest.fixture
def retrieved_chunks() -> list[dict[str, Any]]:
    return [
        {
            "content": (
                "Year Ended September 28, 2024 2023 "
                "Total net sales 391,035 383,285"
            ),
            "metadata": {
                "company": "Apple",
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
        }
    ]


class TestInitialization:
    def test_initializes_value_extractor(self) -> None:
        service = FinancialCalculationService()

        assert service.value_extractor is not None

    def test_initializes_financial_calculator(self) -> None:
        service = FinancialCalculationService()

        assert service.calculator is not None


class TestCalculate:
    def test_returns_failure_when_metric_is_none(
        self,
        service: FinancialCalculationService,
    ) -> None:
        result = service.calculate(
            question="Calculate revenue growth.",
            metric=None,
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result == {
            "success": False,
            "calculation_type": None,
            "metric": None,
            "results": [],
            "error": "No supported financial metric was detected.",
        }

    def test_returns_failure_when_metric_is_empty(
        self,
        service: FinancialCalculationService,
    ) -> None:
        result = service.calculate(
            question="Calculate revenue growth.",
            metric="",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["error"] == (
            "No supported financial metric was detected."
        )

    def test_returns_failure_when_companies_are_empty(
        self,
        service: FinancialCalculationService,
    ) -> None:
        result = service.calculate(
            question="Calculate revenue growth.",
            metric="revenue",
            companies=[],
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["error"] == (
            "No company was detected for the calculation."
        )

    def test_returns_failure_when_companies_are_none(
        self,
        service: FinancialCalculationService,
    ) -> None:
        result = service.calculate(
            question="Calculate revenue growth.",
            metric="revenue",
            companies=None,  # type: ignore[arg-type]
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["error"] == (
            "No company was detected for the calculation."
        )

    def test_routes_growth_question_to_calculate_growth(
        self,
        service: FinancialCalculationService,
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        service._calculate_growth = MagicMock(
            return_value={
                "success": True,
            }
        )

        result = service.calculate(
            question="Calculate Apple's revenue growth.",
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=retrieved_chunks,
        )

        assert result == {
            "success": True,
        }

        service._calculate_growth.assert_called_once_with(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=retrieved_chunks,
        )

    def test_question_is_normalized_before_growth_detection(
        self,
        service: FinancialCalculationService,
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        service._calculate_growth = MagicMock(
            return_value={
                "success": True,
            }
        )

        result = service.calculate(
            question="   CALCULATE REVENUE GROWTH   ",
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=retrieved_chunks,
        )

        assert result["success"] is True
        service._calculate_growth.assert_called_once()

    def test_returns_failure_for_unsupported_calculation(
        self,
        service: FinancialCalculationService,
    ) -> None:
        result = service.calculate(
            question="What was Apple's revenue?",
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result == {
            "success": False,
            "calculation_type": None,
            "metric": None,
            "results": [],
            "error": (
                "The selected financial calculation "
                "is not connected yet."
            ),
        }

    @pytest.mark.parametrize(
        "question",
        [
            "Calculate revenue growth.",
            "Calculate the growth rate.",
            "Show year over year revenue.",
            "Show year-over-year revenue.",
            "Show yoy revenue.",
            "Calculate percentage change.",
            "Calculate percent change.",
            "Calculate the increase from 2023.",
            "Calculate the decrease from 2023.",
        ],
    )
    def test_supported_growth_questions_are_routed(
        self,
        service: FinancialCalculationService,
        question: str,
    ) -> None:
        service._calculate_growth = MagicMock(
            return_value={
                "success": True,
            }
        )

        result = service.calculate(
            question=question,
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is True
        service._calculate_growth.assert_called_once()


class TestCalculateGrowth:
    def test_returns_failure_when_no_values_are_extracted(
        self,
        service: FinancialCalculationService,
    ) -> None:
        service.value_extractor.extract_year_values.return_value = []

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["error"] == (
            "No structured yearly values could be extracted "
            "from the retrieved report context."
        )

    def test_passes_chunks_and_metric_to_value_extractor(
        self,
        service: FinancialCalculationService,
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        service.value_extractor.extract_year_values.return_value = []

        service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=retrieved_chunks,
        )

        service.value_extractor.extract_year_values.assert_called_once_with(
            retrieved_chunks=retrieved_chunks,
            metric="revenue",
        )

    def test_skips_company_with_less_than_two_values(
        self,
        service: FinancialCalculationService,
    ) -> None:
        service.value_extractor.extract_year_values.return_value = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 391035.0,
            }
        ]

        service.value_extractor.get_latest_two_values.return_value = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 391035.0,
            }
        ]

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["results"] == []

    def test_calls_get_latest_two_values_for_company(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
    ) -> None:
        extracted_values = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 391035.0,
            }
        ]

        service.value_extractor.extract_year_values.return_value = (
            extracted_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            apple_year_values
        )
        service.calculator.calculate_growth_rate.return_value = {
            "result": 2.02,
            "unit": "percent",
            "formula": "growth formula",
        }

        service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        service.value_extractor.get_latest_two_values.assert_called_once_with(
            extracted_values=extracted_values,
            company="Apple",
        )

    def test_calculates_revenue_growth(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
    ) -> None:
        service.value_extractor.extract_year_values.return_value = (
            apple_year_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            apple_year_values
        )
        service.calculator.calculate_growth_rate.return_value = {
            "result": 2.02,
            "unit": "percent",
            "formula": (
                "((current_value - previous_value) "
                "/ previous_value) * 100"
            ),
        }

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is True
        assert result["calculation_type"] == "growth_rate"
        assert result["metric"] == "revenue"
        assert result["error"] is None
        assert len(result["results"]) == 1

    def test_calls_standard_growth_calculator(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
    ) -> None:
        service.value_extractor.extract_year_values.return_value = (
            apple_year_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            apple_year_values
        )
        service.calculator.calculate_growth_rate.return_value = {
            "result": 2.02,
            "unit": "percent",
            "formula": "growth formula",
        }

        service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        service.calculator.calculate_growth_rate.assert_called_once_with(
            current_value=391035.0,
            previous_value=383285.0,
        )

        service.calculator.calculate_eps_growth.assert_not_called()

    def test_calls_eps_growth_calculator(
        self,
        service: FinancialCalculationService,
    ) -> None:
        eps_values = [
            {
                "company": "Apple",
                "metric": "eps",
                "year": 2024,
                "value": 6.08,
                "source_file": "apple_2024_10k.pdf",
                "page": 31,
            },
            {
                "company": "Apple",
                "metric": "eps",
                "year": 2023,
                "value": 6.13,
                "source_file": "apple_2024_10k.pdf",
                "page": 31,
            },
        ]

        service.value_extractor.extract_year_values.return_value = (
            eps_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            eps_values
        )
        service.calculator.calculate_eps_growth.return_value = {
            "result": -0.82,
            "unit": "percent",
            "formula": "eps growth formula",
        }

        result = service._calculate_growth(
            metric="eps",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is True

        service.calculator.calculate_eps_growth.assert_called_once_with(
            current_eps=6.08,
            previous_eps=6.13,
        )

        service.calculator.calculate_growth_rate.assert_not_called()

    def test_builds_complete_company_result(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
    ) -> None:
        service.value_extractor.extract_year_values.return_value = (
            apple_year_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            apple_year_values
        )
        service.calculator.calculate_growth_rate.return_value = {
            "result": 2.02,
            "unit": "percent",
            "formula": "growth formula",
        }

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["results"][0] == {
            "company": "Apple",
            "metric": "revenue",
            "current_year": 2024,
            "current_value": 391035.0,
            "previous_year": 2023,
            "previous_value": 383285.0,
            "result": 2.02,
            "unit": "percent",
            "formula": "growth formula",
            "current_source": {
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
            "previous_source": {
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
        }

    def test_handles_missing_source_metadata(
        self,
        service: FinancialCalculationService,
    ) -> None:
        values = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 100.0,
            },
            {
                "company": "Apple",
                "year": 2023,
                "value": 80.0,
            },
        ]

        service.value_extractor.extract_year_values.return_value = values
        service.value_extractor.get_latest_two_values.return_value = values
        service.calculator.calculate_growth_rate.return_value = {
            "result": 25.0,
            "unit": "percent",
            "formula": "growth formula",
        }

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["results"][0]["current_source"] == {
            "source_file": None,
            "page": None,
        }
        assert result["results"][0]["previous_source"] == {
            "source_file": None,
            "page": None,
        }

    def test_skips_company_when_calculator_raises_error(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
    ) -> None:
        service.value_extractor.extract_year_values.return_value = (
            apple_year_values
        )
        service.value_extractor.get_latest_two_values.return_value = (
            apple_year_values
        )
        service.calculator.calculate_growth_rate.side_effect = (
            FinancialCalculatorError("Previous value cannot be zero.")
        )

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is False
        assert result["error"] == (
            "Two valid yearly values were not available for "
            "the requested company or companies."
        )

    def test_returns_failure_when_all_companies_are_skipped(
        self,
        service: FinancialCalculationService,
    ) -> None:
        service.value_extractor.extract_year_values.return_value = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 100.0,
            }
        ]
        service.value_extractor.get_latest_two_values.return_value = []

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple", "Microsoft"],
            retrieved_chunks=[],
        )

        assert result == {
            "success": False,
            "calculation_type": None,
            "metric": None,
            "results": [],
            "error": (
                "Two valid yearly values were not available for "
                "the requested company or companies."
            ),
        }

    def test_calculates_multiple_companies(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
        microsoft_year_values: list[dict[str, Any]],
    ) -> None:
        extracted_values = (
            apple_year_values + microsoft_year_values
        )

        service.value_extractor.extract_year_values.return_value = (
            extracted_values
        )

        service.value_extractor.get_latest_two_values.side_effect = [
            apple_year_values,
            microsoft_year_values,
        ]

        service.calculator.calculate_growth_rate.side_effect = [
            {
                "result": 2.02,
                "unit": "percent",
                "formula": "growth formula",
            },
            {
                "result": 15.67,
                "unit": "percent",
                "formula": "growth formula",
            },
        ]

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple", "Microsoft"],
            retrieved_chunks=[],
        )

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["company"] == "Apple"
        assert result["results"][1]["company"] == "Microsoft"
        assert result["results"][0]["result"] == 2.02
        assert result["results"][1]["result"] == 15.67

    def test_keeps_valid_company_when_another_company_fails(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
        microsoft_year_values: list[dict[str, Any]],
    ) -> None:
        extracted_values = (
            apple_year_values + microsoft_year_values
        )

        service.value_extractor.extract_year_values.return_value = (
            extracted_values
        )
        service.value_extractor.get_latest_two_values.side_effect = [
            apple_year_values,
            microsoft_year_values,
        ]
        service.calculator.calculate_growth_rate.side_effect = [
            {
                "result": 2.02,
                "unit": "percent",
                "formula": "growth formula",
            },
            FinancialCalculatorError("Invalid Microsoft data."),
        ]

        result = service._calculate_growth(
            metric="revenue",
            companies=["Apple", "Microsoft"],
            retrieved_chunks=[],
        )

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["company"] == "Apple"

    def test_company_order_is_preserved(
        self,
        service: FinancialCalculationService,
        apple_year_values: list[dict[str, Any]],
        microsoft_year_values: list[dict[str, Any]],
    ) -> None:
        extracted_values = (
            apple_year_values + microsoft_year_values
        )

        service.value_extractor.extract_year_values.return_value = (
            extracted_values
        )

        service.value_extractor.get_latest_two_values.side_effect = [
            microsoft_year_values,
            apple_year_values,
        ]

        service.calculator.calculate_growth_rate.side_effect = [
            {
                "result": 15.67,
                "unit": "percent",
                "formula": "growth formula",
            },
            {
                "result": 2.02,
                "unit": "percent",
                "formula": "growth formula",
            },
        ]

        result = service._calculate_growth(
            metric="revenue",
            companies=["Microsoft", "Apple"],
            retrieved_chunks=[],
        )

        assert [
            item["company"]
            for item in result["results"]
        ] == [
            "Microsoft",
            "Apple",
        ]


class TestGrowthQuestionDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "calculate revenue growth",
            "calculate the growth rate",
            "show year over year performance",
            "show year-over-year performance",
            "show yoy revenue",
            "calculate percentage change",
            "calculate percent change",
            "calculate increase from 2023",
            "calculate decrease from 2023",
        ],
    )
    def test_detects_growth_questions(
        self,
        question: str,
    ) -> None:
        assert (
            FinancialCalculationService._is_growth_question(
                question
            )
            is True
        )

    @pytest.mark.parametrize(
        "question",
        [
            "what was apple revenue",
            "show microsoft revenue",
            "compare apple and microsoft",
            "what is apple eps",
            "",
        ],
    )
    def test_rejects_non_growth_questions(
        self,
        question: str,
    ) -> None:
        assert (
            FinancialCalculationService._is_growth_question(
                question
            )
            is False
        )

    def test_growth_detection_is_case_sensitive_without_normalization(
        self,
    ) -> None:
        assert (
            FinancialCalculationService._is_growth_question(
                "Calculate Revenue GROWTH"
            )
            is False
        )

    def test_calculate_method_normalizes_case_before_detection(
        self,
        service: FinancialCalculationService,
    ) -> None:
        service._calculate_growth = MagicMock(
            return_value={
                "success": True,
            }
        )

        result = service.calculate(
            question="Calculate Revenue GROWTH",
            metric="revenue",
            companies=["Apple"],
            retrieved_chunks=[],
        )

        assert result["success"] is True


class TestFailure:
    def test_failure_response_schema(self) -> None:
        result = FinancialCalculationService._failure(
            "Calculation failed."
        )

        assert result == {
            "success": False,
            "calculation_type": None,
            "metric": None,
            "results": [],
            "error": "Calculation failed.",
        }

    def test_failure_preserves_message(self) -> None:
        result = FinancialCalculationService._failure(
            "Missing financial values."
        )

        assert result["error"] == (
            "Missing financial values."
        )


class TestBuildPromptContext:
    def test_builds_failure_prompt_context(self) -> None:
        result = FinancialCalculationService.build_prompt_context(
            {
                "success": False,
                "error": "Calculation failed.",
            }
        )

        assert result == (
            "Verified Python calculation was unavailable.\n"
            "Reason: Calculation failed."
        )

    def test_builds_failure_context_when_success_is_missing(
        self,
    ) -> None:
        result = FinancialCalculationService.build_prompt_context(
            {
                "error": "No calculation was performed.",
            }
        )

        assert result == (
            "Verified Python calculation was unavailable.\n"
            "Reason: No calculation was performed."
        )

    def test_builds_failure_context_with_missing_error(
        self,
    ) -> None:
        result = FinancialCalculationService.build_prompt_context(
            {
                "success": False,
            }
        )

        assert result == (
            "Verified Python calculation was unavailable.\n"
            "Reason: None"
        )

    def test_builds_successful_prompt_context(self) -> None:
        calculation_result = {
            "success": True,
            "calculation_type": "growth_rate",
            "metric": "revenue",
            "results": [
                {
                    "company": "Apple",
                    "metric": "revenue",
                    "current_year": 2024,
                    "current_value": 391035.0,
                    "previous_year": 2023,
                    "previous_value": 383285.0,
                    "result": 2.02,
                    "unit": "percent",
                    "formula": "growth formula",
                    "current_source": {
                        "source_file": "apple_2024_10k.pdf",
                        "page": 37,
                    },
                    "previous_source": {
                        "source_file": "apple_2024_10k.pdf",
                        "page": 37,
                    },
                }
            ],
            "error": None,
        }

        result = FinancialCalculationService.build_prompt_context(
            calculation_result
        )

        assert result == (
            "Verified deterministic Python calculation:\n"
            "\n"
            "Company: Apple\n"
            "Metric: revenue\n"
            "Current year: 2024\n"
            "Current value: 391035.0\n"
            "Previous year: 2023\n"
            "Previous value: 383285.0\n"
            "Calculated growth: 2.02%\n"
            "Formula: growth formula\n"
            "Current value source: "
            "{'source_file': 'apple_2024_10k.pdf', 'page': 37}\n"
            "Previous value source: "
            "{'source_file': 'apple_2024_10k.pdf', 'page': 37}\n"
            "\n"
            "Use these verified values exactly.\n"
            "Do not recalculate or change the result.\n"
            "Only explain the calculation clearly."
        )

    def test_builds_context_for_multiple_companies(self) -> None:
        calculation_result = {
            "success": True,
            "results": [
                {
                    "company": "Apple",
                    "metric": "revenue",
                    "current_year": 2024,
                    "current_value": 391035.0,
                    "previous_year": 2023,
                    "previous_value": 383285.0,
                    "result": 2.02,
                    "formula": "growth formula",
                    "current_source": {},
                    "previous_source": {},
                },
                {
                    "company": "Microsoft",
                    "metric": "revenue",
                    "current_year": 2024,
                    "current_value": 245122.0,
                    "previous_year": 2023,
                    "previous_value": 211915.0,
                    "result": 15.67,
                    "formula": "growth formula",
                    "current_source": {},
                    "previous_source": {},
                },
            ],
        }

        result = FinancialCalculationService.build_prompt_context(
            calculation_result
        )

        assert "Company: Apple" in result
        assert "Calculated growth: 2.02%" in result
        assert "Company: Microsoft" in result
        assert "Calculated growth: 15.67%" in result

    def test_context_ends_with_llm_instructions(self) -> None:
        result = FinancialCalculationService.build_prompt_context(
            {
                "success": True,
                "results": [],
            }
        )

        assert result.endswith(
            "Use these verified values exactly.\n"
            "Do not recalculate or change the result.\n"
            "Only explain the calculation clearly."
        )

    def test_empty_successful_results_still_build_context(
        self,
    ) -> None:
        result = FinancialCalculationService.build_prompt_context(
            {
                "success": True,
                "results": [],
            }
        )

        assert result == (
            "Verified deterministic Python calculation:\n"
            "\n"
            "Use these verified values exactly.\n"
            "Do not recalculate or change the result.\n"
            "Only explain the calculation clearly."
        )