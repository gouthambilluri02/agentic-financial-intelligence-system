"""
Tests for DeterministicResponseBuilder.

Covered behavior:

- Initialization
- build_answer()
- build_calculation_answer()
- _format_calculation_result()
- build_comparison_answer()
- build_calculated_comparison_answer()
- _build_growth_interpretation()
- _build_result_suffix()
- _format_comparison_type()
- _format_metric_name()
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.app.services.deterministic_response_builder import (
    DeterministicResponseBuilder,
)
from backend.app.services.financial_response_formatter import (
    FinancialResponseFormatter,
)


@pytest.fixture
def formatter() -> MagicMock:
    """Return a mocked FinancialResponseFormatter."""

    mocked = MagicMock(spec=FinancialResponseFormatter)
    mocked.format_error.return_value = "FORMATTED ERROR"
    mocked.format_calculation_result.return_value = (
        "FORMATTED CALCULATION"
    )
    mocked.format_comparison_result.return_value = (
        "FORMATTED COMPARISON"
    )
    mocked.format_number.side_effect = lambda value: (
        "Unavailable"
        if value is None
        else str(value)
    )
    return mocked


@pytest.fixture
def builder(
    formatter: MagicMock,
) -> DeterministicResponseBuilder:
    """Return a builder using the mocked formatter."""

    return DeterministicResponseBuilder(formatter=formatter)


class TestInitialization:
    """Tests for builder initialization."""

    def test_uses_injected_formatter(
        self,
        formatter: MagicMock,
    ) -> None:
        builder = DeterministicResponseBuilder(
            formatter=formatter
        )

        assert builder.formatter is formatter

    def test_creates_default_formatter_when_none_is_provided(
        self,
    ) -> None:
        builder = DeterministicResponseBuilder()

        assert isinstance(
            builder.formatter,
            FinancialResponseFormatter,
        )


class TestBuildAnswer:
    """Tests for build_answer()."""

    def test_financial_calculator_routes_to_calculation_builder(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calculation = {"success": True}
        mocked_method = MagicMock(
            return_value="CALCULATION ANSWER"
        )

        monkeypatch.setattr(
            builder,
            "build_calculation_answer",
            mocked_method,
        )

        result = builder.build_answer(
            selected_tool="financial_calculator",
            calculation=calculation,
        )

        assert result == "CALCULATION ANSWER"
        mocked_method.assert_called_once_with(calculation)

    def test_company_comparison_routes_to_comparison_builder(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        comparison = {"success": True}
        mocked_method = MagicMock(
            return_value="COMPARISON ANSWER"
        )

        monkeypatch.setattr(
            builder,
            "build_comparison_answer",
            mocked_method,
        )

        result = builder.build_answer(
            selected_tool="company_comparison",
            comparison=comparison,
        )

        assert result == "COMPARISON ANSWER"
        mocked_method.assert_called_once_with(comparison)

    def test_calculated_comparison_routes_to_reasoning_builder(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reasoning = {"success": True}
        mocked_method = MagicMock(
            return_value="REASONING ANSWER"
        )

        monkeypatch.setattr(
            builder,
            "build_calculated_comparison_answer",
            mocked_method,
        )

        result = builder.build_answer(
            selected_tool="calculated_comparison",
            reasoning=reasoning,
        )

        assert result == "REASONING ANSWER"
        mocked_method.assert_called_once_with(reasoning)

    @pytest.mark.parametrize(
        "selected_tool",
        [
            "risk_analysis",
            "document_retrieval",
            "unknown_tool",
            "",
            " ",
            None,
            123,
            [],
            {},
        ],
    )
    def test_unsupported_tool_returns_none(
        self,
        builder: DeterministicResponseBuilder,
        selected_tool: Any,
    ) -> None:
        result = builder.build_answer(
            selected_tool=selected_tool,  # type: ignore[arg-type]
            calculation={"success": True},
            comparison={"success": True},
            reasoning={"success": True},
        )

        assert result is None

    @pytest.mark.parametrize(
        ("selected_tool", "expected"),
        [
            (" financial_calculator ", "CALC"),
            ("\tcompany_comparison\t", "COMP"),
            ("\ncalculated_comparison\n", "REASON"),
        ],
    )
    def test_selected_tool_is_trimmed(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
        selected_tool: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(
            builder,
            "build_calculation_answer",
            MagicMock(return_value="CALC"),
        )
        monkeypatch.setattr(
            builder,
            "build_comparison_answer",
            MagicMock(return_value="COMP"),
        )
        monkeypatch.setattr(
            builder,
            "build_calculated_comparison_answer",
            MagicMock(return_value="REASON"),
        )

        result = builder.build_answer(
            selected_tool=selected_tool,
        )

        assert result == expected

    def test_tool_matching_is_case_sensitive(
        self,
        builder: DeterministicResponseBuilder,
    ) -> None:
        assert builder.build_answer(
            selected_tool="FINANCIAL_CALCULATOR"
        ) is None


class TestBuildCalculationAnswer:
    """Tests for build_calculation_answer()."""

    @pytest.mark.parametrize(
        "calculation",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_non_dictionary_calculation_returns_missing_result_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        calculation: Any,
    ) -> None:
        result = builder.build_calculation_answer(
            calculation
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason=(
                "No verified calculation result was available."
            ),
        )

    def test_empty_dictionary_uses_default_unsuccessful_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_calculation_answer({})

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason=(
                "The required financial values were unavailable."
            ),
        )

    def test_unsuccessful_calculation_uses_provided_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_calculation_answer(
            {
                "success": False,
                "error": "Division by zero.",
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason="Division by zero.",
        )

    @pytest.mark.parametrize(
        ("error_value", "expected"),
        [
            (None, "None"),
            (123, "123"),
            (4.5, "4.5"),
            (True, "True"),
            (["error"], "['error']"),
            ({"message": "error"}, "{'message': 'error'}"),
        ],
    )
    def test_unsuccessful_calculation_converts_error_to_string(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        error_value: Any,
        expected: str,
    ) -> None:
        builder.build_calculation_answer(
            {
                "success": False,
                "error": error_value,
            }
        )

        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason=expected,
        )

    @pytest.mark.parametrize(
        "results",
        [
            None,
            [],
            "",
            0,
            False,
            {},
            (),
        ],
    )
    def test_invalid_or_empty_results_return_usable_result_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        results: Any,
    ) -> None:
        result = builder.build_calculation_answer(
            {
                "success": True,
                "results": results,
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason=(
                "The calculation completed without a usable "
                "verified result."
            ),
        )

    @pytest.mark.parametrize(
        "invalid_result",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_all_invalid_result_items_return_valid_company_data_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        invalid_result: Any,
    ) -> None:
        result = builder.build_calculation_answer(
            {
                "success": True,
                "results": [invalid_result],
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Financial Calculation",
            reason=(
                "The calculation results did not contain "
                "valid company data."
            ),
        )

    def test_invalid_result_items_are_skipped(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid = {"company": "Apple"}
        mocked = MagicMock(return_value="APPLE")

        monkeypatch.setattr(
            builder,
            "_format_calculation_result",
            mocked,
        )

        result = builder.build_calculation_answer(
            {
                "success": True,
                "results": [
                    None,
                    "invalid",
                    valid,
                    123,
                ],
            }
        )

        assert result == "APPLE"
        mocked.assert_called_once_with(valid)

    def test_multiple_valid_results_are_joined_with_separator(
        self,
        builder: DeterministicResponseBuilder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = {"company": "Apple"}
        second = {"company": "Microsoft"}
        mocked = MagicMock(
            side_effect=[
                "APPLE SECTION",
                "MICROSOFT SECTION",
            ]
        )

        monkeypatch.setattr(
            builder,
            "_format_calculation_result",
            mocked,
        )

        result = builder.build_calculation_answer(
            {
                "success": True,
                "results": [first, second],
            }
        )

        assert result == (
            "APPLE SECTION\n\n---\n\nMICROSOFT SECTION"
        )
        assert mocked.call_count == 2


class TestFormatCalculationResult:
    """Tests for _format_calculation_result()."""

    def test_forwards_all_explicit_values_to_formatter(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder._format_calculation_result(
            {
                "company": "Apple",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 391035.0,
                "previous_value": 383285.0,
                "result": 2.02,
                "unit": "percent",
                "formula": "CUSTOM FORMULA",
            }
        )

        assert result == "FORMATTED CALCULATION"
        formatter.format_calculation_result.assert_called_once_with(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=383285.0,
            current_year=2024,
            current_value=391035.0,
            formula="CUSTOM FORMULA",
            result_value=2.02,
            result_suffix="%",
            interpretation=(
                "Apple's revenue increased modestly in 2024 "
                "compared with the previous period."
            ),
        )

    def test_missing_values_use_defaults(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder._format_calculation_result({})

        formatter.format_calculation_result.assert_called_once_with(
            company="The company",
            metric_title="Financial Metric",
            previous_year="Previous period",
            previous_value=None,
            current_year="Current period",
            current_value=None,
            formula=(
                "((Current Value - Previous Value) / "
                "Previous Value) × 100"
            ),
            result_value=None,
            result_suffix="%",
            interpretation=(
                "The verified result describes the change in "
                "The company's financial metric for "
                "Current period."
            ),
        )

    def test_company_and_formula_are_converted_to_strings(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder._format_calculation_result(
            {
                "company": 123,
                "formula": 456,
            }
        )

        kwargs = (
            formatter.format_calculation_result.call_args.kwargs
        )
        assert kwargs["company"] == "123"
        assert kwargs["formula"] == "456"

    @pytest.mark.parametrize(
        ("unit", "expected_suffix"),
        [
            ("percent", "%"),
            ("percentage", "%"),
            ("%", "%"),
            ("USD", " USD"),
            (" ratio ", " ratio"),
            ("", ""),
            ("   ", ""),
            (None, ""),
            (123, ""),
        ],
    )
    def test_result_suffix_is_derived_from_unit(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        unit: Any,
        expected_suffix: str,
    ) -> None:
        builder._format_calculation_result(
            {
                "unit": unit,
            }
        )

        assert (
            formatter.format_calculation_result.call_args.kwargs[
                "result_suffix"
            ]
            == expected_suffix
        )


class TestBuildComparisonAnswer:
    """Tests for build_comparison_answer()."""

    @pytest.mark.parametrize(
        "comparison",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_non_dictionary_comparison_returns_missing_result_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        comparison: Any,
    ) -> None:
        result = builder.build_comparison_answer(comparison)

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Company Comparison",
            reason=(
                "No verified comparison result was available."
            ),
        )

    def test_empty_dictionary_uses_default_unsuccessful_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder.build_comparison_answer({})

        formatter.format_error.assert_called_once_with(
            title="Company Comparison",
            reason=(
                "The required company values were unavailable."
            ),
        )

    def test_unsuccessful_comparison_uses_provided_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_comparison_answer(
            {
                "success": False,
                "error": "Missing company data.",
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Company Comparison",
            reason="Missing company data.",
        )

    @pytest.mark.parametrize(
        "results",
        [
            None,
            [],
            [{}],
            [{"company": "Apple"}],
            "",
            {},
            (),
        ],
    )
    def test_invalid_or_too_short_results_return_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        results: Any,
    ) -> None:
        result = builder.build_comparison_answer(
            {
                "success": True,
                "results": results,
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Company Comparison",
            reason=(
                "Verified values for at least two companies "
                "are required."
            ),
        )

    def test_less_than_two_valid_dictionaries_returns_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_comparison_answer(
            {
                "success": True,
                "results": [
                    {"company": "Apple"},
                    "invalid",
                    None,
                ],
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Company Comparison",
            reason=(
                "The comparison results did not contain "
                "valid data for at least two companies."
            ),
        )

    def test_invalid_items_are_removed_before_formatting(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        apple = {
            "company": "Apple",
            "year": 2024,
        }
        microsoft = {
            "company": "Microsoft",
            "year": 2024,
        }

        builder.build_comparison_answer(
            {
                "success": True,
                "results": [
                    apple,
                    "invalid",
                    microsoft,
                    None,
                ],
            }
        )

        kwargs = (
            formatter.format_comparison_result.call_args.kwargs
        )
        assert kwargs["results"] == [apple, microsoft]

    def test_valid_comparison_forwards_all_values(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        results = [
            {
                "company": "Apple",
                "year": 2024,
                "value": 391035.0,
            },
            {
                "company": "Microsoft",
                "year": 2024,
                "value": 245122.0,
            },
        ]

        result = builder.build_comparison_answer(
            {
                "success": True,
                "metric": "revenue",
                "results": results,
                "highest_company": "Apple",
                "highest_value": 391035.0,
                "lowest_company": "Microsoft",
                "lowest_value": 245122.0,
                "absolute_difference": 145913.0,
                "percentage_difference": 59.53,
            }
        )

        assert result == "FORMATTED COMPARISON"
        formatter.format_comparison_result.assert_called_once_with(
            metric_title="Revenue",
            results=results,
            highest_company="Apple",
            highest_value=391035.0,
            lowest_company="Microsoft",
            lowest_value=245122.0,
            absolute_difference=145913.0,
            percentage_difference=59.53,
            reporting_period_warning=False,
        )

    def test_different_years_enable_reporting_period_warning(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder.build_comparison_answer(
            {
                "success": True,
                "results": [
                    {
                        "company": "Apple",
                        "year": 2024,
                    },
                    {
                        "company": "Microsoft",
                        "year": 2023,
                    },
                ],
            }
        )

        assert (
            formatter.format_comparison_result.call_args.kwargs[
                "reporting_period_warning"
            ]
            is True
        )

    def test_years_are_compared_after_string_conversion(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder.build_comparison_answer(
            {
                "success": True,
                "results": [
                    {
                        "company": "Apple",
                        "year": 2024,
                    },
                    {
                        "company": "Microsoft",
                        "year": "2024",
                    },
                ],
            }
        )

        assert (
            formatter.format_comparison_result.call_args.kwargs[
                "reporting_period_warning"
            ]
            is False
        )

    def test_highest_and_lowest_company_are_stringified(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder.build_comparison_answer(
            {
                "success": True,
                "results": [{}, {}],
                "highest_company": 123,
                "lowest_company": 456,
            }
        )

        kwargs = (
            formatter.format_comparison_result.call_args.kwargs
        )
        assert kwargs["highest_company"] == "123"
        assert kwargs["lowest_company"] == "456"


class TestBuildCalculatedComparisonAnswer:
    """Tests for build_calculated_comparison_answer()."""

    @pytest.mark.parametrize(
        "reasoning",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_non_dictionary_reasoning_returns_missing_result_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        reasoning: Any,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            reasoning
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Calculated Comparison",
            reason=(
                "No verified calculated comparison result "
                "was available."
            ),
        )

    def test_empty_dictionary_uses_default_unsuccessful_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        builder.build_calculated_comparison_answer({})

        formatter.format_error.assert_called_once_with(
            title="Calculated Comparison",
            reason=(
                "The calculated comparison could not "
                "be completed."
            ),
        )

    def test_unsuccessful_reasoning_uses_provided_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": False,
                "error": "Reasoning failed.",
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Calculated Comparison",
            reason="Reasoning failed.",
        )

    @pytest.mark.parametrize(
        "ranked_results",
        [
            None,
            [],
            [{}],
            "",
            {},
            (),
        ],
    )
    def test_invalid_or_too_short_ranked_results_return_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
        ranked_results: Any,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "ranked_results": ranked_results,
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Calculated Comparison",
            reason=(
                "Verified calculations for at least two "
                "companies are required."
            ),
        )

    def test_less_than_two_valid_ranked_results_returns_error(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "ranked_results": [
                    {"company": "Apple"},
                    "invalid",
                    None,
                ],
            }
        )

        assert result == "FORMATTED ERROR"
        formatter.format_error.assert_called_once_with(
            title="Calculated Comparison",
            reason=(
                "The reasoning result did not contain valid "
                "data for at least two companies."
            ),
        )

    def test_builds_ranked_markdown_answer(
        self,
        builder: DeterministicResponseBuilder,
        formatter: MagicMock,
    ) -> None:
        formatter.format_number.side_effect = lambda value: str(value)

        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "comparison_type": "growth_rate",
                "metric": "revenue",
                "ranked_results": [
                    {
                        "company": "Microsoft",
                        "previous_year": 2023,
                        "current_year": 2024,
                        "result": 15.67,
                        "unit": "percent",
                    },
                    {
                        "company": "Apple",
                        "previous_year": 2023,
                        "current_year": 2024,
                        "result": 2.02,
                        "unit": "percent",
                    },
                ],
                "strongest_company": "Microsoft",
                "strongest_result": 15.67,
                "weakest_company": "Apple",
                "weakest_result": 2.02,
                "difference_percentage_points": 13.65,
                "conclusion": "Microsoft was stronger.",
            }
        )

        assert result.startswith(
            "## Revenue Growth Comparison"
        )
        assert (
            "| 1 | Microsoft | 2023 | 2024 | 15.67% |"
            in result
        )
        assert (
            "| 2 | Apple | 2023 | 2024 | 2.02% |"
            in result
        )
        assert (
            "**Strongest performer:** Microsoft with 15.67%"
            in result
        )
        assert (
            "**Weakest performer:** Apple with 2.02%"
            in result
        )
        assert (
            "**Difference:** 13.65 percentage points"
            in result
        )
        assert result.endswith(
            "Microsoft was stronger."
        )

    def test_invalid_ranked_items_are_removed(
        self,
        builder: DeterministicResponseBuilder,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "company": "Apple",
                        "result": 10,
                    },
                    "invalid",
                    {
                        "company": "Microsoft",
                        "result": 5,
                    },
                ],
            }
        )

        assert "| 1 | Apple |" in result
        assert "| 2 | Microsoft |" in result
        assert "invalid" not in result

    def test_difference_is_omitted_when_none(
        self,
        builder: DeterministicResponseBuilder,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "ranked_results": [{}, {}],
                "difference_percentage_points": None,
            }
        )

        assert "**Difference:**" not in result

    def test_default_conclusion_is_used_when_missing(
        self,
        builder: DeterministicResponseBuilder,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "metric": "revenue",
                "ranked_results": [{}, {}],
                "strongest_company": "Apple",
            }
        )

        assert result.endswith(
            "Apple produced the strongest verified revenue result."
        )

    def test_default_values_are_rendered(
        self,
        builder: DeterministicResponseBuilder,
    ) -> None:
        result = builder.build_calculated_comparison_answer(
            {
                "success": True,
                "ranked_results": [{}, {}],
            }
        ) 

        assert (
            "## Financial Metric Calculated Metric Comparison"
            in result
        )
        assert "Unknown company" in result
        assert "Previous period" in result
        assert "Current period" in result


class TestBuildGrowthInterpretation:
    """Tests for _build_growth_interpretation()."""

    @pytest.mark.parametrize(
        "result",
        [
            None,
            "2.5",
            [],
            {},
            (),
        ],
    )
    def test_non_numeric_result_uses_generic_interpretation(
        self,
        result: Any,
    ) -> None:
        interpretation = (
            DeterministicResponseBuilder
            ._build_growth_interpretation(
                company="Apple",
                metric_title="Revenue",
                result=result,
                current_year=2024,
            )
        )

        assert interpretation == (
            "The verified result describes the change in "
            "Apple's revenue for 2024."
        )

    @pytest.mark.parametrize(
        ("result", "strength"),
        [
            (0.1, "modestly"),
            (2.99, "modestly"),
            (3, "moderately"),
            (9.99, "moderately"),
            (10, "significantly"),
            (25, "significantly"),
        ],
    )
    def test_positive_result_interpretation(
        self,
        result: float,
        strength: str,
    ) -> None:
        interpretation = (
            DeterministicResponseBuilder
            ._build_growth_interpretation(
                company="Apple",
                metric_title="Revenue",
                result=result,
                current_year=2024,
            )
        )

        assert interpretation == (
            f"Apple's revenue increased {strength} in 2024 "
            "compared with the previous period."
        )

    @pytest.mark.parametrize(
        ("result", "strength"),
        [
            (-0.1, "modestly"),
            (-2.99, "modestly"),
            (-3, "moderately"),
            (-9.99, "moderately"),
            (-10, "significantly"),
            (-25, "significantly"),
        ],
    )
    def test_negative_result_interpretation(
        self,
        result: float,
        strength: str,
    ) -> None:
        interpretation = (
            DeterministicResponseBuilder
            ._build_growth_interpretation(
                company="Apple",
                metric_title="Revenue",
                result=result,
                current_year=2024,
            )
        )

        assert interpretation == (
            f"Apple's revenue decreased {strength} in 2024 "
            "compared with the previous period."
        )

    def test_zero_result_interpretation(self) -> None:
        interpretation = (
            DeterministicResponseBuilder
            ._build_growth_interpretation(
                company="Apple",
                metric_title="Revenue",
                result=0,
                current_year=2024,
            )
        )

        assert interpretation == (
            "Apple's revenue remained unchanged in 2024 "
            "compared with the previous period."
        )


class TestBuildResultSuffix:
    """Tests for _build_result_suffix()."""

    @pytest.mark.parametrize(
        "unit",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
        ],
    )
    def test_non_string_unit_returns_empty_suffix(
        self,
        unit: Any,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._build_result_suffix(unit)
            == ""
        )

    @pytest.mark.parametrize(
        "unit",
        [
            "percent",
            "percentage",
            "%",
            " Percent ",
            " PERCENTAGE ",
        ],
    )
    def test_percentage_units_return_percent_sign(
        self,
        unit: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._build_result_suffix(unit)
            == "%"
        )

    @pytest.mark.parametrize(
        "unit",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
        ],
    )
    def test_blank_unit_returns_empty_suffix(
        self,
        unit: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._build_result_suffix(unit)
            == ""
        )

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [
            ("USD", " USD"),
            ("ratio", " ratio"),
            (" points ", " points"),
            ("million", " million"),
        ],
    )
    def test_other_units_are_trimmed_and_prefixed_with_space(
        self,
        unit: str,
        expected: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._build_result_suffix(unit)
            == expected
        )


class TestFormatComparisonType:
    """Tests for _format_comparison_type()."""

    @pytest.mark.parametrize(
        "comparison_type",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
            "",
            " ",
            "   ",
        ],
    )
    def test_invalid_or_blank_type_returns_calculated(
        self,
        comparison_type: Any,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_comparison_type(comparison_type)
            == "Calculated"
        )

    @pytest.mark.parametrize(
        ("comparison_type", "expected"),
        [
            ("growth_rate", "Growth"),
            ("percentage_change", "Percentage Change"),
            ("profit_margin", "Profit Margin"),
            ("gross_margin", "Gross Margin"),
            ("operating_margin", "Operating Margin"),
            ("net_profit_margin", "Net Profit Margin"),
            ("current_ratio", "Current Ratio"),
            ("debt_ratio", "Debt Ratio"),
            ("return_on_equity", "Return on Equity"),
            ("return_on_assets", "Return on Assets"),
            ("calculated_metric", "Calculated"),
        ],
    )
    def test_known_comparison_types(
        self,
        comparison_type: str,
        expected: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_comparison_type(comparison_type)
            == expected
        )

    def test_comparison_type_is_trimmed(self) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_comparison_type("  growth_rate  ")
            == "Growth"
        )

    @pytest.mark.parametrize(
        ("comparison_type", "expected"),
        [
            ("custom_metric", "Custom Metric"),
            ("return_on_investment", "Return On Investment"),
            ("debt", "Debt"),
        ],
    )
    def test_unknown_comparison_type_is_humanized(
        self,
        comparison_type: str,
        expected: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_comparison_type(comparison_type)
            == expected
        )


class TestFormatMetricName:
    """Tests for _format_metric_name()."""

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("revenue", "Revenue"),
            ("net_income", "Net Income"),
            ("operating_income", "Operating Income"),
            ("eps", "EPS"),
            ("cash_flow", "Cash Flow"),
            ("free_cash_flow", "Free Cash Flow"),
            ("gross_margin", "Gross Margin"),
            ("operating_margin", "Operating Margin"),
            ("net_profit_margin", "Net Profit Margin"),
            ("current_ratio", "Current Ratio"),
            ("debt_ratio", "Debt Ratio"),
            ("return_on_equity", "Return on Equity"),
            ("roe", "Return on Equity"),
            ("return_on_assets", "Return on Assets"),
            ("roa", "Return on Assets"),
        ],
    )
    def test_known_metric_names(
        self,
        metric: str,
        expected: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_metric_name(metric)
            == expected
        )

    def test_metric_is_trimmed(self) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_metric_name("  revenue  ")
            == "Revenue"
        )

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("custom_metric", "Custom Metric"),
            ("return_on_investment", "Return On Investment"),
            ("debt", "Debt"),
            ("total_revenue_growth", "Total Revenue Growth"),
            ("", ""),
        ],
    )
    def test_unknown_metric_names_are_humanized(
        self,
        metric: str,
        expected: str,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_metric_name(metric)
            == expected
        )

    @pytest.mark.parametrize(
        "metric",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
        ],
    )
    def test_non_string_metric_returns_generic_name(
        self,
        metric: Any,
    ) -> None:
        assert (
            DeterministicResponseBuilder
            ._format_metric_name(metric)
            == "Financial Metric"
        )
