"""
Tests for FinancialResponseFormatter.

Covered methods:
- format_number()
- format_calculation_result()
- format_comparison_result()
- format_error()
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.financial_response_formatter import (
    FinancialResponseFormatter,
)


@pytest.fixture
def formatter() -> FinancialResponseFormatter:
    return FinancialResponseFormatter()


class TestFormatNumber:
    def test_none_returns_unavailable(self) -> None:
        assert FinancialResponseFormatter.format_number(None) == "Unavailable"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "True"), (False, "False")],
    )
    def test_boolean_values_are_preserved(self, value: bool, expected: str) -> None:
        assert FinancialResponseFormatter.format_number(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (-1, "-1"),
            (1000, "1,000"),
            (1000000, "1,000,000"),
            (-2500000, "-2,500,000"),
        ],
    )
    def test_integer_values_use_thousands_separators(
        self,
        value: int,
        expected: str,
    ) -> None:
        assert FinancialResponseFormatter.format_number(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "0"),
            (1.0, "1"),
            (-1.0, "-1"),
            (1000.0, "1,000"),
            (1000000.0, "1,000,000"),
            (-2500000.0, "-2,500,000"),
        ],
    )
    def test_whole_number_floats_render_as_integers(
        self,
        value: float,
        expected: str,
    ) -> None:
        assert FinancialResponseFormatter.format_number(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.2, "1.20"),
            (1.234, "1.23"),
            (1.235, "1.24"),
            (-1.234, "-1.23"),
            (1234.567, "1,234.57"),
            (0.004, "0.00"),
        ],
    )
    def test_decimal_floats_render_with_two_decimal_places(
        self,
        value: float,
        expected: str,
    ) -> None:
        assert FinancialResponseFormatter.format_number(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["1234", "Revenue", "", [1, 2], {"value": 10}, (1, 2)],
    )
    def test_non_numeric_values_use_string_conversion(self, value: Any) -> None:
        assert FinancialResponseFormatter.format_number(value) == str(value)


class TestFormatCalculationResult:
    def test_builds_expected_calculation_markdown(
        self,
        formatter: FinancialResponseFormatter,
    ) -> None:
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=383285.0,
            current_year=2024,
            current_value=391035.0,
            formula="((Current Value - Previous Value) / Previous Value) × 100",
            result_value=2.02,
            result_suffix="%",
            interpretation="Apple's revenue increased modestly in 2024.",
        )

        assert result == (
            "## Revenue Growth\n\n"
            "### Apple\n\n"
            "| Period | Value |\n"
            "|---|---:|\n"
            "| 2023 | 383,285 |\n"
            "| 2024 | 391,035 |\n\n"
            "### Verified Result\n\n"
            "**2.02%**\n\n"
            "### Formula\n\n"
            "`((Current Value - Previous Value) / Previous Value) × 100`\n\n"
            "### Interpretation\n\n"
            "Apple's revenue increased modestly in 2024."
        )

    def test_title_contains_metric_name(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_calculation_result(
            company="Microsoft",
            metric_title="Operating Margin",
            previous_year=2023,
            previous_value=40,
            current_year=2024,
            current_value=42,
            formula="Current - Previous",
            result_value=2,
            result_suffix=" pts",
            interpretation="Margin improved.",
        )
        assert result.startswith("## Operating Margin Growth")

    def test_company_heading_is_included(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_calculation_result(
            company="Microsoft",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=100,
            current_year=2024,
            current_value=120,
            formula="(120 - 100) / 100",
            result_value=20,
            result_suffix="%",
            interpretation="Revenue increased.",
        )
        assert "### Microsoft" in result

    def test_previous_and_current_values_are_formatted(
        self,
        formatter: FinancialResponseFormatter,
    ) -> None:
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=1000000,
            current_year=2024,
            current_value=1250000.5,
            formula="formula",
            result_value=25.05,
            result_suffix="%",
            interpretation="Revenue increased.",
        )
        assert "| 2023 | 1,000,000 |" in result
        assert "| 2024 | 1,250,000.50 |" in result

    def test_none_values_render_as_unavailable(
        self,
        formatter: FinancialResponseFormatter,
    ) -> None:
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=None,
            current_year=2024,
            current_value=None,
            formula="formula",
            result_value=None,
            result_suffix="%",
            interpretation="Insufficient data.",
        )
        assert "| 2023 | Unavailable |" in result
        assert "| 2024 | Unavailable |" in result
        assert "**Unavailable%**" in result

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("%", "**12.50%**"),
            (" pts", "**12.50 pts**"),
            ("x", "**12.50x**"),
            ("", "**12.50**"),
        ],
    )
    def test_result_suffix_is_appended_directly(
        self,
        formatter: FinancialResponseFormatter,
        suffix: str,
        expected: str,
    ) -> None:
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=100,
            current_year=2024,
            current_value=112.5,
            formula="formula",
            result_value=12.5,
            result_suffix=suffix,
            interpretation="Interpretation.",
        )
        assert expected in result

    def test_formula_is_wrapped_in_inline_code(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=100,
            current_year=2024,
            current_value=110,
            formula="((110 - 100) / 100) × 100",
            result_value=10,
            result_suffix="%",
            interpretation="Revenue increased.",
        )
        assert "`((110 - 100) / 100) × 100`" in result

    def test_interpretation_is_preserved_verbatim(
        self,
        formatter: FinancialResponseFormatter,
    ) -> None:
        interpretation = "Revenue increased, but interpret with caution."
        result = formatter.format_calculation_result(
            company="Apple",
            metric_title="Revenue",
            previous_year=2023,
            previous_value=100,
            current_year=2024,
            current_value=110,
            formula="formula",
            result_value=10,
            result_suffix="%",
            interpretation=interpretation,
        )
        assert result.endswith(interpretation)


class TestFormatComparisonResult:
    def test_builds_expected_comparison_markdown(
        self,
        formatter: FinancialResponseFormatter,
    ) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[
                {"company": "Apple", "year": 2024, "value": 391035.0},
                {"company": "Microsoft", "year": 2024, "value": 245122.0},
            ],
            highest_company="Apple",
            highest_value=391035.0,
            lowest_company="Microsoft",
            lowest_value=245122.0,
            absolute_difference=145913.0,
            percentage_difference=59.53,
            reporting_period_warning=False,
        )
        assert result == (
            "## Revenue Comparison\n\n"
            "| Company | Fiscal Year | Value |\n"
            "|---|---:|---:|\n"
            "| Apple | 2024 | 391,035 |\n"
            "| Microsoft | 2024 | 245,122 |\n\n"
            "### Comparison Summary\n\n"
            "**Higher value:** Apple with 391,035\n\n"
            "**Lower value:** Microsoft with 245,122\n\n"
            "**Absolute difference:** 145,913\n\n"
            "**Percentage difference:** 59.53%\n\n"
            "### Conclusion\n\n"
            "Apple reported the higher revenue value."
        )

    def test_empty_results_still_builds_summary(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=11.11,
            reporting_period_warning=False,
        )
        assert "| Company | Fiscal Year | Value |" in result
        assert "| Apple |" not in result
        assert "**Higher value:** Apple with 100" in result

    def test_missing_result_fields_use_defaults(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[{}],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert "| Unknown company | Unknown year | Unavailable |" in result

    @pytest.mark.parametrize(
        ("result_row", "expected"),
        [
            ({"company": "Apple", "year": 2024, "value": 1000}, "| Apple | 2024 | 1,000 |"),
            ({"company": "Microsoft", "year": "FY2024", "value": 1234.56}, "| Microsoft | FY2024 | 1,234.56 |"),
            ({"company": "Nvidia", "year": None, "value": None}, "| Nvidia | None | Unavailable |"),
        ],
    )
    def test_result_rows_are_formatted(
        self,
        formatter: FinancialResponseFormatter,
        result_row: dict[str, Any],
        expected: str,
    ) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[result_row],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert expected in result

    def test_percentage_difference_is_omitted_when_none(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert "**Percentage difference:**" not in result

    @pytest.mark.parametrize(
        ("percentage_difference", "expected"),
        [
            (0, "**Percentage difference:** 0%"),
            (10, "**Percentage difference:** 10%"),
            (10.5, "**Percentage difference:** 10.50%"),
            (1234.567, "**Percentage difference:** 1,234.57%"),
        ],
    )
    def test_percentage_difference_is_formatted_when_present(
        self,
        formatter: FinancialResponseFormatter,
        percentage_difference: Any,
        expected: str,
    ) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=percentage_difference,
            reporting_period_warning=False,
        )
        assert expected in result

    def test_reporting_period_warning_is_added_when_true(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=True,
        )
        assert (
            "> Note: The companies use different reporting periods, so this comparison should be interpreted with that limitation."
            in result
        )

    def test_reporting_period_warning_is_omitted_when_false(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=100,
            lowest_company="Microsoft",
            lowest_value=90,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert "> Note:" not in result

    def test_conclusion_uses_lowercase_metric_title(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Operating Margin",
            results=[],
            highest_company="Apple",
            highest_value=50,
            lowest_company="Microsoft",
            lowest_value=40,
            absolute_difference=10,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert "Apple reported the higher operating margin value." in result

    def test_summary_values_are_formatted(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_comparison_result(
            metric_title="Revenue",
            results=[],
            highest_company="Apple",
            highest_value=1000000,
            lowest_company="Microsoft",
            lowest_value=500000.5,
            absolute_difference=499999.5,
            percentage_difference=None,
            reporting_period_warning=False,
        )
        assert "**Higher value:** Apple with 1,000,000" in result
        assert "**Lower value:** Microsoft with 500,000.50" in result
        assert "**Absolute difference:** 499,999.50" in result


class TestFormatError:
    def test_builds_expected_error_markdown(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_error(
            title="Calculation Error",
            reason="Required financial values were unavailable.",
        )
        assert result == (
            "## Calculation Error\n\n"
            "The requested operation could not be completed.\n\n"
            "**Reason:** Required financial values were unavailable."
        )

    @pytest.mark.parametrize(
        "title",
        ["Calculation Error", "Comparison Error", "Financial Data Error", ""],
    )
    def test_error_title_is_preserved(
        self,
        formatter: FinancialResponseFormatter,
        title: str,
    ) -> None:
        result = formatter.format_error(title=title, reason="Reason.")
        assert result.startswith(f"## {title}")

    @pytest.mark.parametrize(
        "reason",
        ["Missing data.", "Division by zero.", "Unsupported metric.", ""],
    )
    def test_error_reason_is_preserved(
        self,
        formatter: FinancialResponseFormatter,
        reason: str,
    ) -> None:
        result = formatter.format_error(title="Error", reason=reason)
        assert result.endswith(f"**Reason:** {reason}")

    def test_error_contains_standard_message(self, formatter: FinancialResponseFormatter) -> None:
        result = formatter.format_error(title="Error", reason="Failure.")
        assert "The requested operation could not be completed." in result
