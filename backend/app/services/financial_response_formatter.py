from __future__ import annotations

from typing import Any


class FinancialResponseFormatter:
    """
    Format verified financial results into consistent Markdown responses.

    This class is responsible only for presentation.

    It does not:
    - perform calculations
    - compare companies
    - retrieve documents
    - change verified values
    """

    def format_calculation_result(
        self,
        company: str,
        metric_title: str,
        previous_year: Any,
        previous_value: Any,
        current_year: Any,
        current_value: Any,
        formula: str,
        result_value: Any,
        result_suffix: str,
        interpretation: str,
    ) -> str:
        """
        Format one verified financial calculation.
        """

        return "\n".join(
            [
                f"## {metric_title} Growth",
                "",
                f"### {company}",
                "",
                "| Period | Value |",
                "|---|---:|",
                (
                    f"| {previous_year} | "
                    f"{self.format_number(previous_value)} |"
                ),
                (
                    f"| {current_year} | "
                    f"{self.format_number(current_value)} |"
                ),
                "",
                "### Verified Result",
                "",
                (
                    f"**{self.format_number(result_value)}"
                    f"{result_suffix}**"
                ),
                "",
                "### Formula",
                "",
                f"`{formula}`",
                "",
                "### Interpretation",
                "",
                interpretation,
            ]
        )

    def format_comparison_result(
        self,
        metric_title: str,
        results: list[dict[str, Any]],
        highest_company: str,
        highest_value: Any,
        lowest_company: str,
        lowest_value: Any,
        absolute_difference: Any,
        percentage_difference: Any | None,
        reporting_period_warning: bool,
    ) -> str:
        """
        Format a verified company comparison.
        """

        answer_parts = [
            f"## {metric_title} Comparison",
            "",
            "| Company | Fiscal Year | Value |",
            "|---|---:|---:|",
        ]

        for result in results:
            company = result.get(
                "company",
                "Unknown company",
            )
            year = result.get(
                "year",
                "Unknown year",
            )
            value = result.get(
                "value",
            )

            answer_parts.append(
                f"| {company} | {year} | "
                f"{self.format_number(value)} |"
            )

        answer_parts.extend(
            [
                "",
                "### Comparison Summary",
                "",
                (
                    f"**Higher value:** {highest_company} "
                    f"with {self.format_number(highest_value)}"
                ),
                "",
                (
                    f"**Lower value:** {lowest_company} "
                    f"with {self.format_number(lowest_value)}"
                ),
                "",
                (
                    "**Absolute difference:** "
                    f"{self.format_number(absolute_difference)}"
                ),
            ]
        )

        if percentage_difference is not None:
            answer_parts.extend(
                [
                    "",
                    (
                        "**Percentage difference:** "
                        f"{self.format_number(percentage_difference)}%"
                    ),
                ]
            )

        answer_parts.extend(
            [
                "",
                "### Conclusion",
                "",
                (
                    f"{highest_company} reported the higher "
                    f"{metric_title.lower()} value."
                ),
            ]
        )

        if reporting_period_warning:
            answer_parts.extend(
                [
                    "",
                    "> Note: The companies use different reporting "
                    "periods, so this comparison should be interpreted "
                    "with that limitation.",
                ]
            )

        return "\n".join(answer_parts)

    def format_error(
        self,
        title: str,
        reason: str,
    ) -> str:
        """
        Format a consistent error response.
        """

        return "\n".join(
            [
                f"## {title}",
                "",
                "The requested operation could not be completed.",
                "",
                f"**Reason:** {reason}",
            ]
        )

    @staticmethod
    def format_number(
        value: Any,
    ) -> str:
        """
        Format numeric values without changing their meaning.
        """

        if value is None:
            return "Unavailable"

        if isinstance(
            value,
            bool,
        ):
            return str(value)

        if isinstance(
            value,
            int,
        ):
            return f"{value:,}"

        if isinstance(
            value,
            float,
        ):
            if value.is_integer():
                return f"{int(value):,}"

            return f"{value:,.2f}"

        return str(value)


if __name__ == "__main__":
    formatter = FinancialResponseFormatter()

    calculation_output = formatter.format_calculation_result(
        company="Apple",
        metric_title="Revenue",
        previous_year=2023,
        previous_value=383285.0,
        current_year=2024,
        current_value=391035.0,
        formula=(
            "((Current Value - Previous Value) / "
            "Previous Value) × 100"
        ),
        result_value=2.02,
        result_suffix="%",
        interpretation=(
            "Apple's revenue increased modestly in 2024 "
            "compared with the previous period."
        ),
    )

    comparison_output = formatter.format_comparison_result(
        metric_title="Revenue",
        results=[
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
        ],
        highest_company="Apple",
        highest_value=391035.0,
        lowest_company="Microsoft",
        lowest_value=245122.0,
        absolute_difference=145913.0,
        percentage_difference=59.53,
        reporting_period_warning=False,
    )

    print("=" * 80)
    print("CALCULATION FORMAT\n")
    print(calculation_output)

    print("\n" + "=" * 80)
    print("COMPARISON FORMAT\n")
    print(comparison_output)