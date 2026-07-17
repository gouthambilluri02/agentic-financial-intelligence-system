from typing import Any

from backend.app.services.financial_value_extractor import (
    FinancialValueExtractor,
)
from backend.app.tools.base_tool import BaseTool


class CompanyComparisonTool(BaseTool):
    """
    Compare a reported financial metric across multiple companies.

    This tool uses structured values extracted from retrieved
    financial-report chunks. Python performs the comparison.
    """

    name = "company_comparison"

    description = (
        "Compares financial metrics such as revenue, net income, "
        "operating income, or EPS across two or more companies "
        "using verified values extracted from company reports."
    )

    def __init__(self) -> None:
        self.value_extractor = FinancialValueExtractor()

    def run(
        self,
        **kwargs: Any,
    ) -> dict:
        """
        Execute a company comparison.

        Required arguments:
        - question
        - metric
        - companies
        - retrieved_chunks
        """

        question = kwargs.get("question")
        metric = kwargs.get("metric")
        companies = kwargs.get(
            "companies",
            [],
        )
        retrieved_chunks = kwargs.get(
            "retrieved_chunks",
            [],
        )

        validation_error = self._validate_inputs(
            question=question,
            metric=metric,
            companies=companies,
            retrieved_chunks=retrieved_chunks,
        )

        if validation_error:
            return self._failure(
                validation_error
            )

        extracted_values = (
            self.value_extractor.extract_year_values(
                retrieved_chunks=retrieved_chunks,
                metric=metric,
            )
        )

        if not extracted_values:
            return self._failure(
                "No structured values could be extracted for "
                f"the requested metric '{metric}'."
            )

        company_results = self._build_company_results(
            companies=companies,
            metric=metric,
            extracted_values=extracted_values,
        )

        if len(company_results) < 2:
            available_companies = [
                result["company"]
                for result in company_results
            ]

            return self._failure(
                "The comparison requires valid values for at least "
                "two companies. Values were available for: "
                f"{available_companies or 'none'}."
            )

        ranked_results = sorted(
            company_results,
            key=lambda item: item["value"],
            reverse=True,
        )

        highest_result = ranked_results[0]
        lowest_result = ranked_results[-1]

        difference = (
            highest_result["value"]
            - lowest_result["value"]
        )

        percentage_difference = (
            self._calculate_percentage_difference(
                higher_value=highest_result["value"],
                lower_value=lowest_result["value"],
            )
        )

        comparison_result = {
            "success": True,
            "tool": self.name,
            "metric": metric,
            "results": ranked_results,
            "highest_company": highest_result["company"],
            "highest_value": highest_result["value"],
            "lowest_company": lowest_result["company"],
            "lowest_value": lowest_result["value"],
            "absolute_difference": round(
                difference,
                2,
            ),
            "percentage_difference": (
                percentage_difference
            ),
            "error": None,
        }

        comparison_result["prompt_context"] = (
            self.build_prompt_context(
                comparison_result
            )
        )

        return comparison_result

    def _build_company_results(
        self,
        companies: list[str],
        metric: str,
        extracted_values: list[dict],
    ) -> list[dict]:
        """
        Select the latest available metric value for each company.
        """

        company_results: list[dict] = []

        for company in companies:
            latest_values = (
                self.value_extractor.get_latest_two_values(
                    extracted_values=extracted_values,
                    company=company,
                )
            )

            if not latest_values:
                continue

            latest_entry = latest_values[0]

            company_results.append(
                {
                    "company": company,
                    "metric": metric,
                    "year": latest_entry["year"],
                    "value": latest_entry["value"],
                    "source_file": latest_entry.get(
                        "source_file"
                    ),
                    "page": latest_entry.get(
                        "page"
                    ),
                }
            )

        return company_results

    @staticmethod
    def _calculate_percentage_difference(
        higher_value: float,
        lower_value: float,
    ) -> float | None:
        """
        Calculate how much higher one value is than another.

        Formula:
        ((higher - lower) / lower) * 100
        """

        if lower_value == 0:
            return None

        result = (
            (higher_value - lower_value)
            / abs(lower_value)
        ) * 100

        return round(
            result,
            2,
        )

    @staticmethod
    def _validate_inputs(
        question: Any,
        metric: Any,
        companies: Any,
        retrieved_chunks: Any,
    ) -> str | None:
        """
        Validate the values required for comparison.
        """

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return (
                "A non-empty comparison question is required."
            )

        if (
            not isinstance(metric, str)
            or not metric.strip()
        ):
            return (
                "A supported financial metric is required "
                "for company comparison."
            )

        if not isinstance(
            companies,
            list,
        ):
            return (
                "companies must be provided as a list."
            )

        unique_companies = {
            company
            for company in companies
            if isinstance(company, str)
            and company.strip()
        }

        if len(unique_companies) < 2:
            return (
                "At least two different companies are required "
                "for comparison."
            )

        if not isinstance(
            retrieved_chunks,
            list,
        ):
            return (
                "retrieved_chunks must be provided as a list."
            )

        if not retrieved_chunks:
            return (
                "No retrieved financial-report chunks were "
                "provided to the comparison tool."
            )

        return None

    def _failure(
        self,
        message: str,
    ) -> dict:
        """
        Return a consistent unsuccessful tool response.
        """

        return {
            "success": False,
            "tool": self.name,
            "metric": None,
            "results": [],
            "highest_company": None,
            "highest_value": None,
            "lowest_company": None,
            "lowest_value": None,
            "absolute_difference": None,
            "percentage_difference": None,
            "prompt_context": (
                "Verified company comparison was unavailable.\n"
                f"Reason: {message}"
            ),
            "error": message,
        }

    @staticmethod
    def build_prompt_context(
        comparison_result: dict,
    ) -> str:
        """
        Convert the verified comparison into LLM prompt context.
        """

        if not comparison_result.get(
            "success"
        ):
            return (
                "Verified company comparison was unavailable.\n"
                f"Reason: {comparison_result.get('error')}"
            )

        metric = comparison_result[
            "metric"
        ]

        sections = [
            "Verified deterministic Python company comparison:",
            "",
            f"Metric: {metric}",
            "",
            "Company values:",
        ]

        for result in comparison_result[
            "results"
        ]:
            sections.extend(
                [
                    (
                        f"- {result['company']}: "
                        f"{result['value']} "
                        f"for fiscal year {result['year']}"
                    ),
                    (
                        f"  Source: "
                        f"{result['source_file']}, "
                        f"page {result['page']}"
                    ),
                ]
            )

        sections.extend(
            [
                "",
                (
                    "Highest company: "
                    f"{comparison_result['highest_company']}"
                ),
                (
                    "Highest value: "
                    f"{comparison_result['highest_value']}"
                ),
                (
                    "Lowest company: "
                    f"{comparison_result['lowest_company']}"
                ),
                (
                    "Lowest value: "
                    f"{comparison_result['lowest_value']}"
                ),
                (
                    "Absolute difference: "
                    f"{comparison_result['absolute_difference']}"
                ),
            ]
        )

        percentage_difference = (
            comparison_result.get(
                "percentage_difference"
            )
        )

        if percentage_difference is not None:
            sections.append(
                (
                    "Percentage difference: "
                    f"{percentage_difference}%"
                )
            )

        sections.extend(
            [
                "",
                "Use these verified values exactly.",
                "Do not recalculate or replace the values.",
                "Clearly mention when company fiscal years differ.",
                "Only explain the comparison.",
            ]
        )

        return "\n".join(
            sections
        )


if __name__ == "__main__":
    tool = CompanyComparisonTool()

    test_chunks = [
        {
            "content": (
                "Year Ended September 28, 2024 2023 2022 "
                "Total net sales 391,035 383,285 394,328"
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
                "Year Ended June 30, 2024 2023 2022 "
                "Revenue 245,122 211,915 198,270"
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

    result = tool.run(
        question=(
            "Compare Apple and Microsoft revenue."
        ),
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=test_chunks,
    )

    print("Tool metadata:")
    print(
        tool.get_metadata()
    )

    print("\nComparison result:")
    print(result)

    print("\nVerified prompt context:")
    print(
        result["prompt_context"]
    )