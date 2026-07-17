from backend.app.services.financial_value_extractor import (
    FinancialValueExtractor,
)
from backend.app.tools.financial_calculator import (
    FinancialCalculator,
    FinancialCalculatorError,
)


class FinancialCalculationService:
    """
    Coordinate structured value extraction and deterministic
    Python-based financial calculations.

    The language model should explain the result, but Python
    performs the arithmetic.
    """

    def __init__(self) -> None:
        self.value_extractor = FinancialValueExtractor()
        self.calculator = FinancialCalculator()

    def calculate(
        self,
        question: str,
        metric: str | None,
        companies: list[str],
        retrieved_chunks: list[dict],
    ) -> dict:
        """
        Perform a supported financial calculation.

        Returns a structured result containing:
        - success status
        - calculation type
        - verified inputs
        - calculated result
        - source metadata
        - error message when calculation is unavailable
        """

        if not metric:
            return self._failure(
                "No supported financial metric was detected."
            )

        if not companies:
            return self._failure(
                "No company was detected for the calculation."
            )

        normalized_question = question.lower().strip()

        if self._is_growth_question(normalized_question):
            return self._calculate_growth(
                metric=metric,
                companies=companies,
                retrieved_chunks=retrieved_chunks,
            )

        return self._failure(
            "The selected financial calculation is not connected yet."
        )

    def _calculate_growth(
        self,
        metric: str,
        companies: list[str],
        retrieved_chunks: list[dict],
    ) -> dict:
        """
        Calculate year-over-year growth for one or more companies.
        """

        extracted_values = (
            self.value_extractor.extract_year_values(
                retrieved_chunks=retrieved_chunks,
                metric=metric,
            )
        )

        if not extracted_values:
            return self._failure(
                "No structured yearly values could be extracted "
                "from the retrieved report context."
            )

        company_results = []

        for company in companies:
            latest_values = (
                self.value_extractor.get_latest_two_values(
                    extracted_values=extracted_values,
                    company=company,
                )
            )

            if len(latest_values) < 2:
                continue

            current_entry = latest_values[0]
            previous_entry = latest_values[1]

            try:
                if metric == "eps":
                    calculation = (
                        self.calculator.calculate_eps_growth(
                            current_eps=current_entry["value"],
                            previous_eps=previous_entry["value"],
                        )
                    )
                else:
                    calculation = (
                        self.calculator.calculate_growth_rate(
                            current_value=current_entry["value"],
                            previous_value=previous_entry["value"],
                        )
                    )

            except FinancialCalculatorError as error:
                continue

            company_results.append(
                {
                    "company": company,
                    "metric": metric,
                    "current_year": current_entry["year"],
                    "current_value": current_entry["value"],
                    "previous_year": previous_entry["year"],
                    "previous_value": previous_entry["value"],
                    "result": calculation["result"],
                    "unit": calculation["unit"],
                    "formula": calculation["formula"],
                    "current_source": {
                        "source_file": current_entry.get(
                            "source_file"
                        ),
                        "page": current_entry.get("page"),
                    },
                    "previous_source": {
                        "source_file": previous_entry.get(
                            "source_file"
                        ),
                        "page": previous_entry.get("page"),
                    },
                }
            )

        if not company_results:
            return self._failure(
                "Two valid yearly values were not available for "
                "the requested company or companies."
            )

        return {
            "success": True,
            "calculation_type": "growth_rate",
            "metric": metric,
            "results": company_results,
            "error": None,
        }

    @staticmethod
    def _is_growth_question(
        question: str,
    ) -> bool:
        """
        Detect whether the user requested a growth calculation.
        """

        growth_terms = [
            "growth",
            "growth rate",
            "year over year",
            "year-over-year",
            "yoy",
            "percentage change",
            "percent change",
            "increase from",
            "decrease from",
        ]

        return any(
            term in question
            for term in growth_terms
        )

    @staticmethod
    def _failure(
        message: str,
    ) -> dict:
        """
        Return a consistent unsuccessful calculation result.
        """

        return {
            "success": False,
            "calculation_type": None,
            "metric": None,
            "results": [],
            "error": message,
        }

    @staticmethod
    def build_prompt_context(
        calculation_result: dict,
    ) -> str:
        """
        Convert verified Python calculation output into text
        that can be included in the LLM prompt.
        """

        if not calculation_result.get("success"):
            return (
                "Verified Python calculation was unavailable.\n"
                f"Reason: {calculation_result.get('error')}"
            )

        sections = [
            "Verified deterministic Python calculation:",
            "",
        ]

        for result in calculation_result["results"]:
            sections.extend(
                [
                    f"Company: {result['company']}",
                    f"Metric: {result['metric']}",
                    (
                        f"Current year: "
                        f"{result['current_year']}"
                    ),
                    (
                        f"Current value: "
                        f"{result['current_value']}"
                    ),
                    (
                        f"Previous year: "
                        f"{result['previous_year']}"
                    ),
                    (
                        f"Previous value: "
                        f"{result['previous_value']}"
                    ),
                    (
                        f"Calculated growth: "
                        f"{result['result']}%"
                    ),
                    f"Formula: {result['formula']}",
                    (
                        "Current value source: "
                        f"{result['current_source']}"
                    ),
                    (
                        "Previous value source: "
                        f"{result['previous_source']}"
                    ),
                    "",
                ]
            )

        sections.extend(
            [
                "Use these verified values exactly.",
                "Do not recalculate or change the result.",
                "Only explain the calculation clearly.",
            ]
        )

        return "\n".join(sections)


if __name__ == "__main__":
    service = FinancialCalculationService()

    test_chunks = [
        {
            "content": (
                "Year Ended September 28, 2024 2023 2022 "
                "Total net sales 391,035 383,285 394,328"
            ),
            "metadata": {
                "company": "Apple",
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
        }
    ]

    result = service.calculate(
        question="Calculate Apple's revenue growth.",
        metric="revenue",
        companies=["Apple"],
        retrieved_chunks=test_chunks,
    )

    print("Calculation result:")
    print(result)

    print("\nPrompt context:")
    print(
        service.build_prompt_context(result)
    )