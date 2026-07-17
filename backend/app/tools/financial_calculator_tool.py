from typing import Any

from backend.app.services.financial_calculation_service import (
    FinancialCalculationService,
)
from backend.app.tools.base_tool import BaseTool


class FinancialCalculatorTool(BaseTool):
    """
    Agent tool that extracts financial values from retrieved report
    chunks and performs deterministic calculations using Python.
    """

    name = "financial_calculator"

    description = (
        "Extracts structured financial values from company reports "
        "and performs verified calculations such as revenue growth "
        "and EPS growth using Python."
    )

    def __init__(self) -> None:
        self.calculation_service = (
            FinancialCalculationService()
        )

    def run(
        self,
        **kwargs: Any,
    ) -> dict:
        """
        Execute a financial calculation.

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
            companies=companies,
            retrieved_chunks=retrieved_chunks,
        )

        if validation_error:
            return self._failure(
                validation_error
            )

        calculation_result = (
            self.calculation_service.calculate(
                question=question,
                metric=metric,
                companies=companies,
                retrieved_chunks=retrieved_chunks,
            )
        )

        prompt_context = (
            self.calculation_service.build_prompt_context(
                calculation_result
            )
        )

        return {
            "success": calculation_result.get(
                "success",
                False,
            ),
            "tool": self.name,
            "calculation": calculation_result,
            "prompt_context": prompt_context,
            "error": calculation_result.get(
                "error"
            ),
        }

    @staticmethod
    def _validate_inputs(
        question: Any,
        companies: Any,
        retrieved_chunks: Any,
    ) -> str | None:
        """
        Validate inputs before executing the calculation service.
        """

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return (
                "A non-empty financial question is required."
            )

        if not isinstance(companies, list):
            return "companies must be provided as a list."

        if not companies:
            return (
                "At least one company is required for "
                "a financial calculation."
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
                "provided to the calculator tool."
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
            "calculation": None,
            "prompt_context": (
                "Verified Python calculation was unavailable.\n"
                f"Reason: {message}"
            ),
            "error": message,
        }


if __name__ == "__main__":
    tool = FinancialCalculatorTool()

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
        }
    ]

    result = tool.run(
        question="Calculate Apple's revenue growth.",
        metric="revenue",
        companies=["Apple"],
        retrieved_chunks=test_chunks,
    )

    print("Tool metadata:")
    print(tool.get_metadata())

    print("\nTool result:")
    print(result)

    print("\nVerified prompt context:")
    print(result["prompt_context"])