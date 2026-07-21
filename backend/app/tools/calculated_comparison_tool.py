from __future__ import annotations

from typing import Any

from backend.app.services.financial_calculation_service import (
    FinancialCalculationService,
)
from backend.app.services.multi_step_reasoning_service import (
    MultiStepReasoningService,
)
from backend.app.services.query_decomposition_service import (
    QueryDecompositionService,
)
from backend.app.tools.base_tool import BaseTool


class CalculatedComparisonTool(BaseTool):
    """
    Perform a deterministic comparison of calculated financial metrics.

    Example:
    "Compare Apple and Microsoft revenue growth and explain
    who performed better."

    Workflow:
    1. Validate the request.
    2. Decompose the question.
    3. Calculate the requested metric for every company.
    4. Compare the verified calculated results.
    5. Produce structured deterministic reasoning.

    This tool performs the complete calculated-comparison workflow
    internally because ToolExecutor does not currently pass one tool's
    output into the next tool.
    """

    name = "calculated_comparison"

    description = (
        "Calculates a financial metric such as revenue growth for "
        "multiple companies and deterministically compares the "
        "verified calculated results."
    )

    def __init__(
        self,
        calculation_service: FinancialCalculationService | None = None,
        decomposition_service: QueryDecompositionService | None = None,
        reasoning_service: MultiStepReasoningService | None = None,
    ) -> None:
        self.calculation_service = (
            calculation_service
            if calculation_service is not None
            else FinancialCalculationService()
        )

        self.decomposition_service = (
            decomposition_service
            if decomposition_service is not None
            else QueryDecompositionService()
        )

        self.reasoning_service = (
            reasoning_service
            if reasoning_service is not None
            else MultiStepReasoningService()
        )

    def run(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a calculated comparison.

        Required arguments:
        - question
        - metric
        - companies
        - retrieved_chunks
        """

        question = kwargs.get(
            "question"
        )

        metric = kwargs.get(
            "metric"
        )

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
                message=validation_error,
            )

        normalized_question = question.strip()

        normalized_companies = self._normalize_companies(
            companies
        )

        decomposition = (
            self.decomposition_service.decompose(
                question=normalized_question,
                companies=normalized_companies,
                intent="comparison",
                metric=metric,
            )
        )

        if not isinstance(
            decomposition,
            dict,
        ):
            return self._failure(
                message=(
                    "Query decomposition returned an invalid result."
                ),
            )

        calculation = (
            self.calculation_service.calculate(
                question=normalized_question,
                metric=metric,
                companies=normalized_companies,
                retrieved_chunks=retrieved_chunks,
            )
        )

        if not isinstance(
            calculation,
            dict,
        ):
            return self._failure(
                message=(
                    "Financial calculation returned an invalid result."
                ),
                decomposition=decomposition,
            )

        reasoning = self.reasoning_service.reason(
            decomposition=decomposition,
            calculation=calculation,
        )

        if not isinstance(
            reasoning,
            dict,
        ):
            return self._failure(
                message=(
                    "Multi-step reasoning returned an invalid result."
                ),
                decomposition=decomposition,
                calculation=calculation,
            )

        success = bool(
            calculation.get(
                "success",
                False,
            )
            and reasoning.get(
                "success",
                False,
            )
        )

        if not success:
            error = (
                reasoning.get("error")
                or calculation.get("error")
                or (
                    "The calculated comparison could not "
                    "be completed."
                )
            )

            return {
                "success": False,
                "tool": self.name,
                "decomposition": decomposition,
                "calculation": calculation,
                "reasoning": reasoning,
                "prompt_context": (
                    self._build_prompt_context(
                        decomposition=decomposition,
                        calculation=calculation,
                        reasoning=reasoning,
                    )
                ),
                "error": str(error),
            }

        prompt_context = self._build_prompt_context(
            decomposition=decomposition,
            calculation=calculation,
            reasoning=reasoning,
        )

        return {
            "success": True,
            "tool": self.name,
            "decomposition": decomposition,
            "calculation": calculation,
            "reasoning": reasoning,
            "prompt_context": prompt_context,
            "error": None,
        }

    def _build_prompt_context(
        self,
        decomposition: dict[str, Any],
        calculation: dict[str, Any],
        reasoning: dict[str, Any],
    ) -> str:
        """
        Build verified context for downstream response generation.

        The deterministic response builder should normally generate
        the final answer directly, but this context remains available
        for diagnostics and fallback generation.
        """

        sections: list[str] = [
            "VERIFIED CALCULATED COMPARISON",
            (
                "All values below were extracted and calculated "
                "deterministically. Do not recalculate, replace, "
                "or contradict them."
            ),
        ]

        metric = reasoning.get(
            "metric",
            calculation.get(
                "metric",
                "financial metric",
            ),
        )

        sections.extend(
            [
                "",
                f"Metric: {self._format_metric_name(metric)}",
            ]
        )

        calculation_results = calculation.get(
            "results",
            [],
        )

        if isinstance(
            calculation_results,
            list,
        ):
            sections.extend(
                [
                    "",
                    "Verified company calculations:",
                ]
            )

            for result in calculation_results:
                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                company = result.get(
                    "company",
                    "Unknown company",
                )

                previous_year = result.get(
                    "previous_year",
                    "previous period",
                )

                previous_value = result.get(
                    "previous_value",
                    "unavailable",
                )

                current_year = result.get(
                    "current_year",
                    "current period",
                )

                current_value = result.get(
                    "current_value",
                    "unavailable",
                )

                calculated_result = result.get(
                    "result",
                )

                unit = result.get(
                    "unit",
                    "percent",
                )

                result_text = self._format_result_value(
                    value=calculated_result,
                    unit=unit,
                )

                sections.extend(
                    [
                        "",
                        f"Company: {company}",
                        (
                            f"Previous period: {previous_year} "
                            f"({previous_value})"
                        ),
                        (
                            f"Current period: {current_year} "
                            f"({current_value})"
                        ),
                        f"Verified result: {result_text}",
                    ]
                )

        ranked_results = reasoning.get(
            "ranked_results",
            [],
        )

        if isinstance(
            ranked_results,
            list,
        ) and ranked_results:
            sections.extend(
                [
                    "",
                    "Verified ranking:",
                ]
            )

            for rank, result in enumerate(
                ranked_results,
                start=1,
            ):
                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                company = result.get(
                    "company",
                    "Unknown company",
                )

                value = result.get(
                    "result",
                )

                unit = result.get(
                    "unit",
                    "percent",
                )

                result_text = self._format_result_value(
                    value=value,
                    unit=unit,
                )

                sections.append(
                    f"{rank}. {company}: {result_text}"
                )

        strongest_company = reasoning.get(
            "strongest_company"
        )

        strongest_result = reasoning.get(
            "strongest_result"
        )

        weakest_company = reasoning.get(
            "weakest_company"
        )

        weakest_result = reasoning.get(
            "weakest_result"
        )

        difference = reasoning.get(
            "difference_percentage_points"
        )

        if strongest_company:
            sections.extend(
                [
                    "",
                    (
                        "Strongest performer: "
                        f"{strongest_company}"
                    ),
                    (
                        "Strongest verified result: "
                        f"{self._format_result_value(strongest_result, 'percent')}"
                    ),
                ]
            )

        if weakest_company:
            sections.extend(
                [
                    (
                        "Weakest performer: "
                        f"{weakest_company}"
                    ),
                    (
                        "Weakest verified result: "
                        f"{self._format_result_value(weakest_result, 'percent')}"
                    ),
                ]
            )

        if isinstance(
            difference,
            (int, float),
        ):
            sections.append(
                (
                    "Difference: "
                    f"{float(difference):.2f} percentage points"
                )
            )

        conclusion = reasoning.get(
            "conclusion"
        )

        if (
            isinstance(conclusion, str)
            and conclusion.strip()
        ):
            sections.extend(
                [
                    "",
                    "Verified conclusion:",
                    conclusion.strip(),
                ]
            )

        decomposition_subtasks = decomposition.get(
            "subtasks",
            [],
        )

        if isinstance(
            decomposition_subtasks,
            list,
        ) and decomposition_subtasks:
            sections.extend(
                [
                    "",
                    "Completed reasoning steps:",
                ]
            )

            for subtask in decomposition_subtasks:
                if not isinstance(
                    subtask,
                    dict,
                ):
                    continue

                step = subtask.get(
                    "step",
                    "?",
                )

                description = subtask.get(
                    "description",
                    "Completed reasoning step.",
                )

                sections.append(
                    f"{step}. {description}"
                )

        return "\n".join(
            sections
        )

    @staticmethod
    def _validate_inputs(
        question: Any,
        metric: Any,
        companies: Any,
        retrieved_chunks: Any,
    ) -> str | None:
        """
        Validate inputs before running the calculated comparison.
        """

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return (
                "A non-empty financial comparison question "
                "is required."
            )

        if (
            not isinstance(metric, str)
            or not metric.strip()
        ):
            return (
                "A supported financial metric is required for "
                "a calculated comparison."
            )

        if not isinstance(
            companies,
            list,
        ):
            return (
                "companies must be provided as a list."
            )

        normalized_companies = (
            CalculatedComparisonTool._normalize_companies(
                companies
            )
        )

        if len(normalized_companies) < 2:
            return (
                "At least two companies are required for "
                "a calculated comparison."
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
                "provided for the calculated comparison."
            )

        return None

    @staticmethod
    def _normalize_companies(
        companies: Any,
    ) -> list[str]:
        """
        Remove invalid and duplicate company names.
        """

        if not isinstance(
            companies,
            list,
        ):
            return []

        normalized_companies: list[str] = []
        seen_companies: set[str] = set()

        for company in companies:
            if not isinstance(
                company,
                str,
            ):
                continue

            cleaned_company = company.strip()

            if not cleaned_company:
                continue

            company_key = cleaned_company.lower()

            if company_key in seen_companies:
                continue

            normalized_companies.append(
                cleaned_company
            )

            seen_companies.add(
                company_key
            )

        return normalized_companies

    @staticmethod
    def _format_result_value(
        value: Any,
        unit: Any,
    ) -> str:
        """
        Format a deterministic calculated result.
        """

        if not isinstance(
            value,
            (int, float),
        ):
            return "Unavailable"

        numeric_value = float(
            value
        )

        if unit == "percent":
            return f"{numeric_value:.2f}%"

        if (
            isinstance(unit, str)
            and unit.strip()
        ):
            return (
                f"{numeric_value:.2f} "
                f"{unit.strip()}"
            )

        return f"{numeric_value:.2f}"

    @staticmethod
    def _format_metric_name(
        metric: Any,
    ) -> str:
        """
        Convert an internal metric name into a readable label.
        """

        if not isinstance(
            metric,
            str,
        ):
            return "Financial Metric"

        cleaned_metric = metric.strip()

        if not cleaned_metric:
            return "Financial Metric"

        return cleaned_metric.replace(
            "_",
            " ",
        ).title()

    def _failure(
        self,
        message: str,
        decomposition: dict[str, Any] | None = None,
        calculation: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a consistent unsuccessful tool response.
        """

        return {
            "success": False,
            "tool": self.name,
            "decomposition": decomposition,
            "calculation": calculation,
            "reasoning": reasoning,
            "prompt_context": (
                "Verified calculated comparison was unavailable.\n"
                f"Reason: {message}"
            ),
            "error": message,
        }


if __name__ == "__main__":
    tool = CalculatedComparisonTool()

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
            "Compare Apple and Microsoft revenue growth and "
            "explain which company performed better."
        ),
        metric="revenue",
        companies=[
            "Apple",
            "Microsoft",
        ],
        retrieved_chunks=test_chunks,
    )

    print("=" * 80)
    print("TOOL METADATA")
    print(tool.get_metadata())

    print("\n" + "=" * 80)
    print("SUCCESS")
    print(result.get("success"))

    print("\n" + "=" * 80)
    print("DECOMPOSITION")
    print(result.get("decomposition"))

    print("\n" + "=" * 80)
    print("CALCULATION")
    print(result.get("calculation"))

    print("\n" + "=" * 80)
    print("REASONING")
    print(result.get("reasoning"))

    print("\n" + "=" * 80)
    print("PROMPT CONTEXT")
    print(result.get("prompt_context"))

    print("\n" + "=" * 80)
    print("ERROR")
    print(result.get("error"))