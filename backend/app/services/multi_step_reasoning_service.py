from __future__ import annotations

from typing import Any


class MultiStepReasoningService:
    """
    Compare verified deterministic calculation results.

    The service does not perform document retrieval and does not
    ask an LLM to calculate financial values.

    It consumes the structured output already produced by the
    financial calculator and determines:

    - the strongest company
    - the weakest company
    - the difference between calculated results
    - a deterministic conclusion
    """

    def reason(
        self,
        decomposition: dict[str, Any],
        calculation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Build a deterministic multi-step reasoning result.
        """

        if not isinstance(decomposition, dict):
            return None

        if not decomposition.get(
            "calculated_comparison",
            False,
        ):
            return None

        if not isinstance(calculation, dict):
            return self._failure(
                "No verified calculation result was available."
            )

        if not calculation.get(
            "success",
            False,
        ):
            return self._failure(
                calculation.get(
                    "error",
                    (
                        "The required financial calculations "
                        "were unsuccessful."
                    ),
                )
            )

        calculation_results = calculation.get(
            "results",
            [],
        )

        if not isinstance(calculation_results, list):
            return self._failure(
                "The calculation result did not contain a valid results list."
            )

        normalized_results = (
            self._normalize_calculation_results(
                calculation_results
            )
        )

        if len(normalized_results) < 2:
            return self._failure(
                "At least two verified company calculations are "
                "required for multi-step comparison."
            )

        ranked_results = sorted(
            normalized_results,
            key=lambda item: item["result"],
            reverse=True,
        )

        strongest = ranked_results[0]
        weakest = ranked_results[-1]

        difference = round(
            strongest["result"]
            - weakest["result"],
            2,
        )

        metric = calculation.get(
            "metric"
        )

        comparison_type = (
            calculation.get(
                "calculation_type",
                "calculated_metric",
            )
        )

        conclusion = self._build_conclusion(
            strongest=strongest,
            weakest=weakest,
            difference=difference,
            metric=metric,
        )

        return {
            "success": True,
            "reasoning_type": "calculated_comparison",
            "comparison_type": comparison_type,
            "metric": metric,
            "ranked_results": ranked_results,
            "strongest_company": strongest["company"],
            "strongest_result": strongest["result"],
            "weakest_company": weakest["company"],
            "weakest_result": weakest["result"],
            "difference_percentage_points": difference,
            "conclusion": conclusion,
            "error": None,
        }

    @staticmethod
    def _normalize_calculation_results(
        calculation_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Keep only valid numeric company calculation results.
        """

        normalized_results: list[
            dict[str, Any]
        ] = []

        seen_companies: set[str] = set()

        for result in calculation_results:
            if not isinstance(result, dict):
                continue

            company = result.get(
                "company"
            )

            calculated_value = result.get(
                "result"
            )

            if (
                not isinstance(company, str)
                or not company.strip()
            ):
                continue

            if not isinstance(
                calculated_value,
                (int, float),
            ):
                continue

            company_name = company.strip()
            company_key = company_name.lower()

            if company_key in seen_companies:
                continue

            normalized_results.append(
                {
                    "company": company_name,
                    "metric": result.get(
                        "metric"
                    ),
                    "current_year": result.get(
                        "current_year"
                    ),
                    "previous_year": result.get(
                        "previous_year"
                    ),
                    "current_value": result.get(
                        "current_value"
                    ),
                    "previous_value": result.get(
                        "previous_value"
                    ),
                    "result": round(
                        float(calculated_value),
                        2,
                    ),
                    "unit": result.get(
                        "unit",
                        "percent",
                    ),
                    "formula": result.get(
                        "formula"
                    ),
                    "current_source": result.get(
                        "current_source"
                    ),
                    "previous_source": result.get(
                        "previous_source"
                    ),
                }
            )

            seen_companies.add(
                company_key
            )

        return normalized_results

    @staticmethod
    def _build_conclusion(
        strongest: dict[str, Any],
        weakest: dict[str, Any],
        difference: float,
        metric: Any,
    ) -> str:
        """
        Create a deterministic conclusion.
        """

        metric_name = (
            metric.replace("_", " ")
            if isinstance(metric, str)
            and metric.strip()
            else "financial metric"
        )

        strongest_company = strongest[
            "company"
        ]

        strongest_result = strongest[
            "result"
        ]

        weakest_company = weakest[
            "company"
        ]

        weakest_result = weakest[
            "result"
        ]

        if strongest_result == weakest_result:
            return (
                f"{strongest_company} and {weakest_company} "
                f"reported the same {metric_name} growth rate "
                f"of {strongest_result:.2f}%."
            )

        return (
            f"{strongest_company} was the stronger performer "
            f"based on {metric_name} growth. "
            f"{strongest_company} reported {strongest_result:.2f}% "
            f"growth compared with {weakest_company}'s "
            f"{weakest_result:.2f}%, a difference of "
            f"{difference:.2f} percentage points."
        )

    @staticmethod
    def build_answer(
        reasoning_result: dict[str, Any] | None,
    ) -> str | None:
        """
        Return the deterministic final answer.
        """

        if not isinstance(reasoning_result, dict):
            return None

        if not reasoning_result.get(
            "success",
            False,
        ):
            error = reasoning_result.get(
                "error",
                (
                    "The multi-step comparison could not "
                    "be completed."
                ),
            )

            return (
                "The calculated comparison could not be completed.\n\n"
                f"Reason: {error}"
            )

        ranked_results = reasoning_result.get(
            "ranked_results",
            [],
        )

        result_lines: list[str] = []

        for result in ranked_results:
            company = result.get(
                "company",
                "Unknown company",
            )

            calculated_value = result.get(
                "result"
            )

            current_year = result.get(
                "current_year",
                "current period",
            )

            previous_year = result.get(
                "previous_year",
                "previous period",
            )

            if isinstance(
                calculated_value,
                (int, float),
            ):
                result_lines.append(
                    (
                        f"- {company}: "
                        f"{float(calculated_value):.2f}% "
                        f"from {previous_year} to {current_year}"
                    )
                )

        conclusion = reasoning_result.get(
            "conclusion",
            "",
        )

        sections = [
            "Verified calculated growth comparison:",
            "",
            *result_lines,
        ]

        if conclusion:
            sections.extend(
                [
                    "",
                    conclusion,
                ]
            )

        return "\n".join(
            sections
        )

    @staticmethod
    def _failure(
        message: str,
    ) -> dict[str, Any]:
        """
        Return a consistent failure response.
        """

        return {
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
            "error": message,
        }


if __name__ == "__main__":
    service = MultiStepReasoningService()

    test_decomposition = {
        "is_complex": True,
        "reasoning_type": "calculated_comparison",
        "calculated_comparison": True,
        "companies": [
            "Apple",
            "Microsoft",
        ],
        "metric": "revenue",
    }

    test_calculation = {
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
            },
        ],
        "error": None,
    }

    reasoning = service.reason(
        decomposition=test_decomposition,
        calculation=test_calculation,
    )

    print(reasoning)
    print()
    print(
        service.build_answer(
            reasoning
        )
    )