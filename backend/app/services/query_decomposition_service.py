from __future__ import annotations

import re
from typing import Any


CALCULATION_PATTERNS = [
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bgrowth\b",
    r"\bgrowth rate\b",
    r"\byear[- ]over[- ]year\b",
    r"\byoy\b",
    r"\bpercentage change\b",
    r"\bpercent change\b",
]

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bwhich company\b",
    r"\bwhich one\b",
    r"\bperformed better\b",
    r"\bstronger\b",
    r"\bweaker\b",
    r"\bhigher growth\b",
    r"\blower growth\b",
    r"\bdifference\b",
]

CONCLUSION_PATTERNS = [
    r"\bexplain\b",
    r"\bidentify\b",
    r"\bdetermine\b",
    r"\bwhich performed better\b",
    r"\bwhich company performed better\b",
    r"\bstronger performer\b",
    r"\bbest performer\b",
]


class QueryDecompositionService:
    """
    Break complex financial questions into deterministic subtasks.

    This service does not call an LLM. It identifies whether the
    question requires multiple stages such as:

    - calculate a metric for multiple companies
    - compare the calculated results
    - identify the stronger performer
    """

    def decompose(
        self,
        question: str,
        companies: list[str],
        intent: str,
        metric: str | None,
    ) -> dict[str, Any]:
        """
        Return a structured decomposition plan.
        """

        normalized_question = (
            question.lower().strip()
            if isinstance(question, str)
            else ""
        )

        safe_companies = self._normalize_companies(
            companies
        )

        calculation_requested = self._matches_any(
            normalized_question,
            CALCULATION_PATTERNS,
        )

        comparison_requested = (
            intent == "comparison"
            or len(safe_companies) > 1
            or self._matches_any(
                normalized_question,
                COMPARISON_PATTERNS,
            )
        )

        conclusion_requested = self._matches_any(
            normalized_question,
            CONCLUSION_PATTERNS,
        )

        calculated_comparison = (
            calculation_requested
            and comparison_requested
            and len(safe_companies) >= 2
        )

        is_complex = (
            calculated_comparison
            or (
                comparison_requested
                and conclusion_requested
                and len(safe_companies) >= 2
            )
        )

        subtasks = self._build_subtasks(
            companies=safe_companies,
            metric=metric,
            calculation_requested=calculation_requested,
            comparison_requested=comparison_requested,
            conclusion_requested=conclusion_requested,
            calculated_comparison=calculated_comparison,
        )

        reasoning_type = self._determine_reasoning_type(
            calculation_requested=calculation_requested,
            comparison_requested=comparison_requested,
            calculated_comparison=calculated_comparison,
        )

        required_capabilities = (
            self._determine_required_capabilities(
                calculation_requested=calculation_requested,
                comparison_requested=comparison_requested,
                calculated_comparison=calculated_comparison,
            )
        )

        return {
            "is_complex": is_complex,
            "reasoning_type": reasoning_type,
            "calculation_requested": calculation_requested,
            "comparison_requested": comparison_requested,
            "conclusion_requested": conclusion_requested,
            "calculated_comparison": calculated_comparison,
            "companies": safe_companies,
            "metric": metric,
            "subtasks": subtasks,
            "required_capabilities": required_capabilities,
        }

    @staticmethod
    def _normalize_companies(
        companies: Any,
    ) -> list[str]:
        """
        Remove invalid and duplicate company names.
        """

        if not isinstance(companies, list):
            return []

        normalized_companies: list[str] = []
        seen_companies: set[str] = set()

        for company in companies:
            if not isinstance(company, str):
                continue

            cleaned_company = company.strip()

            if not cleaned_company:
                continue

            normalized_key = cleaned_company.lower()

            if normalized_key in seen_companies:
                continue

            normalized_companies.append(
                cleaned_company
            )

            seen_companies.add(
                normalized_key
            )

        return normalized_companies

    @staticmethod
    def _matches_any(
        question: str,
        patterns: list[str],
    ) -> bool:
        """
        Return True when at least one pattern matches.
        """

        return any(
            re.search(pattern, question)
            for pattern in patterns
        )

    @staticmethod
    def _build_subtasks(
        companies: list[str],
        metric: str | None,
        calculation_requested: bool,
        comparison_requested: bool,
        conclusion_requested: bool,
        calculated_comparison: bool,
    ) -> list[dict[str, Any]]:
        """
        Build ordered structured subtasks.
        """

        subtasks: list[dict[str, Any]] = []
        step_number = 1

        metric_name = (
            metric.replace("_", " ")
            if isinstance(metric, str)
            and metric.strip()
            else "requested financial metric"
        )

        if calculation_requested:
            for company in companies:
                subtasks.append(
                    {
                        "step": step_number,
                        "type": "calculation",
                        "company": company,
                        "metric": metric,
                        "description": (
                            f"Calculate {company}'s "
                            f"{metric_name} growth."
                        ),
                    }
                )

                step_number += 1

        if calculated_comparison:
            subtasks.append(
                {
                    "step": step_number,
                    "type": "calculated_comparison",
                    "company": None,
                    "metric": metric,
                    "description": (
                        "Compare the verified calculated "
                        f"{metric_name} growth rates."
                    ),
                }
            )

            step_number += 1

        elif comparison_requested:
            subtasks.append(
                {
                    "step": step_number,
                    "type": "absolute_comparison",
                    "company": None,
                    "metric": metric,
                    "description": (
                        "Compare the latest verified "
                        f"{metric_name} values."
                    ),
                }
            )

            step_number += 1

        if conclusion_requested or calculated_comparison:
            subtasks.append(
                {
                    "step": step_number,
                    "type": "conclusion",
                    "company": None,
                    "metric": metric,
                    "description": (
                        "Identify the stronger performer using "
                        "only verified deterministic results."
                    ),
                }
            )

        if not subtasks:
            subtasks.append(
                {
                    "step": 1,
                    "type": "retrieval",
                    "company": (
                        companies[0]
                        if len(companies) == 1
                        else None
                    ),
                    "metric": metric,
                    "description": (
                        "Retrieve relevant verified report evidence."
                    ),
                }
            )

        return subtasks

    @staticmethod
    def _determine_reasoning_type(
        calculation_requested: bool,
        comparison_requested: bool,
        calculated_comparison: bool,
    ) -> str:
        """
        Return the reasoning category.
        """

        if calculated_comparison:
            return "calculated_comparison"

        if calculation_requested:
            return "calculation"

        if comparison_requested:
            return "absolute_comparison"

        return "retrieval"

    @staticmethod
    def _determine_required_capabilities(
        calculation_requested: bool,
        comparison_requested: bool,
        calculated_comparison: bool,
    ) -> list[str]:
        """
        Return the ordered capabilities required.
        """

        capabilities = [
            "document_retrieval",
        ]

        if calculation_requested:
            capabilities.append(
                "financial_calculator"
            )

        if calculated_comparison:
            capabilities.append(
                "multi_step_reasoning"
            )

        elif comparison_requested:
            capabilities.append(
                "company_comparison"
            )

        return capabilities


if __name__ == "__main__":
    service = QueryDecompositionService()

    questions = [
        {
            "question": (
                "Compare Apple and Microsoft revenue growth "
                "and identify which company performed better."
            ),
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "intent": "comparison",
            "metric": "revenue",
        },
        {
            "question": (
                "Calculate Apple's revenue growth."
            ),
            "companies": [
                "Apple",
            ],
            "intent": "financial_metric",
            "metric": "revenue",
        },
        {
            "question": (
                "Compare Apple and Microsoft revenue."
            ),
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "intent": "comparison",
            "metric": "revenue",
        },
    ]

    for test_case in questions:
        result = service.decompose(
            question=test_case["question"],
            companies=test_case["companies"],
            intent=test_case["intent"],
            metric=test_case["metric"],
        )

        print("\n" + "=" * 80)
        print(test_case["question"])
        print(result)