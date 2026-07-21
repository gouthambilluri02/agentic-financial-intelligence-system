from __future__ import annotations

import re


SUPPORTED_TOOLS = {
    "document_retrieval",
    "financial_calculator",
    "company_comparison",
    "calculated_comparison",
    "risk_analysis",
    "report_summary",
}


CALCULATION_PATTERNS = [
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bgrowth rate\b",
    r"\byear[- ]over[- ]year\b",
    r"\byoy\b",
    r"\bpercentage change\b",
    r"\bprofit margin\b",
    r"\boperating margin\b",
    r"\bgross margin\b",
    r"\bnet profit margin\b",
    r"\bcurrent ratio\b",
    r"\bdebt ratio\b",
    r"\breturn on equity\b",
    r"\broe\b",
    r"\breturn on assets\b",
    r"\broa\b",
]


COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bwhich company\b",
    r"\bwhich one\b",
    r"\bdifference between\b",
    r"\bhigher than\b",
    r"\blower than\b",
    r"\bstronger\b",
    r"\bweaker\b",
    r"\bperformed better\b",
    r"\bbetter performance\b",
]


CALCULATED_COMPARISON_PATTERNS = [
    r"\bcompare\b.*\bgrowth\b",
    r"\bgrowth\b.*\bcompare\b",
    r"\bcomparison\b.*\bgrowth\b",
    r"\bgrowth\b.*\bcomparison\b",
    r"\bwhich company\b.*\bgrowth\b",
    r"\bwhich company\b.*\bperformed better\b",
    r"\bwhich one\b.*\bperformed better\b",
    r"\bperformed better\b.*\bgrowth\b",
    r"\bcompare\b.*\bpercentage change\b",
    r"\bpercentage change\b.*\bcompare\b",
    r"\bcompare\b.*\bmargin\b",
    r"\bmargin\b.*\bcompare\b",
    r"\bcompare\b.*\bratio\b",
    r"\bratio\b.*\bcompare\b",
    r"\bcompare\b.*\breturn on equity\b",
    r"\bcompare\b.*\breturn on assets\b",
    r"\bcompare\b.*\broe\b",
    r"\bcompare\b.*\broa\b",
]


RISK_PATTERNS = [
    r"\brisk\b",
    r"\brisks\b",
    r"\brisk factor\b",
    r"\brisk factors\b",
    r"\bdisclosed risks\b",
    r"\bwhat could affect\b",
    r"\bwhat could impact\b",
    r"\bthreats\b",
    r"\bchallenges\b",
    r"\bregulatory risk\b",
    r"\bcybersecurity risk\b",
    r"\boperational risk\b",
    r"\blegal risk\b",
    r"\bprivacy risk\b",
    r"\bsupply chain risk\b",
    r"\bcompetition risk\b",
]


SUMMARY_PATTERNS = [
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\bexecutive summary\b",
    r"\bbriefly explain\b",
    r"\bkey highlights\b",
    r"\bmain findings\b",
]


class ToolRouter:
    """
    Select the most appropriate tool for a financial question.

    Routing is deterministic so the behavior is easy to test,
    understand, and debug.
    """

    def select_tool(
        self,
        question: str,
        intent: str,
        metric: str | None,
        companies: list[str],
    ) -> str:
        """
        Return the tool that should handle the question.

        Routing priority:

        1. Calculated multi-company comparisons
        2. Single or multi-company calculations
        3. Direct company value comparisons
        4. Risk analysis
        5. Report summaries
        6. Document retrieval
        """

        normalized_question = (
            question.lower().strip()
            if isinstance(question, str)
            else ""
        )

        normalized_intent = (
            intent.strip().lower()
            if isinstance(intent, str)
            else "general_question"
        )

        normalized_companies = self._normalize_companies(
            companies
        )

        has_calculation_signal = self._matches_any(
            normalized_question,
            CALCULATION_PATTERNS,
        )

        has_comparison_signal = (
            normalized_intent == "comparison"
            or len(normalized_companies) > 1
            or self._matches_any(
                normalized_question,
                COMPARISON_PATTERNS,
            )
        )

        has_calculated_comparison_signal = (
            len(normalized_companies) > 1
            and has_comparison_signal
            and (
                has_calculation_signal
                or self._matches_any(
                    normalized_question,
                    CALCULATED_COMPARISON_PATTERNS,
                )
            )
        )

        if has_calculated_comparison_signal:
            return "calculated_comparison"

        if has_calculation_signal:
            return "financial_calculator"

        if has_comparison_signal:
            return "company_comparison"

        if (
            normalized_intent == "risk_analysis"
            or self._matches_any(
                normalized_question,
                RISK_PATTERNS,
            )
        ):
            return "risk_analysis"

        if (
            normalized_intent == "summary"
            or self._matches_any(
                normalized_question,
                SUMMARY_PATTERNS,
            )
        ):
            return "report_summary"

        return "document_retrieval"

    @staticmethod
    def _matches_any(
        question: str,
        patterns: list[str],
    ) -> bool:
        """
        Return True when the question matches at least one pattern.
        """

        return any(
            re.search(
                pattern,
                question,
            )
            is not None
            for pattern in patterns
        )

    @staticmethod
    def _normalize_companies(
        companies: object,
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
    def is_supported(
        tool_name: str,
    ) -> bool:
        """
        Check whether a tool name is supported by the router.
        """

        if not isinstance(
            tool_name,
            str,
        ):
            return False

        return (
            tool_name.strip()
            in SUPPORTED_TOOLS
        )


if __name__ == "__main__":
    router = ToolRouter()

    test_cases = [
        {
            "question": (
                "What risks did Apple disclose?"
            ),
            "intent": "risk_analysis",
            "metric": None,
            "companies": ["Apple"],
            "expected_tool": "risk_analysis",
        },
        {
            "question": (
                "What cybersecurity threats did "
                "Microsoft disclose?"
            ),
            "intent": "general_question",
            "metric": None,
            "companies": ["Microsoft"],
            "expected_tool": "risk_analysis",
        },
        {
            "question": (
                "Calculate Apple's revenue growth."
            ),
            "intent": "financial_metric",
            "metric": "revenue",
            "companies": ["Apple"],
            "expected_tool": "financial_calculator",
        },
        {
            "question": (
                "Compare Apple and Microsoft revenue."
            ),
            "intent": "comparison",
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "company_comparison",
        },
        {
            "question": (
                "Compare Apple and Microsoft revenue "
                "growth."
            ),
            "intent": "comparison",
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "calculated_comparison",
        },
        {
            "question": (
                "Compare Apple and Microsoft revenue "
                "growth and explain which company "
                "performed better."
            ),
            "intent": "comparison",
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "calculated_comparison",
        },
        {
            "question": (
                "Which company had the higher revenue "
                "growth rate, Apple or Microsoft?"
            ),
            "intent": "comparison",
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "calculated_comparison",
        },
        {
            "question": (
                "Compare Apple and Microsoft operating "
                "margin."
            ),
            "intent": "comparison",
            "metric": "operating_margin",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "calculated_comparison",
        },
        {
            "question": (
                "Summarize Microsoft's annual report."
            ),
            "intent": "summary",
            "metric": None,
            "companies": ["Microsoft"],
            "expected_tool": "report_summary",
        },
        {
            "question": (
                "What was Microsoft's revenue?"
            ),
            "intent": "financial_metric",
            "metric": "revenue",
            "companies": ["Microsoft"],
            "expected_tool": "document_retrieval",
        },
        {
            "question": (
                "Which company reported higher revenue?"
            ),
            "intent": "comparison",
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "expected_tool": "company_comparison",
        },
    ]

    passed_count = 0

    for test_case in test_cases:
        selected_tool = router.select_tool(
            question=test_case["question"],
            intent=test_case["intent"],
            metric=test_case["metric"],
            companies=test_case["companies"],
        )

        passed = (
            selected_tool
            == test_case["expected_tool"]
        )

        if passed:
            passed_count += 1

        print("\n" + "=" * 70)
        print(
            f"Question: {test_case['question']}"
        )
        print(
            f"Intent: {test_case['intent']}"
        )
        print(
            f"Metric: {test_case['metric']}"
        )
        print(
            f"Companies: {test_case['companies']}"
        )
        print(
            f"Selected tool: {selected_tool}"
        )
        print(
            "Expected tool: "
            f"{test_case['expected_tool']}"
        )
        print(
            "Supported: "
            f"{router.is_supported(selected_tool)}"
        )
        print(
            f"Test passed: {passed}"
        )

    print("\n" + "=" * 70)
    print(
        f"Passed {passed_count} "
        f"of {len(test_cases)} tests."
    )