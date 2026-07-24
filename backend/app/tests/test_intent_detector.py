"""
Tests for IntentDetector.

Covered behavior:

- Supported intent patterns
- Case-insensitive matching
- Word-boundary behavior
- Intent precedence
- General-question fallback
- Invalid input behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.intent_detector import (
    INTENT_PATTERNS,
    IntentDetector,
)


@pytest.fixture
def detector() -> IntentDetector:
    """Return a fresh IntentDetector instance."""

    return IntentDetector()


class TestIntentPatternsConfiguration:
    """Tests for the INTENT_PATTERNS configuration."""

    def test_patterns_is_dictionary(self) -> None:
        assert isinstance(INTENT_PATTERNS, dict)

    def test_contains_expected_intents(self) -> None:
        assert list(INTENT_PATTERNS) == [
            "comparison",
            "summary",
            "risk_analysis",
            "financial_metric",
        ]

    @pytest.mark.parametrize(
        "intent",
        [
            "comparison",
            "summary",
            "risk_analysis",
            "financial_metric",
        ],
    )
    def test_each_intent_has_pattern_list(
        self,
        intent: str,
    ) -> None:
        assert isinstance(INTENT_PATTERNS[intent], list)
        assert INTENT_PATTERNS[intent]

    def test_all_patterns_are_non_empty_strings(self) -> None:
        for patterns in INTENT_PATTERNS.values():
            for pattern in patterns:
                assert isinstance(pattern, str)
                assert pattern

    def test_comparison_patterns_are_configured(self) -> None:
        assert INTENT_PATTERNS["comparison"] == [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bwhich company\b",
            r"\bdifference between\b",
        ]

    def test_summary_patterns_are_configured(self) -> None:
        assert INTENT_PATTERNS["summary"] == [
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\boverview\b",
            r"\bbriefly explain\b",
        ]

    def test_risk_patterns_are_configured(self) -> None:
        assert INTENT_PATTERNS["risk_analysis"] == [
            r"\brisk\b",
            r"\brisks\b",
            r"\brisk factors\b",
            r"\bcybersecurity\b",
            r"\blegal proceedings\b",
        ]

    def test_financial_metric_patterns_are_configured(self) -> None:
        assert INTENT_PATTERNS["financial_metric"] == [
            r"\brevenue\b",
            r"\bnet income\b",
            r"\bearnings per share\b",
            r"\beps\b",
            r"\bcash flow\b",
            r"\boperating margin\b",
            r"\bgross margin\b",
            r"\bdebt\b",
            r"\bassets\b",
            r"\bliabilities\b",
        ]


class TestComparisonIntent:
    """Tests for comparison intent detection."""

    @pytest.mark.parametrize(
        "question",
        [
            "Compare Apple and Microsoft.",
            "Can you compare Apple with Microsoft?",
            "Please compare their revenue.",
            "COMPARE APPLE AND MICROSOFT.",
            "A comparison of Apple and Microsoft.",
            "Show me a comparison between these companies.",
            "Apple versus Microsoft.",
            "Apple Versus Microsoft.",
            "Apple vs Microsoft.",
            "Apple vs. Microsoft.",
            "Which company has higher revenue?",
            "Which company performed better?",
            "What is the difference between Apple and Microsoft?",
            "Explain the difference between their revenue values.",
        ],
    )
    def test_detects_comparison_phrases(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "comparison"

    @pytest.mark.parametrize(
        "question",
        [
            "Compare Apple and Microsoft revenue.",
            "Which company has the higher net income?",
            "Apple versus Microsoft operating margin.",
            "What is the difference between their assets?",
        ],
    )
    def test_comparison_takes_precedence_over_financial_metric(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "comparison"

    @pytest.mark.parametrize(
        "question",
        [
            "Compare the disclosed risks.",
            "Which company has greater cybersecurity risk?",
            "Apple versus Microsoft risk factors.",
        ],
    )
    def test_comparison_takes_precedence_over_risk_analysis(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "comparison"

    @pytest.mark.parametrize(
        "question",
        [
            "Compare and summarize Apple and Microsoft.",
            "Give a comparison and overview.",
            "Which company is stronger? Briefly explain.",
        ],
    )
    def test_comparison_takes_precedence_over_summary(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "comparison"

    @pytest.mark.parametrize(
        "question",
        [
            "comparable businesses",
            "comparisonless",
            "versuslike",
            "advs",
            "which companies operate here?",
            "difference-between values",
        ],
    )
    def test_partial_or_non_matching_comparison_words_do_not_match(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) != "comparison"


class TestSummaryIntent:
    """Tests for summary intent detection."""

    @pytest.mark.parametrize(
        "question",
        [
            "Summarize Apple's annual report.",
            "Please summarize the filing.",
            "SUMMARIZE THE REPORT.",
            "Give me a summary of Microsoft's business.",
            "Provide a short summary.",
            "Give me an overview of Apple.",
            "Business overview for Microsoft.",
            "Briefly explain Apple's performance.",
            "Can you briefly explain the report?",
        ],
    )
    def test_detects_summary_phrases(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "summary"

    @pytest.mark.parametrize(
        "question",
        [
            "Summarize Apple's revenue.",
            "Give me an overview of net income.",
            "Briefly explain the operating margin.",
        ],
    )
    def test_summary_takes_precedence_over_financial_metric(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "summary"

    @pytest.mark.parametrize(
        "question",
        [
            "Summarize the risks.",
            "Give me an overview of cybersecurity risk.",
            "Briefly explain the risk factors.",
        ],
    )
    def test_summary_takes_precedence_over_risk_analysis(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "summary"

    @pytest.mark.parametrize(
        "question",
        [
            "summarized results",
            "summarizer",
            "overviewing",
            "brief explanation",
            "briefly explained",
        ],
    )
    def test_partial_or_different_summary_words_do_not_match(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) != "summary"


class TestRiskAnalysisIntent:
    """Tests for risk-analysis intent detection."""

    @pytest.mark.parametrize(
        "question",
        [
            "What risk did Apple disclose?",
            "What risks did Microsoft disclose?",
            "Describe the company's risk factors.",
            "List the major risk factors.",
            "What cybersecurity concerns were disclosed?",
            "Explain Microsoft's cybersecurity exposure.",
            "What legal proceedings did Apple disclose?",
            "Summarize the legal proceedings.",
            "RISK",
            "RISKS",
            "CYBERSECURITY",
            "LEGAL PROCEEDINGS",
        ],
    )
    def test_detects_risk_phrases(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        expected = (
            "summary"
            if "summarize" in question.lower()
            else "risk_analysis"
        )
        assert detector.detect_intent(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "What revenue risk did Apple disclose?",
            "What risk affects net income?",
            "Explain cybersecurity impacts on cash flow.",
            "What legal proceedings affected assets?",
        ],
    )
    def test_risk_analysis_takes_precedence_over_financial_metric(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "risk_analysis"

    @pytest.mark.parametrize(
        "question",
        [
            "risky business",
            "riskier operations",
            "legal proceeding",
            "proceedings",
        ],
    )
    def test_partial_or_different_risk_words_do_not_match(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) != "risk_analysis"


class TestFinancialMetricIntent:
    """Tests for financial-metric intent detection."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was Apple's revenue?",
            "Show the revenue for Microsoft.",
            "What was net income?",
            "Explain Apple's net income.",
            "What were earnings per share?",
            "Show earnings per share for 2024.",
            "What was EPS?",
            "Calculate the EPS.",
            "What was cash flow?",
            "Show operating cash flow.",
            "What was the operating margin?",
            "Explain gross margin.",
            "How much debt did Apple report?",
            "What were total assets?",
            "What liabilities were reported?",
            "REVENUE",
            "NET INCOME",
            "EARNINGS PER SHARE",
            "EPS",
            "CASH FLOW",
            "OPERATING MARGIN",
            "GROSS MARGIN",
            "DEBT",
            "ASSETS",
            "LIABILITIES",
        ],
    )
    def test_detects_financial_metric_phrases(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "financial_metric"

    @pytest.mark.parametrize(
        "question",
        [
            "revenues",
            "net incomes",
            "earning per share",
            "epstein",
            "cashflow",
            "operating-margin",
            "gross-margin",
            "indebtedness",
            "asset",
            "liability",
        ],
    )
    def test_non_matching_metric_variants_fall_back_to_general(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "general_question"

    def test_first_matching_metric_pattern_returns_financial_metric(
        self,
        detector: IntentDetector,
    ) -> None:
        question = (
            "Discuss revenue, net income, EPS, cash flow, debt, "
            "assets, and liabilities."
        )

        assert detector.detect_intent(question) == "financial_metric"


class TestGeneralQuestionIntent:
    """Tests for general-question fallback."""

    @pytest.mark.parametrize(
        "question",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            "What products does Microsoft offer?",
            "Who is the chief executive?",
            "Describe the company's business.",
            "When was the company founded?",
            "What markets does Apple serve?",
            "Tell me about the company.",
            "How many employees work there?",
            "What is the company's strategy?",
        ],
    )
    def test_returns_general_question_when_no_pattern_matches(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "general_question"

    def test_punctuation_only_returns_general_question(
        self,
        detector: IntentDetector,
    ) -> None:
        assert detector.detect_intent("?!.,;:") == "general_question"

    def test_numbers_only_returns_general_question(
        self,
        detector: IntentDetector,
    ) -> None:
        assert detector.detect_intent("2024 2023 100") == (
            "general_question"
        )

    def test_unicode_text_without_pattern_returns_general_question(
        self,
        detector: IntentDetector,
    ) -> None:
        assert detector.detect_intent(
            "Компания туралы ақпарат беріңіз."
        ) == "general_question"


class TestCaseInsensitivity:
    """Tests for case-insensitive intent matching."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("compare apple and microsoft", "comparison"),
            ("COMPARE APPLE AND MICROSOFT", "comparison"),
            ("CoMpArE Apple and Microsoft", "comparison"),
            ("summarize apple", "summary"),
            ("SUMMARIZE APPLE", "summary"),
            ("SuMmArIzE Apple", "summary"),
            ("what RISKS were disclosed?", "risk_analysis"),
            ("what CyberSecurity issues exist?", "risk_analysis"),
            ("what was REVENUE?", "financial_metric"),
            ("what was Net Income?", "financial_metric"),
        ],
    )
    def test_matching_is_case_insensitive(
        self,
        detector: IntentDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_intent(question) == expected


class TestWordBoundaryBehavior:
    """Tests for regular-expression word boundaries."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("compare?", "comparison"),
            ("compare.", "comparison"),
            ("(compare)", "comparison"),
            ("summary:", "summary"),
            ("risk?", "risk_analysis"),
            ("revenue.", "financial_metric"),
            ("EPS?", "financial_metric"),
            ("assets/liabilities", "financial_metric"),
        ],
    )
    def test_keywords_match_next_to_punctuation(
        self,
        detector: IntentDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_intent(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "prerevenue",
            "revenueless",
            "assetbased",
            "liabilitiesrelated",
            "cybersecurityrisk",
            "riskfactor",
            "summarizethis",
            "comparethem",
        ],
    )
    def test_keywords_do_not_match_inside_larger_words(
        self,
        detector: IntentDetector,
        question: str,
    ) -> None:
        assert detector.detect_intent(question) == "general_question"


class TestIntentPrecedence:
    """Tests for the configured intent priority order."""

    def test_comparison_has_highest_precedence(
        self,
        detector: IntentDetector,
    ) -> None:
        question = (
            "Compare and summarize the revenue risks of Apple "
            "and Microsoft."
        )

        assert detector.detect_intent(question) == "comparison"

    def test_summary_has_precedence_over_risk_and_metric(
        self,
        detector: IntentDetector,
    ) -> None:
        question = (
            "Summarize the revenue and cybersecurity risks."
        )

        assert detector.detect_intent(question) == "summary"

    def test_risk_has_precedence_over_metric(
        self,
        detector: IntentDetector,
    ) -> None:
        question = (
            "What risk could affect revenue and net income?"
        )

        assert detector.detect_intent(question) == "risk_analysis"

    def test_financial_metric_used_when_no_higher_priority_match(
        self,
        detector: IntentDetector,
    ) -> None:
        question = "What was revenue and net income?"

        assert detector.detect_intent(question) == "financial_metric"


class TestInvalidInput:
    """Tests documenting current invalid-input behavior."""

    @pytest.mark.parametrize(
        "question",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
            object(),
        ],
    )
    def test_non_string_input_raises_attribute_error(
        self,
        detector: IntentDetector,
        question: Any,
    ) -> None:
        with pytest.raises(AttributeError):
            detector.detect_intent(question)  # type: ignore[arg-type]


class TestDetectorInstance:
    """Basic tests for detector construction and repeatability."""

    def test_detector_can_be_instantiated(self) -> None:
        detector = IntentDetector()

        assert isinstance(detector, IntentDetector)

    def test_same_question_returns_same_intent(
        self,
        detector: IntentDetector,
    ) -> None:
        question = "Compare Apple and Microsoft revenue."

        first = detector.detect_intent(question)
        second = detector.detect_intent(question)

        assert first == second == "comparison"

    def test_detector_is_stateless_across_questions(
        self,
        detector: IntentDetector,
    ) -> None:
        assert detector.detect_intent(
            "Compare Apple and Microsoft."
        ) == "comparison"

        assert detector.detect_intent(
            "Summarize Apple."
        ) == "summary"

        assert detector.detect_intent(
            "What risks were disclosed?"
        ) == "risk_analysis"

        assert detector.detect_intent(
            "What was revenue?"
        ) == "financial_metric"

        assert detector.detect_intent(
            "What products are offered?"
        ) == "general_question"
