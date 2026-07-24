"""
Tests for FinancialMetricDetector.

Covered behavior:

- METRICS configuration
- Revenue detection
- Net-income detection
- Operating-income detection
- EPS detection
- Cash-flow detection
- Assets detection
- Liabilities detection
- Case-insensitive matching
- Word-boundary behavior
- Metric precedence
- No-match fallback
- Invalid input behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.financial_metrics import (
    FinancialMetricDetector,
)


@pytest.fixture
def detector() -> FinancialMetricDetector:
    """Return a fresh FinancialMetricDetector instance."""

    return FinancialMetricDetector()


class TestMetricsConfiguration:
    """Tests for the METRICS configuration."""

    def test_metrics_is_dictionary(self) -> None:
        assert isinstance(FinancialMetricDetector.METRICS, dict)

    def test_contains_expected_metric_keys(self) -> None:
        assert list(FinancialMetricDetector.METRICS) == [
            "revenue",
            "net_income",
            "operating_income",
            "eps",
            "cash_flow",
            "assets",
            "liabilities",
        ]

    @pytest.mark.parametrize(
        "metric",
        [
            "revenue",
            "net_income",
            "operating_income",
            "eps",
            "cash_flow",
            "assets",
            "liabilities",
        ],
    )
    def test_each_metric_has_non_empty_keyword_list(
        self,
        metric: str,
    ) -> None:
        keywords = FinancialMetricDetector.METRICS[metric]

        assert isinstance(keywords, list)
        assert keywords
        assert all(
            isinstance(keyword, str) and keyword
            for keyword in keywords
        )

    def test_revenue_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS["revenue"] == [
            "revenue",
            "net sales",
            "sales",
        ]

    def test_net_income_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS["net_income"] == [
            "net income",
            "profit",
            "earnings",
        ]

    def test_operating_income_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS[
            "operating_income"
        ] == [
            "operating income",
            "operating profit",
        ]

    def test_eps_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS["eps"] == [
            "eps",
            "earnings per share",
        ]

    def test_cash_flow_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS["cash_flow"] == [
            "cash flow",
            "operating cash flow",
        ]

    def test_assets_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS["assets"] == [
            "assets",
            "total assets",
        ]

    def test_liabilities_keywords_are_configured(self) -> None:
        assert FinancialMetricDetector.METRICS[
            "liabilities"
        ] == [
            "liabilities",
            "total liabilities",
        ]


class TestMetricDetection:
    """Tests for supported financial metric keywords."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What was Apple's revenue?", "revenue"),
            ("Show net sales.", "revenue"),
            ("How much sales did the company report?", "revenue"),
            ("What was net income?", "net_income"),
            ("What was the company's profit?", "net_income"),
            ("Show annual earnings.", "net_income"),
            ("What was operating income?", "operating_income"),
            ("Show operating profit.", "net_income"),
            ("What was EPS?", "eps"),
            ("Show earnings per share.", "net_income"),
            ("What was cash flow?", "cash_flow"),
            ("Show operating cash flow.", "cash_flow"),
            ("What were assets?", "assets"),
            ("Show total assets.", "assets"),
            ("What were liabilities?", "liabilities"),
            ("Show total liabilities.", "liabilities"),
        ],
    )
    def test_detects_supported_metric_terms(
        self,
        detector: FinancialMetricDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_metric(question) == expected

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("REVENUE", "revenue"),
            ("Net Sales", "revenue"),
            ("NET INCOME", "net_income"),
            ("Operating Income", "operating_income"),
            ("ePs", "eps"),
            ("Cash Flow", "cash_flow"),
            ("TOTAL ASSETS", "assets"),
            ("Total Liabilities", "liabilities"),
        ],
    )
    def test_matching_is_case_insensitive(
        self,
        detector: FinancialMetricDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_metric(question) == expected

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Revenue?", "revenue"),
            ("(net income)", "net_income"),
            ("operating income:", "operating_income"),
            ("EPS.", "eps"),
            ("cash flow/analysis", "cash_flow"),
            ("assets,", "assets"),
            ("liabilities;", "liabilities"),
        ],
    )
    def test_keywords_match_next_to_punctuation(
        self,
        detector: FinancialMetricDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_metric(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "prerevenue",
            "revenueless",
            "netsales",
            "salesforce",
            "netincome",
            "profitable",
            "earningscall",
            "operatingincome",
            "operatingprofit",
            "epstein",
            "epsgrowth",
            "earningspershare",
            "cashflow",
            "operatingcashflow",
            "asset",
            "assetbased",
            "totalassets",
            "liability",
            "liabilitiesrelated",
            "totalliabilities",
        ],
    )
    def test_keywords_do_not_match_inside_larger_words(
        self,
        detector: FinancialMetricDetector,
        question: str,
    ) -> None:
        assert detector.detect_metric(question) is None


class TestMetricPrecedence:
    """Tests for metric priority based on dictionary order."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Compare revenue and net income.", "revenue"),
            (
                "Compare net income and operating income.",
                "net_income",
            ),
            ("Compare operating income and EPS.", "operating_income"),
            ("Compare EPS and cash flow.", "eps"),
            ("Compare cash flow and assets.", "cash_flow"),
            ("Compare assets and liabilities.", "assets"),
        ],
    )
    def test_returns_first_metric_in_configured_order(
        self,
        detector: FinancialMetricDetector,
        question: str,
        expected: str,
    ) -> None:
        assert detector.detect_metric(question) == expected

    def test_first_matching_metric_is_returned(
        self,
        detector: FinancialMetricDetector,
    ) -> None:
        question = (
            "Discuss revenue, net income, operating income, EPS, "
            "cash flow, assets, and liabilities."
        )

        assert detector.detect_metric(question) == "revenue"


class TestOverlappingKeywordPrecedence:
    """Tests for overlapping phrases and configured metric order."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was operating profit?",
            "Show annual operating profit.",
            "OPERATING PROFIT",
        ],
    )
    def test_operating_profit_matches_net_income_first(
        self,
        detector: FinancialMetricDetector,
        question: str,
    ) -> None:
        assert detector.detect_metric(question) == "net_income"

    @pytest.mark.parametrize(
        "question",
        [
            "What were earnings per share?",
            "Show diluted earnings per share.",
            "EARNINGS PER SHARE",
        ],
    )
    def test_earnings_per_share_matches_net_income_first(
        self,
        detector: FinancialMetricDetector,
        question: str,
    ) -> None:
        assert detector.detect_metric(question) == "net_income"

    def test_free_cash_flow_matches_cash_flow(
        self,
        detector: FinancialMetricDetector,
    ) -> None:
        assert detector.detect_metric(
            "What was free cash flow?"
        ) == "cash_flow"


class TestNoMatchBehavior:
    """Tests for questions without supported metrics."""

    @pytest.mark.parametrize(
        "question",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            "How many employees does Apple have?",
            "Who is the CEO?",
            "What products does Microsoft sell?",
            "Describe the company's strategy.",
            "What risks were disclosed?",
            "Summarize the annual report.",
            "Compare Apple and Microsoft.",
            "What was gross margin?",
            "What was operating margin?",
            "How much debt was reported?",
            "?!.,;:",
            "2024 2023 100",
            "Компания туралы ақпарат беріңіз.",
        ],
    )
    def test_returns_none_when_no_supported_metric_matches(
        self,
        detector: FinancialMetricDetector,
        question: str,
    ) -> None:
        assert detector.detect_metric(question) is None


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
        detector: FinancialMetricDetector,
        question: Any,
    ) -> None:
        with pytest.raises(AttributeError):
            detector.detect_metric(question)  # type: ignore[arg-type]


class TestDetectorInstance:
    """Basic construction and repeatability tests."""

    def test_detector_can_be_instantiated(self) -> None:
        detector = FinancialMetricDetector()

        assert isinstance(detector, FinancialMetricDetector)

    def test_same_question_returns_same_metric(
        self,
        detector: FinancialMetricDetector,
    ) -> None:
        question = "What was Apple's revenue?"

        first = detector.detect_metric(question)
        second = detector.detect_metric(question)

        assert first == second == "revenue"

    def test_detector_is_stateless_across_questions(
        self,
        detector: FinancialMetricDetector,
    ) -> None:
        assert detector.detect_metric(
            "What was revenue?"
        ) == "revenue"

        assert detector.detect_metric(
            "What was net income?"
        ) == "net_income"

        assert detector.detect_metric(
            "What was operating income?"
        ) == "operating_income"

        assert detector.detect_metric(
            "What was EPS?"
        ) == "eps"

        assert detector.detect_metric(
            "What was cash flow?"
        ) == "cash_flow"

        assert detector.detect_metric(
            "What were assets?"
        ) == "assets"

        assert detector.detect_metric(
            "What were liabilities?"
        ) == "liabilities"

        assert detector.detect_metric(
            "Who is the CEO?"
        ) is None
