"""
Tests for CompanyDetector.

Covered behavior:

- COMPANY_ALIASES configuration
- Canonical company-name detection
- Ticker-symbol detection
- Alternate alias detection
- Case-insensitive matching
- Word-boundary behavior
- Multiple-company detection
- Duplicate suppression
- Detection order
- No-match behavior
- Invalid input behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.company_detector import (
    COMPANY_ALIASES,
    CompanyDetector,
)


@pytest.fixture
def detector() -> CompanyDetector:
    """Return a fresh CompanyDetector instance."""

    return CompanyDetector()


class TestCompanyAliasesConfiguration:
    """Tests for COMPANY_ALIASES."""

    def test_aliases_is_dictionary(self) -> None:
        assert isinstance(COMPANY_ALIASES, dict)

    def test_contains_expected_companies(self) -> None:
        assert list(COMPANY_ALIASES) == [
            "Apple",
            "Microsoft",
            "Tesla",
            "Amazon",
            "NVIDIA",
            "Meta",
            "Alphabet",
        ]

    @pytest.mark.parametrize(
        "company",
        [
            "Apple",
            "Microsoft",
            "Tesla",
            "Amazon",
            "NVIDIA",
            "Meta",
            "Alphabet",
        ],
    )
    def test_each_company_has_non_empty_alias_list(
        self,
        company: str,
    ) -> None:
        aliases = COMPANY_ALIASES[company]

        assert isinstance(aliases, list)
        assert aliases
        assert all(
            isinstance(alias, str) and alias
            for alias in aliases
        )

    def test_apple_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Apple"] == [
            "apple",
            "aapl",
            "apple inc",
        ]

    def test_microsoft_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Microsoft"] == [
            "microsoft",
            "msft",
            "microsoft corporation",
        ]

    def test_tesla_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Tesla"] == [
            "tesla",
            "tsla",
            "tesla inc",
        ]

    def test_amazon_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Amazon"] == [
            "amazon",
            "amzn",
            "amazon.com",
        ]

    def test_nvidia_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["NVIDIA"] == [
            "nvidia",
            "nvda",
        ]

    def test_meta_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Meta"] == [
            "meta",
            "meta platforms",
            "facebook",
        ]

    def test_alphabet_aliases_are_configured(self) -> None:
        assert COMPANY_ALIASES["Alphabet"] == [
            "alphabet",
            "google",
            "goog",
            "googl",
        ]


class TestAppleDetection:
    """Tests for Apple aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was Apple's revenue?",
            "Tell me about Apple.",
            "APPLE",
            "How is AAPL performing?",
            "What was AAPL revenue?",
            "What did Apple Inc report?",
        ],
    )
    def test_detects_apple_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Apple"]

    @pytest.mark.parametrize(
        "question",
        [
            "Apple?",
            "(Apple)",
            "AAPL.",
            "Apple Inc:",
        ],
    )
    def test_apple_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Apple"]

    @pytest.mark.parametrize(
        "question",
        [
            "pineapple",
            "apples",
            "aaplex",
            "appleinc",
        ],
    )
    def test_apple_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestMicrosoftDetection:
    """Tests for Microsoft aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What risks did Microsoft disclose?",
            "MICROSOFT",
            "How is MSFT performing?",
            "Show MSFT revenue.",
            "What did Microsoft Corporation report?",
            "Microsoft Corporation earnings.",
        ],
    )
    def test_detects_microsoft_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == [
            "Microsoft"
        ]

    @pytest.mark.parametrize(
        "question",
        [
            "Microsoft?",
            "(MSFT)",
            "Microsoft Corporation.",
        ],
    )
    def test_microsoft_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == [
            "Microsoft"
        ]

    @pytest.mark.parametrize(
        "question",
        [
            "microsofts",
            "msftx",
            "premicrosoft",
            "microsoftcorporation",
        ],
    )
    def test_microsoft_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestTeslaDetection:
    """Tests for Tesla aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was Tesla revenue?",
            "TESLA",
            "How is TSLA performing?",
            "Show TSLA risks.",
            "What did Tesla Inc report?",
        ],
    )
    def test_detects_tesla_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Tesla"]

    @pytest.mark.parametrize(
        "question",
        [
            "Tesla?",
            "(TSLA)",
            "Tesla Inc.",
        ],
    )
    def test_tesla_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Tesla"]

    @pytest.mark.parametrize(
        "question",
        [
            "teslas",
            "tslax",
            "pretesla",
            "teslainc",
        ],
    )
    def test_tesla_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestAmazonDetection:
    """Tests for Amazon aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was Amazon revenue?",
            "AMAZON",
            "How is AMZN performing?",
            "Show AMZN risks.",
            "What did Amazon.com report?",
        ],
    )
    def test_detects_amazon_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Amazon"]

    @pytest.mark.parametrize(
        "question",
        [
            "Amazon?",
            "(AMZN)",
            "Amazon.com.",
        ],
    )
    def test_amazon_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Amazon"]

    @pytest.mark.parametrize(
        "question",
        [
            "amazons",
            "amznx",
            "preamazon",
            "amazoncom",
        ],
    )
    def test_amazon_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestNvidiaDetection:
    """Tests for NVIDIA aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was NVIDIA revenue?",
            "NVIDIA",
            "How is NVDA performing?",
            "Show NVDA risks.",
        ],
    )
    def test_detects_nvidia_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["NVIDIA"]

    @pytest.mark.parametrize(
        "question",
        [
            "NVIDIA?",
            "(NVDA)",
            "NVDA.",
        ],
    )
    def test_nvidia_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["NVIDIA"]

    @pytest.mark.parametrize(
        "question",
        [
            "nvidias",
            "nvdax",
            "prenvidia",
        ],
    )
    def test_nvidia_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestMetaDetection:
    """Tests for Meta aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What risks did Meta disclose?",
            "META",
            "What did Meta Platforms report?",
            "Show Meta Platforms revenue.",
            "What did Facebook report?",
            "FACEBOOK",
        ],
    )
    def test_detects_meta_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Meta"]

    @pytest.mark.parametrize(
        "question",
        [
            "Meta?",
            "(Meta Platforms)",
            "Facebook.",
        ],
    )
    def test_meta_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Meta"]

    @pytest.mark.parametrize(
        "question",
        [
            "metadata",
            "metaverse",
            "facebooking",
            "metaplatforms",
        ],
    )
    def test_meta_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestAlphabetDetection:
    """Tests for Alphabet aliases."""

    @pytest.mark.parametrize(
        "question",
        [
            "What was Alphabet revenue?",
            "ALPHABET",
            "What did Google report?",
            "GOOGLE",
            "How is GOOG performing?",
            "Show GOOGL risks.",
        ],
    )
    def test_detects_alphabet_aliases(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Alphabet"]

    @pytest.mark.parametrize(
        "question",
        [
            "Alphabet?",
            "(Google)",
            "GOOG.",
            "GOOGL:",
        ],
    )
    def test_alphabet_aliases_match_next_to_punctuation(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == ["Alphabet"]

    @pytest.mark.parametrize(
        "question",
        [
            "alphabets",
            "googler",
            "googling",
            "googlx",
            "prealphabet",
        ],
    )
    def test_alphabet_aliases_do_not_match_inside_larger_words(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


class TestCaseInsensitivity:
    """Tests for case-insensitive matching."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("apple", ["Apple"]),
            ("Apple", ["Apple"]),
            ("APPLE", ["Apple"]),
            ("aapl", ["Apple"]),
            ("AaPl", ["Apple"]),
            ("microsoft", ["Microsoft"]),
            ("MSFT", ["Microsoft"]),
            ("tesla", ["Tesla"]),
            ("TSLA", ["Tesla"]),
            ("amazon", ["Amazon"]),
            ("AMZN", ["Amazon"]),
            ("nvidia", ["NVIDIA"]),
            ("NvDa", ["NVIDIA"]),
            ("meta", ["Meta"]),
            ("FACEBOOK", ["Meta"]),
            ("alphabet", ["Alphabet"]),
            ("GOOGLE", ["Alphabet"]),
            ("GoOgL", ["Alphabet"]),
        ],
    )
    def test_matching_is_case_insensitive(
        self,
        detector: CompanyDetector,
        question: str,
        expected: list[str],
    ) -> None:
        assert detector.detect_companies(question) == expected


class TestMultipleCompanyDetection:
    """Tests for questions containing several companies."""

    def test_detects_two_companies(self) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Compare Apple and Microsoft revenue."
        )

        assert result == ["Apple", "Microsoft"]

    def test_detects_companies_from_tickers(self) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Compare AAPL, MSFT, and NVDA."
        )

        assert result == [
            "Apple",
            "Microsoft",
            "NVIDIA",
        ]

    def test_detects_all_supported_companies(self) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Compare Apple, Microsoft, Tesla, Amazon, NVIDIA, "
            "Meta, and Alphabet."
        )

        assert result == [
            "Apple",
            "Microsoft",
            "Tesla",
            "Amazon",
            "NVIDIA",
            "Meta",
            "Alphabet",
        ]

    def test_detection_order_follows_configuration_order(self) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Google, Meta, NVIDIA, Amazon, Tesla, Microsoft, "
            "and Apple."
        )

        assert result == [
            "Apple",
            "Microsoft",
            "Tesla",
            "Amazon",
            "NVIDIA",
            "Meta",
            "Alphabet",
        ]

    def test_question_order_does_not_control_result_order(self) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Microsoft and Apple"
        )

        assert result == ["Apple", "Microsoft"]

    def test_aliases_for_same_company_do_not_duplicate_result(
        self,
    ) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Apple, AAPL, and Apple Inc all refer to the same company."
        )

        assert result == ["Apple"]

    def test_multiple_aliases_for_multiple_companies_do_not_duplicate(
        self,
    ) -> None:
        detector = CompanyDetector()

        result = detector.detect_companies(
            "Apple AAPL Microsoft MSFT Google GOOGL"
        )

        assert result == [
            "Apple",
            "Microsoft",
            "Alphabet",
        ]


class TestNoMatchBehavior:
    """Tests for questions without supported companies."""

    @pytest.mark.parametrize(
        "question",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            "What is the company's revenue?",
            "Compare two companies.",
            "What risks were disclosed?",
            "How many employees are there?",
            "What was the net income?",
            "Tell me about Oracle.",
            "What did IBM report?",
            "How is Netflix performing?",
            "What was Samsung revenue?",
        ],
    )
    def test_returns_empty_list_when_no_company_matches(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []

    def test_punctuation_only_returns_empty_list(
        self,
        detector: CompanyDetector,
    ) -> None:
        assert detector.detect_companies("?!.,;:") == []

    def test_numbers_only_returns_empty_list(
        self,
        detector: CompanyDetector,
    ) -> None:
        assert detector.detect_companies("2024 2023 100") == []

    def test_unicode_text_without_alias_returns_empty_list(
        self,
        detector: CompanyDetector,
    ) -> None:
        assert detector.detect_companies(
            "Компания туралы ақпарат беріңіз."
        ) == []


class TestWordBoundaryBehavior:
    """Tests for regular-expression word boundaries."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Apple/Microsoft", ["Apple", "Microsoft"]),
            ("AAPL-MSFT", ["Apple", "Microsoft"]),
            ("Google,NVIDIA", ["NVIDIA", "Alphabet"]),
            ("Meta;Tesla", ["Tesla", "Meta"]),
            ("Amazon.com", ["Amazon"]),
        ],
    )
    def test_aliases_match_around_non_word_characters(
        self,
        detector: CompanyDetector,
        question: str,
        expected: list[str],
    ) -> None:
        assert detector.detect_companies(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "apple123",
            "123apple",
            "msft2024",
            "2024tsla",
            "nvda_growth",
            "meta_platform",
            "googlecloud",
        ],
    )
    def test_aliases_do_not_match_when_attached_to_word_characters(
        self,
        detector: CompanyDetector,
        question: str,
    ) -> None:
        assert detector.detect_companies(question) == []


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
        detector: CompanyDetector,
        question: Any,
    ) -> None:
        with pytest.raises(AttributeError):
            detector.detect_companies(question)  # type: ignore[arg-type]


class TestDetectorInstance:
    """Basic construction and repeatability tests."""

    def test_detector_can_be_instantiated(self) -> None:
        detector = CompanyDetector()

        assert isinstance(detector, CompanyDetector)

    def test_same_question_returns_same_result(
        self,
        detector: CompanyDetector,
    ) -> None:
        question = "Compare Apple and Microsoft."

        first = detector.detect_companies(question)
        second = detector.detect_companies(question)

        assert first == second == [
            "Apple",
            "Microsoft",
        ]

    def test_detector_is_stateless_across_questions(
        self,
        detector: CompanyDetector,
    ) -> None:
        assert detector.detect_companies(
            "Apple"
        ) == ["Apple"]

        assert detector.detect_companies(
            "Microsoft"
        ) == ["Microsoft"]

        assert detector.detect_companies(
            "Tesla"
        ) == ["Tesla"]

        assert detector.detect_companies(
            "Amazon"
        ) == ["Amazon"]

        assert detector.detect_companies(
            "NVIDIA"
        ) == ["NVIDIA"]

        assert detector.detect_companies(
            "Meta"
        ) == ["Meta"]

        assert detector.detect_companies(
            "Google"
        ) == ["Alphabet"]

        assert detector.detect_companies(
            "Oracle"
        ) == []
