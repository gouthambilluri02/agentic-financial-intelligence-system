"""
Tests for FinancialValueExtractor.

Covered methods:

- extract_year_values()
- get_latest_two_values()
- _extract_from_content()
- _normalize_content()
- _extract_table_values()
- _extract_year_headers()
- _extract_financial_numbers()
- _extract_sentence_values()
- _remove_duplicates()
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.financial_value_extractor import (
    FinancialValueExtractor,
)


@pytest.fixture
def extractor() -> FinancialValueExtractor:
    return FinancialValueExtractor()


class TestNormalizeContent:
    """Tests for _normalize_content()."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("", ""),
            ("Revenue", "Revenue"),
            ("  Revenue  ", "Revenue"),
            ("Revenue\t391,035", "Revenue 391035"),
            ("$391,035", "391035"),
            ("$1,234.56", "1234.56"),
            ("Revenue\n\n391,035", "Revenue 391035"),
            ("Revenue    391,035", "Revenue 391035"),
            ("\t$391,035\t", "391035"),
        ],
    )
    def test_normalizes_content(
        self,
        content: str,
        expected: str,
    ) -> None:
        result = FinancialValueExtractor._normalize_content(content)

        assert result == expected

    def test_preserves_negative_sign(self) -> None:
        result = FinancialValueExtractor._normalize_content(
            "Net income $(1,250)"
        )

        assert "1250" in result

    def test_preserves_decimal_point(self) -> None:
        result = FinancialValueExtractor._normalize_content(
            "EPS $6.43"
        )

        assert result == "EPS 6.43"


class TestExtractYearHeaders:
    """Tests for _extract_year_headers()."""

    def test_extracts_years_in_displayed_order(self) -> None:
        result = FinancialValueExtractor._extract_year_headers(
            "2024 2023 2022"
        )

        assert result == [2024, 2023, 2022]

    def test_removes_duplicate_years(self) -> None:
        result = FinancialValueExtractor._extract_year_headers(
            "2024 2023 2024 2022 2023"
        )

        assert result == [2024, 2023, 2022]

    def test_returns_empty_list_without_years(self) -> None:
        result = FinancialValueExtractor._extract_year_headers(
            "No fiscal year was provided."
        )

        assert result == []

    @pytest.mark.parametrize(
        "content",
        [
            "1999 2010",
            "2100 2024",
            "24 2024",
            "FY24 2024",
        ],
    )
    def test_only_extracts_four_digit_20xx_years(
        self,
        content: str,
    ) -> None:
        result = FinancialValueExtractor._extract_year_headers(
            content
        )

        assert all(2000 <= year <= 2099 for year in result)

    def test_extracts_year_from_sentence(self) -> None:
        result = FinancialValueExtractor._extract_year_headers(
            "Revenue increased in 2024 compared with 2023."
        )

        assert result == [2024, 2023]


class TestExtractFinancialNumbers:
    """Tests for _extract_financial_numbers()."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("391035 383285 394328", [391035.0, 383285.0, 394328.0]),
            ("6.43 6.11 5.89", [6.43, 6.11, 5.89]),
            ("-100 200 -300", [-100.0, 200.0, -300.0]),
            ("0 0.0 1", [0.0, 0.0, 1.0]),
            ("", []),
        ],
    )
    def test_extracts_numeric_values(
        self,
        text: str,
        expected: list[float],
    ) -> None:
        result = FinancialValueExtractor._extract_financial_numbers(
            text
        )

        assert result == expected

    def test_ignores_non_numeric_words(self) -> None:
        result = FinancialValueExtractor._extract_financial_numbers(
            "was approximately 391035 compared with 383285"
        )

        assert result == [391035.0, 383285.0]

    def test_extracts_values_after_normalization(self) -> None:
        normalized = FinancialValueExtractor._normalize_content(
            "$391,035 $383,285"
        )

        result = FinancialValueExtractor._extract_financial_numbers(
            normalized
        )

        assert result == [391035.0, 383285.0]


class TestExtractTableValues:
    """Tests for _extract_table_values()."""

    @pytest.mark.parametrize(
        ("metric", "label"),
        [
            ("revenue", "Total net sales"),
            ("revenue", "Net sales"),
            ("revenue", "Total revenue"),
            ("revenue", "Revenue"),
            ("net_income", "Net income"),
            ("net_income", "Net earnings"),
            ("operating_income", "Operating income"),
            ("operating_income", "Operating profit"),
            ("eps", "Diluted earnings per share"),
            ("eps", "Basic earnings per share"),
            ("eps", "Earnings per share"),
            ("eps", "Diluted EPS"),
            ("eps", "Basic EPS"),
        ],
    )
    def test_extracts_supported_metric_table(
        self,
        extractor: FinancialValueExtractor,
        metric: str,
        label: str,
    ) -> None:
        content = (
            f"Year Ended 2024 2023 2022 "
            f"{label} 300 200 100"
        )

        result = extractor._extract_table_values(
            content=content,
            metric=metric,
        )

        assert result == [
            {"year": 2024, "value": 300.0},
            {"year": 2023, "value": 200.0},
            {"year": 2022, "value": 100.0},
        ]

    def test_metric_matching_is_case_insensitive(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content=(
                "YEAR ENDED 2024 2023 "
                "TOTAL NET SALES 200 100"
            ),
            metric="revenue",
        )

        assert result == [
            {"year": 2024, "value": 200.0},
            {"year": 2023, "value": 100.0},
        ]

    def test_unknown_metric_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content="2024 2023 Revenue 200 100",
            metric="unknown_metric",
        )

        assert result == []

    def test_missing_metric_label_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content="2024 2023 Gross profit 200 100",
            metric="revenue",
        )

        assert result == []

    def test_less_than_two_years_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content="2024 Revenue 200",
            metric="revenue",
        )

        assert result == []

    def test_insufficient_values_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content="2024 2023 2022 Revenue 200 100",
            metric="revenue",
        )

        assert result == []

    def test_extra_values_are_ignored_after_year_count(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content="2024 2023 Revenue 200 100 50",
            metric="revenue",
        )

        assert result == [
            {"year": 2024, "value": 200.0},
            {"year": 2023, "value": 100.0},
        ]

    def test_duplicate_year_headers_are_removed(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content=(
                "2024 2023 2024 "
                "Revenue 200 100"
            ),
            metric="revenue",
        )

        assert result == [
            {"year": 2024, "value": 200.0},
            {"year": 2023, "value": 100.0},
        ]

    def test_decimal_values_are_supported(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content=(
                "2024 2023 "
                "Diluted earnings per share 6.43 6.11"
            ),
            metric="eps",
        )

        assert result == [
            {"year": 2024, "value": 6.43},
            {"year": 2023, "value": 6.11},
        ]

    def test_negative_values_are_supported(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_table_values(
            content=(
                "2024 2023 "
                "Operating income -200 100"
            ),
            metric="operating_income",
        )

        assert result == [
            {"year": 2024, "value": -200.0},
            {"year": 2023, "value": 100.0},
        ]


class TestExtractSentenceValues:
    """Tests for _extract_sentence_values()."""

    def test_metric_then_year_then_value(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="Revenue in 2024 was 391035.",
            metric="revenue",
        )

        assert {"year": 2024, "value": 391035.0} in result

    def test_year_then_metric_then_value(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="In 2024, revenue was 391035.",
            metric="revenue",
        )

        assert {"year": 2024, "value": 391035.0} in result

    def test_metric_then_value_then_year(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="Revenue was 391035 in 2024.",
            metric="revenue",
        )

        assert {"year": 2024, "value": 391035.0} in result

    @pytest.mark.parametrize(
        ("metric", "sentence", "year", "value"),
        [
            (
                "net_income",
                "Net income in 2024 was 93736.",
                2024,
                93736.0,
            ),
            (
                "operating_income",
                "Operating profit was 12345 in 2023.",
                2023,
                12345.0,
            ),
            (
                "eps",
                "Diluted EPS in 2024 was 6.43.",
                2024,
                6.43,
            ),
        ],
    )
    def test_supported_metrics_in_sentences(
        self,
        extractor: FinancialValueExtractor,
        metric: str,
        sentence: str,
        year: int,
        value: float,
    ) -> None:
        result = extractor._extract_sentence_values(
            content=sentence,
            metric=metric,
        )

        assert {"year": year, "value": value} in result

    def test_unknown_metric_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="Gross profit in 2024 was 100.",
            metric="gross_profit",
        )

        assert result == []

    def test_sentence_without_metric_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="The value in 2024 was 100.",
            metric="revenue",
        )

        assert result == []

    def test_sentence_without_year_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content="Revenue was 100.",
            metric="revenue",
        )

        assert result == []

    def test_duplicate_sentence_matches_are_removed(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_sentence_values(
            content=(
                "Revenue was 100 in 2024. "
                "Revenue was 100 in 2024."
            ),
            metric="revenue",
        )

        assert result.count(
            {"year": 2024, "value": 100.0}
        ) == 1


class TestExtractFromContent:
    """Tests for _extract_from_content()."""

    @pytest.mark.parametrize(
        "content",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
        ],
    )
    def test_blank_content_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
        content: str,
    ) -> None:
        result = extractor._extract_from_content(
            content=content,
            metric="revenue",
        )

        assert result == []

    def test_prefers_table_extraction_when_available(
        self,
        extractor: FinancialValueExtractor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        table_result = [
            {"year": 2024, "value": 200.0},
            {"year": 2023, "value": 100.0},
        ]

        sentence_called = False

        def fake_table_values(
            *,
            content: str,
            metric: str,
        ) -> list[dict]:
            return table_result

        def fake_sentence_values(
            *,
            content: str,
            metric: str,
        ) -> list[dict]:
            nonlocal sentence_called
            sentence_called = True
            return []

        monkeypatch.setattr(
            extractor,
            "_extract_table_values",
            fake_table_values,
        )
        monkeypatch.setattr(
            extractor,
            "_extract_sentence_values",
            fake_sentence_values,
        )

        result = extractor._extract_from_content(
            content="Revenue data",
            metric="revenue",
        )

        assert result == table_result
        assert sentence_called is False

    def test_falls_back_to_sentence_extraction(
        self,
        extractor: FinancialValueExtractor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentence_result = [
            {"year": 2024, "value": 200.0},
        ]

        monkeypatch.setattr(
            extractor,
            "_extract_table_values",
            lambda **_: [],
        )
        monkeypatch.setattr(
            extractor,
            "_extract_sentence_values",
            lambda **_: sentence_result,
        )

        result = extractor._extract_from_content(
            content="Revenue was 200 in 2024.",
            metric="revenue",
        )

        assert result == sentence_result

    def test_normalizes_content_before_extraction(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor._extract_from_content(
            content=(
                "Year Ended\t2024\t2023 "
                "Total net sales\t$391,035\t$383,285"
            ),
            metric="revenue",
        )

        assert result == [
            {"year": 2024, "value": 391035.0},
            {"year": 2023, "value": 383285.0},
        ]


class TestExtractYearValues:
    """Tests for extract_year_values()."""

    def test_extracts_values_with_metadata(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        chunks = [
            {
                "content": (
                    "Year Ended 2024 2023 "
                    "Total net sales 391035 383285"
                ),
                "metadata": {
                    "company": "Apple",
                    "source_file": "apple_2024_10k.pdf",
                    "page": 37,
                },
            }
        ]

        result = extractor.extract_year_values(
            retrieved_chunks=chunks,
            metric="revenue",
        )

        assert result == [
            {
                "company": "Apple",
                "metric": "revenue",
                "year": 2024,
                "value": 391035.0,
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
            {
                "company": "Apple",
                "metric": "revenue",
                "year": 2023,
                "value": 383285.0,
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
        ]

    def test_missing_metadata_uses_defaults(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.extract_year_values(
            retrieved_chunks=[
                {
                    "content": (
                        "2024 2023 Revenue 200 100"
                    ),
                }
            ],
            metric="revenue",
        )

        assert result == [
            {
                "company": "Unknown company",
                "metric": "revenue",
                "year": 2024,
                "value": 200.0,
                "source_file": "Unknown source",
                "page": "Unknown page",
            },
            {
                "company": "Unknown company",
                "metric": "revenue",
                "year": 2023,
                "value": 100.0,
                "source_file": "Unknown source",
                "page": "Unknown page",
            },
        ]

    def test_empty_chunks_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.extract_year_values(
            retrieved_chunks=[],
            metric="revenue",
        )

        assert result == []

    def test_unsupported_metric_returns_empty_list(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.extract_year_values(
            retrieved_chunks=[
                {
                    "content": (
                        "2024 2023 Revenue 200 100"
                    ),
                    "metadata": {
                        "company": "Apple",
                    },
                }
            ],
            metric="unsupported",
        )

        assert result == []

    def test_combines_values_from_multiple_chunks(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.extract_year_values(
            retrieved_chunks=[
                {
                    "content": (
                        "2024 2023 Revenue 200 100"
                    ),
                    "metadata": {
                        "company": "Apple",
                        "source_file": "apple.pdf",
                        "page": 1,
                    },
                },
                {
                    "content": (
                        "2024 2023 Revenue 300 250"
                    ),
                    "metadata": {
                        "company": "Microsoft",
                        "source_file": "microsoft.pdf",
                        "page": 2,
                    },
                },
            ],
            metric="revenue",
        )

        assert len(result) == 4
        assert result[0]["company"] == "Apple"
        assert result[2]["company"] == "Microsoft"

    def test_exact_duplicate_values_are_removed(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        chunk = {
            "content": "2024 2023 Revenue 200 100",
            "metadata": {
                "company": "Apple",
                "source_file": "apple.pdf",
                "page": 1,
            },
        }

        result = extractor.extract_year_values(
            retrieved_chunks=[chunk, chunk],
            metric="revenue",
        )

        assert len(result) == 2


class TestGetLatestTwoValues:
    """Tests for get_latest_two_values()."""

    def test_returns_latest_two_values_for_company(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        values = [
            {"company": "Apple", "year": 2022, "value": 100},
            {"company": "Apple", "year": 2024, "value": 300},
            {"company": "Apple", "year": 2023, "value": 200},
        ]

        result = extractor.get_latest_two_values(
            extracted_values=values,
            company="Apple",
        )

        assert result == [
            {"company": "Apple", "year": 2024, "value": 300},
            {"company": "Apple", "year": 2023, "value": 200},
        ]

    def test_filters_values_by_company(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        values = [
            {"company": "Apple", "year": 2024, "value": 300},
            {"company": "Microsoft", "year": 2025, "value": 500},
            {"company": "Apple", "year": 2023, "value": 200},
        ]

        result = extractor.get_latest_two_values(
            extracted_values=values,
            company="Apple",
        )

        assert all(item["company"] == "Apple" for item in result)

    def test_returns_empty_list_for_unknown_company(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.get_latest_two_values(
            extracted_values=[
                {"company": "Apple", "year": 2024, "value": 300}
            ],
            company="Microsoft",
        )

        assert result == []

    def test_ignores_non_integer_years(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        result = extractor.get_latest_two_values(
            extracted_values=[
                {"company": "Apple", "year": "2024", "value": 300},
                {"company": "Apple", "year": None, "value": 200},
                {"company": "Apple", "year": 2023, "value": 100},
            ],
            company="Apple",
        )

        assert result == [
            {"company": "Apple", "year": 2023, "value": 100}
        ]

    def test_keeps_first_item_for_duplicate_year(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        first = {
            "company": "Apple",
            "year": 2024,
            "value": 300,
            "source_file": "first.pdf",
        }
        second = {
            "company": "Apple",
            "year": 2024,
            "value": 350,
            "source_file": "second.pdf",
        }

        result = extractor.get_latest_two_values(
            extracted_values=[first, second],
            company="Apple",
        )

        assert result == [first]

    def test_returns_one_value_when_only_one_year_exists(
        self,
        extractor: FinancialValueExtractor,
    ) -> None:
        value = {
            "company": "Apple",
            "year": 2024,
            "value": 300,
        }

        result = extractor.get_latest_two_values(
            extracted_values=[value],
            company="Apple",
        )

        assert result == [value]


class TestRemoveDuplicates:
    """Tests for _remove_duplicates()."""

    def test_empty_list_returns_empty_list(self) -> None:
        assert FinancialValueExtractor._remove_duplicates([]) == []

    def test_exact_duplicates_are_removed(self) -> None:
        item = {
            "company": "Apple",
            "metric": "revenue",
            "year": 2024,
            "value": 300,
            "source_file": "apple.pdf",
            "page": 1,
        }

        result = FinancialValueExtractor._remove_duplicates(
            [item, item.copy()]
        )

        assert result == [item]

    def test_order_is_preserved(self) -> None:
        first = {"year": 2024, "value": 300}
        second = {"year": 2023, "value": 200}

        result = FinancialValueExtractor._remove_duplicates(
            [first, second, first.copy()]
        )

        assert result == [first, second]

    @pytest.mark.parametrize(
        ("field", "first_value", "second_value"),
        [
            ("company", "Apple", "Microsoft"),
            ("metric", "revenue", "net_income"),
            ("year", 2024, 2023),
            ("value", 300, 301),
            ("source_file", "a.pdf", "b.pdf"),
            ("page", 1, 2),
        ],
    )
    def test_items_with_different_key_fields_are_not_duplicates(
        self,
        field: str,
        first_value: Any,
        second_value: Any,
    ) -> None:
        base = {
            "company": "Apple",
            "metric": "revenue",
            "year": 2024,
            "value": 300,
            "source_file": "a.pdf",
            "page": 1,
        }

        first = {
            **base,
            field: first_value,
        }
        second = {
            **base,
            field: second_value,
        }

        result = FinancialValueExtractor._remove_duplicates(
            [first, second]
        )

        assert result == [first, second]

    def test_missing_fields_are_supported(self) -> None:
        result = FinancialValueExtractor._remove_duplicates(
            [
                {"year": 2024, "value": 300},
                {"year": 2024, "value": 300},
            ]
        )

        assert result == [
            {"year": 2024, "value": 300}
        ]
