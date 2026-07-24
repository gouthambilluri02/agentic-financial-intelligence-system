"""
Tests for FinancialAgent helper methods.

Covered methods:

- _build_context()
- _extract_llm_answer()
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.agents.financial_agent import FinancialAgent


class TestBuildContext:
    """Tests for FinancialAgent._build_context()."""

    def test_empty_chunks_returns_empty_string(self) -> None:
        result = FinancialAgent._build_context([])

        assert result == ""

    def test_single_chunk_builds_context(self) -> None:
        chunk = {
            "content": "Apple revenue increased.",
            "distance": 0.15,
            "metadata": {
                "company": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "document_type": "10-K",
                "source_file": "apple.pdf",
                "page": 5,
            },
        }

        result = FinancialAgent._build_context([chunk])

        assert "Context 1" in result
        assert "Company: Apple" in result
        assert "Ticker: AAPL" in result
        assert "Fiscal year: 2025" in result
        assert "Document type: 10-K" in result
        assert "Source file: apple.pdf" in result
        assert "Page: 6" in result
        assert "Retrieval distance: 0.15" in result
        assert "Apple revenue increased." in result

    def test_multiple_chunks_are_joined(self) -> None:
        chunk_one = {
            "content": "Apple revenue.",
            "distance": 0.1,
            "metadata": {
                "company": "Apple",
            },
        }

        chunk_two = {
            "content": "Microsoft revenue.",
            "distance": 0.2,
            "metadata": {
                "company": "Microsoft",
            },
        }

        result = FinancialAgent._build_context(
            [
                chunk_one,
                chunk_two,
            ]
        )

        assert "Context 1" in result
        assert "Context 2" in result
        assert "Company: Apple" in result
        assert "Company: Microsoft" in result
        assert "Apple revenue." in result
        assert "Microsoft revenue." in result

    @pytest.mark.parametrize(
        "invalid_chunk",
        [
            None,
            "invalid",
            123,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_chunk_is_skipped(
        self,
        invalid_chunk: Any,
    ) -> None:
        valid_chunk = {
            "content": "Apple",
            "metadata": {},
        }

        result = FinancialAgent._build_context(
            [
                invalid_chunk,
                valid_chunk,
            ]
        )

        assert "Apple" in result

    def test_all_invalid_chunks_return_empty_string(self) -> None:
        result = FinancialAgent._build_context(
            [
                None,
                "invalid",
                123,
                [],
            ]
        )

        assert result == ""

    def test_missing_metadata_uses_defaults(self) -> None:
        chunk = {
            "content": "Revenue",
            "distance": 0.1,
        }

        result = FinancialAgent._build_context([chunk])

        assert "Unknown company" in result
        assert "Unknown ticker" in result
        assert "Unknown year" in result
        assert "Unknown document type" in result
        assert "Unknown source" in result
        assert "Unknown page" in result

    @pytest.mark.parametrize(
        "invalid_metadata",
        [
            None,
            "invalid",
            123,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_metadata_uses_defaults(
        self,
        invalid_metadata: Any,
    ) -> None:
        chunk = {
            "content": "Revenue",
            "distance": 0.1,
            "metadata": invalid_metadata,
        }

        result = FinancialAgent._build_context([chunk])

        assert "Unknown company" in result
        assert "Unknown ticker" in result
        assert "Unknown year" in result
        assert "Unknown document type" in result
        assert "Unknown source" in result
        assert "Unknown page" in result

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (12345, "12345"),
            (4.5, "4.5"),
            (True, "True"),
            (None, "None"),
            (["revenue"], "['revenue']"),
            ({"value": 100}, "{'value': 100}"),
        ],
    )
    def test_non_string_content_is_converted_to_string(
        self,
        content: Any,
        expected: str,
    ) -> None:
        chunk = {
            "content": content,
            "distance": 0.1,
            "metadata": {},
        }

        result = FinancialAgent._build_context([chunk])

        assert expected in result

    def test_missing_content_defaults_to_empty_string(self) -> None:
        chunk = {
            "distance": 0.1,
            "metadata": {},
        }

        result = FinancialAgent._build_context([chunk])

        assert "Content:\n" in result

    @pytest.mark.parametrize(
        ("stored_page", "expected_page"),
        [
            (0, 1),
            (1, 2),
            (5, 6),
            (-1, 0),
        ],
    )
    def test_integer_page_number_is_incremented(
        self,
        stored_page: int,
        expected_page: int,
    ) -> None:
        chunk = {
            "content": "Revenue",
            "distance": 0.1,
            "metadata": {
                "page": stored_page,
            },
        }

        result = FinancialAgent._build_context([chunk])

        assert f"Page: {expected_page}" in result

    @pytest.mark.parametrize(
        "stored_page",
        [
            "Appendix",
            "5",
            5.5,
            None,
            [],
            {},
        ],
    )
    def test_non_integer_page_is_not_incremented(
        self,
        stored_page: Any,
    ) -> None:
        chunk = {
            "content": "Revenue",
            "distance": 0.1,
            "metadata": {
                "page": stored_page,
            },
        }

        result = FinancialAgent._build_context([chunk])

        assert f"Page: {stored_page}" in result

    def test_missing_distance_uses_default(self) -> None:
        chunk = {
            "content": "Revenue",
            "metadata": {},
        }

        result = FinancialAgent._build_context([chunk])

        assert "Retrieval distance: Unknown distance" in result

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (0.0, "0.0"),
            (0.25, "0.25"),
            (1, "1"),
            ("unknown", "unknown"),
            (None, "None"),
        ],
    )
    def test_distance_value_is_rendered(
        self,
        distance: Any,
        expected: str,
    ) -> None:
        chunk = {
            "content": "Revenue",
            "distance": distance,
            "metadata": {},
        }

        result = FinancialAgent._build_context([chunk])

        assert f"Retrieval distance: {expected}" in result

    def test_context_sections_are_separated_by_blank_line(self) -> None:
        chunks = [
            {
                "content": "Apple",
                "metadata": {},
            },
            {
                "content": "Microsoft",
                "metadata": {},
            },
        ]

        result = FinancialAgent._build_context(chunks)

        assert "\n\nContext 2" in result


class TestExtractLlmAnswer:
    """Tests for FinancialAgent._extract_llm_answer()."""

    def test_valid_dictionary_response_returns_cleaned_answer(
        self,
    ) -> None:
        response = {
            "message": {
                "content": "  Apple revenue increased.  ",
            }
        }

        result = FinancialAgent._extract_llm_answer(response)

        assert result == "Apple revenue increased."

    def test_valid_object_response_returns_cleaned_answer(
        self,
    ) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(
                content="  Microsoft disclosed several risks.  "
            )
        )

        result = FinancialAgent._extract_llm_answer(response)

        assert result == "Microsoft disclosed several risks."

    def test_dictionary_response_with_missing_message_returns_invalid_answer(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer({})

        assert result == (
            "The language model returned an invalid answer."
        )

    def test_dictionary_response_with_empty_message_returns_invalid_answer(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": {},
            }
        )

        assert result == (
            "The language model returned an invalid answer."
        )

    def test_dictionary_response_with_none_content_returns_invalid_answer(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": {
                    "content": None,
                },
            }
        )

        assert result == (
            "The language model returned an invalid answer."
        )

    @pytest.mark.parametrize(
        "invalid_content",
        [
            123,
            4.5,
            True,
            [],
            {},
            (),
        ],
    )
    def test_dictionary_response_with_non_string_content_returns_invalid_answer(
        self,
        invalid_content: Any,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": {
                    "content": invalid_content,
                },
            }
        )

        assert result == (
            "The language model returned an invalid answer."
        )

    @pytest.mark.parametrize(
        "blank_content",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            " \n\t ",
        ],
    )
    def test_blank_string_content_returns_invalid_answer(
        self,
        blank_content: str,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": {
                    "content": blank_content,
                },
            }
        )

        assert result == (
            "The language model returned an invalid answer."
        )

    def test_object_without_message_returns_unexpected_response(
        self,
    ) -> None:
        response = SimpleNamespace()

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "The language model returned an unexpected response."
        )

    def test_object_with_message_without_content_returns_unexpected_response(
        self,
    ) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace()
        )

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "The language model returned an unexpected response."
        )

    def test_none_response_returns_unexpected_response(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(None)

        assert result == (
            "The language model returned an unexpected response."
        )

    @pytest.mark.parametrize(
        "invalid_response",
        [
            123,
            4.5,
            True,
            [],
            (),
            "plain string",
        ],
    )
    def test_invalid_response_type_returns_unexpected_response(
        self,
        invalid_response: Any,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            invalid_response
        )

        assert result == (
            "The language model returned an unexpected response."
        )

    def test_dictionary_message_as_none_returns_unexpected_response(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": None,
            }
        )

        assert result == (
            "The language model returned an unexpected response."
        )

    def test_dictionary_message_as_string_returns_unexpected_response(
        self,
    ) -> None:
        result = FinancialAgent._extract_llm_answer(
            {
                "message": "invalid",
            }
        )

        assert result == (
            "The language model returned an unexpected response."
        )

    def test_multiline_answer_is_preserved_except_outer_whitespace(
        self,
    ) -> None:
        response = {
            "message": {
                "content": (
                    "\nRevenue increased.\n"
                    "Operating margin improved.\n"
                ),
            }
        }

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "Revenue increased.\n"
            "Operating margin improved."
        )

    def test_internal_whitespace_is_preserved(
        self,
    ) -> None:
        response = {
            "message": {
                "content": "Apple  revenue   increased.",
            }
        }

        result = FinancialAgent._extract_llm_answer(response)

        assert result == "Apple  revenue   increased."

    def test_unicode_answer_is_returned(self) -> None:
        response = {
            "message": {
                "content": (
                    "Revenue grew by 12.5% — stronger than expected."
                ),
            }
        }

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "Revenue grew by 12.5% — stronger than expected."
        )

    def test_only_outer_spaces_are_removed(self) -> None:
        response = {
            "message": {
                "content": "   Revenue grew by 10%.   ",
            }
        }

        result = FinancialAgent._extract_llm_answer(response)

        assert result == "Revenue grew by 10%."

    def test_object_response_with_none_content_returns_invalid_answer(
        self,
    ) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(
                content=None
            )
        )

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "The language model returned an invalid answer."
        )

    def test_object_response_with_blank_content_returns_invalid_answer(
        self,
    ) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(
                content="   "
            )
        )

        result = FinancialAgent._extract_llm_answer(response)

        assert result == (
            "The language model returned an invalid answer."
        )