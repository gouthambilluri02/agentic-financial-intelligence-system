"""
Tests for AnswerEvaluator.

Covered behavior:

- should_retry()
- _get_minimum_chunks()
- _contains_metric_terms()
- Missing companies
- Minimum chunk requirements
- Metric keyword detection
- Unknown metric behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.answer_evaluator import AnswerEvaluator


@pytest.fixture
def evaluator() -> AnswerEvaluator:
    """Return a fresh AnswerEvaluator instance."""

    return AnswerEvaluator()


class TestShouldRetry:
    """Tests for AnswerEvaluator.should_retry()."""

    def test_empty_chunks_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[],
            plan={},
        )

        assert result is True

    @pytest.mark.parametrize(
        "chunks",
        [
            [],
            (),
            None,
            "",
            0,
            False,
        ],
    )
    def test_any_falsy_chunks_value_requires_retry(
        self,
        evaluator: AnswerEvaluator,
        chunks: Any,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=chunks,  # type: ignore[arg-type]
            plan={},
        )

        assert result is True

    def test_general_question_with_one_chunk_does_not_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "General company information.",
                    "metadata": {},
                }
            ],
            plan={
                "companies": [],
                "intent": "general_question",
                "metric": None,
            },
        )

        assert result is False

    def test_missing_company_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Microsoft reported revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "More Microsoft revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "Additional Microsoft details.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "Further Microsoft details.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
            ],
            plan={
                "companies": [
                    "Apple",
                    "Microsoft",
                ],
                "intent": "comparison",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_all_requested_companies_present_can_continue(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple reported revenue.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
                {
                    "content": "More Apple revenue.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
                {
                    "content": "Microsoft reported revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "More Microsoft revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
            ],
            plan={
                "companies": [
                    "Apple",
                    "Microsoft",
                ],
                "intent": "comparison",
                "metric": "revenue",
            },
        )

        assert result is False

    def test_company_matching_is_case_sensitive(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple reported revenue.",
                    "metadata": {
                        "company": "apple",
                    },
                },
                {
                    "content": "More Apple revenue.",
                    "metadata": {
                        "company": "apple",
                    },
                },
            ],
            plan={
                "companies": ["Apple"],
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_missing_metadata_results_in_missing_company(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple revenue.",
                },
                {
                    "content": "More Apple revenue.",
                },
            ],
            plan={
                "companies": ["Apple"],
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_too_few_chunks_for_comparison_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple revenue.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
                {
                    "content": "Microsoft revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "More revenue information.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
            ],
            plan={
                "companies": [
                    "Apple",
                    "Microsoft",
                ],
                "intent": "comparison",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_comparison_with_four_chunks_is_sufficient(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple revenue.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
                {
                    "content": "More Apple revenue.",
                    "metadata": {
                        "company": "Apple",
                    },
                },
                {
                    "content": "Microsoft revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
                {
                    "content": "More Microsoft revenue.",
                    "metadata": {
                        "company": "Microsoft",
                    },
                },
            ],
            plan={
                "companies": [
                    "Apple",
                    "Microsoft",
                ],
                "intent": "comparison",
                "metric": "revenue",
            },
        )

        assert result is False

    def test_comparison_with_three_companies_requires_six_chunks(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        chunks = [
            {
                "content": "Revenue information.",
                "metadata": {
                    "company": company,
                },
            }
            for company in [
                "Apple",
                "Microsoft",
                "NVIDIA",
                "Apple",
                "Microsoft",
            ]
        ]

        result = evaluator.should_retry(
            retrieved_chunks=chunks,
            plan={
                "companies": [
                    "Apple",
                    "Microsoft",
                    "NVIDIA",
                ],
                "intent": "comparison",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_summary_with_three_chunks_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {"content": "One.", "metadata": {}},
                {"content": "Two.", "metadata": {}},
                {"content": "Three.", "metadata": {}},
            ],
            plan={
                "intent": "summary",
            },
        )

        assert result is True

    def test_summary_with_four_chunks_does_not_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {"content": "One.", "metadata": {}},
                {"content": "Two.", "metadata": {}},
                {"content": "Three.", "metadata": {}},
                {"content": "Four.", "metadata": {}},
            ],
            plan={
                "intent": "summary",
            },
        )

        assert result is False

    def test_risk_analysis_with_two_chunks_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {"content": "Risk one.", "metadata": {}},
                {"content": "Risk two.", "metadata": {}},
            ],
            plan={
                "intent": "risk_analysis",
            },
        )

        assert result is True

    def test_risk_analysis_with_three_chunks_does_not_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {"content": "Risk one.", "metadata": {}},
                {"content": "Risk two.", "metadata": {}},
                {"content": "Risk three.", "metadata": {}},
            ],
            plan={
                "intent": "risk_analysis",
            },
        )

        assert result is False

    def test_metric_question_with_one_chunk_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Revenue information.",
                    "metadata": {},
                }
            ],
            plan={
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_metric_question_with_two_matching_chunks_does_not_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Revenue information.",
                    "metadata": {},
                },
                {
                    "content": "Additional revenue details.",
                    "metadata": {},
                },
            ],
            plan={
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is False

    def test_missing_metric_terms_requires_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "Apple makes devices.",
                    "metadata": {},
                },
                {
                    "content": "Apple provides services.",
                    "metadata": {},
                },
            ],
            plan={
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is True

    def test_unknown_metric_does_not_require_keyword_match(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "General information.",
                    "metadata": {},
                },
                {
                    "content": "More information.",
                    "metadata": {},
                },
            ],
            plan={
                "intent": "financial_metric",
                "metric": "unsupported_metric",
            },
        )

        assert result is False

    def test_missing_plan_fields_use_defaults(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        result = evaluator.should_retry(
            retrieved_chunks=[
                {
                    "content": "General information.",
                    "metadata": {},
                }
            ],
            plan={},
        )

        assert result is False

    def test_extra_chunks_do_not_force_retry(
        self,
        evaluator: AnswerEvaluator,
    ) -> None:
        chunks = [
            {
                "content": "Revenue information.",
                "metadata": {},
            }
            for _ in range(10)
        ]

        result = evaluator.should_retry(
            retrieved_chunks=chunks,
            plan={
                "intent": "financial_metric",
                "metric": "revenue",
            },
        )

        assert result is False


class TestGetMinimumChunks:
    """Tests for _get_minimum_chunks()."""

    @pytest.mark.parametrize(
        ("company_count", "expected"),
        [
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 6),
            (4, 8),
            (10, 20),
        ],
    )
    def test_comparison_minimum_scales_with_company_count(
        self,
        company_count: int,
        expected: int,
    ) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="comparison",
            metric=None,
            company_count=company_count,
        )

        assert result == expected

    def test_summary_requires_four_chunks(self) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="summary",
            metric=None,
            company_count=0,
        )

        assert result == 4

    def test_risk_analysis_requires_three_chunks(self) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="risk_analysis",
            metric=None,
            company_count=0,
        )

        assert result == 3

    @pytest.mark.parametrize(
        "metric",
        [
            "revenue",
            "net_income",
            "eps",
            "unsupported_metric",
            "",
            0,
            False,
        ],
    )
    def test_non_none_metric_requires_two_chunks(
        self,
        metric: Any,
    ) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="general_question",
            metric=metric,
            company_count=0,
        )

        expected = 1 if metric is None else 2
        assert result == expected

    def test_none_metric_general_question_requires_one_chunk(
        self,
    ) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="general_question",
            metric=None,
            company_count=0,
        )

        assert result == 1

    def test_unknown_intent_without_metric_requires_one_chunk(
        self,
    ) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="unknown_intent",
            metric=None,
            company_count=0,
        )

        assert result == 1

    def test_intent_matching_is_case_sensitive(self) -> None:
        result = AnswerEvaluator._get_minimum_chunks(
            intent="COMPARISON",
            metric=None,
            company_count=2,
        )

        assert result == 1


class TestContainsMetricTerms:
    """Tests for _contains_metric_terms()."""

    @pytest.mark.parametrize(
        "content",
        [
            "Revenue increased during the year.",
            "The company reported net sales.",
            "Total sales were higher.",
            "REVENUE increased.",
        ],
    )
    def test_revenue_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="revenue",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Net income increased.",
            "The company reported net earnings.",
            "Profit was higher.",
        ],
    )
    def test_net_income_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="net_income",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Operating income increased.",
            "The company reported operating profit.",
        ],
    )
    def test_operating_income_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="operating_income",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Earnings per share increased.",
            "Diluted earnings per share was 6.43.",
            "Basic earnings per share was 6.11.",
            "EPS increased.",
        ],
    )
    def test_eps_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="eps",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Cash flow increased.",
            (
                "Net cash provided by operating activities "
                "was higher."
            ),
            "Operating activities generated cash.",
        ],
    )
    def test_cash_flow_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="cash_flow",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Total assets increased.",
            "Assets were 100 million.",
        ],
    )
    def test_assets_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="assets",
        )

        assert result is True

    @pytest.mark.parametrize(
        "content",
        [
            "Total liabilities increased.",
            "Liabilities were 50 million.",
        ],
    )
    def test_liabilities_keywords_match(
        self,
        content: str,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": content,
                }
            ],
            metric="liabilities",
        )

        assert result is True

    def test_keywords_can_appear_in_later_chunk(self) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": "General business information.",
                },
                {
                    "content": "Revenue increased.",
                },
            ],
            metric="revenue",
        )

        assert result is True

    def test_combined_content_is_case_insensitive(self) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": "TOTAL ASSETS increased.",
                }
            ],
            metric="assets",
        )

        assert result is True

    def test_missing_content_defaults_to_empty_string(self) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {},
                {
                    "metadata": {},
                },
            ],
            metric="revenue",
        )

        assert result is False

    def test_unrelated_content_returns_false(self) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": "The company develops products.",
                },
                {
                    "content": "It operates globally.",
                },
            ],
            metric="revenue",
        )

        assert result is False

    @pytest.mark.parametrize(
        "metric",
        [
            "unsupported_metric",
            "",
            None,
            123,
        ],
    )
    def test_unknown_or_hashable_invalid_metric_returns_true(
        self,
        metric: Any,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": "Unrelated content.",
                }
            ],
            metric=metric,  # type: ignore[arg-type]
        )

        assert result is True

    @pytest.mark.parametrize(
        "metric",
        [
            [],
            {},
        ],
    )
    def test_unhashable_metric_raises_type_error(
        self,
        metric: Any,
    ) -> None:
        with pytest.raises(TypeError):
            AnswerEvaluator._contains_metric_terms(
                retrieved_chunks=[
                    {
                        "content": "Unrelated content.",
                    }
                ],
                metric=metric,  # type: ignore[arg-type]
            )

    def test_empty_chunk_list_with_known_metric_returns_false(
        self,
    ) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[],
            metric="revenue",
        )

        assert result is False

    def test_matching_is_substring_based(self) -> None:
        result = AnswerEvaluator._contains_metric_terms(
            retrieved_chunks=[
                {
                    "content": "The company discussed profitability.",
                }
            ],
            metric="net_income",
        )

        assert result is True
