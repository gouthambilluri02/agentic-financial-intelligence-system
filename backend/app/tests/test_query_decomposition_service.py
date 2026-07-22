from __future__ import annotations

import pytest

from backend.app.services.query_decomposition_service import (
    CALCULATION_PATTERNS,
    COMPARISON_PATTERNS,
    CONCLUSION_PATTERNS,
    QueryDecompositionService,
)


@pytest.fixture
def service() -> QueryDecompositionService:
    return QueryDecompositionService()


class TestDecompose:
    def test_retrieval_only_question(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="What was Apple's revenue?",
            companies=["Apple"],
            intent="retrieval",
            metric="revenue",
        )

        assert result["is_complex"] is False
        assert result["reasoning_type"] == "retrieval"
        assert result["calculation_requested"] is False
        assert result["comparison_requested"] is False
        assert result["conclusion_requested"] is False
        assert result["calculated_comparison"] is False
        assert result["companies"] == ["Apple"]
        assert result["metric"] == "revenue"
        assert result["required_capabilities"] == [
            "document_retrieval",
        ]

    def test_single_company_calculation(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Calculate Apple's revenue growth.",
            companies=["Apple"],
            intent="financial_metric",
            metric="revenue",
        )

        assert result["is_complex"] is False
        assert result["reasoning_type"] == "calculation"
        assert result["calculation_requested"] is True
        assert result["comparison_requested"] is False
        assert result["calculated_comparison"] is False
        assert result["companies"] == ["Apple"]
        assert result["required_capabilities"] == [
            "document_retrieval",
            "financial_calculator",
        ]

    def test_absolute_comparison_from_intent(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Show Apple and Microsoft revenue.",
            companies=["Apple", "Microsoft"],
            intent="comparison",
            metric="revenue",
        )

        assert result["reasoning_type"] == "absolute_comparison"
        assert result["comparison_requested"] is True
        assert result["calculation_requested"] is False
        assert result["calculated_comparison"] is False
        assert result["required_capabilities"] == [
            "document_retrieval",
            "company_comparison",
        ]

    def test_absolute_comparison_from_multiple_companies(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Show their latest revenue.",
            companies=["Apple", "Microsoft"],
            intent="retrieval",
            metric="revenue",
        )

        assert result["comparison_requested"] is True
        assert result["reasoning_type"] == "absolute_comparison"

    def test_absolute_comparison_from_question_pattern(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Compare Apple revenue.",
            companies=["Apple"],
            intent="retrieval",
            metric="revenue",
        )

        assert result["comparison_requested"] is True
        assert result["reasoning_type"] == "absolute_comparison"
        assert result["calculated_comparison"] is False

    def test_calculated_comparison(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question=(
                "Compare Apple and Microsoft revenue growth "
                "and identify which company performed better."
            ),
            companies=["Apple", "Microsoft"],
            intent="comparison",
            metric="revenue",
        )

        assert result["is_complex"] is True
        assert result["reasoning_type"] == "calculated_comparison"
        assert result["calculation_requested"] is True
        assert result["comparison_requested"] is True
        assert result["conclusion_requested"] is True
        assert result["calculated_comparison"] is True
        assert result["required_capabilities"] == [
            "document_retrieval",
            "financial_calculator",
            "multi_step_reasoning",
        ]

    def test_calculated_comparison_without_explicit_conclusion(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Compare Apple and Microsoft revenue growth.",
            companies=["Apple", "Microsoft"],
            intent="comparison",
            metric="revenue",
        )

        assert result["calculated_comparison"] is True
        assert result["conclusion_requested"] is False
        assert result["is_complex"] is True

        assert result["subtasks"][-1]["type"] == "conclusion"

    def test_complex_absolute_comparison_with_conclusion(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question=(
                "Compare Apple and Microsoft revenue and "
                "identify the stronger performer."
            ),
            companies=["Apple", "Microsoft"],
            intent="comparison",
            metric="revenue",
        )

        assert result["calculation_requested"] is False
        assert result["comparison_requested"] is True
        assert result["conclusion_requested"] is True
        assert result["calculated_comparison"] is False
        assert result["is_complex"] is True
        assert result["reasoning_type"] == "absolute_comparison"

    def test_non_string_question_is_treated_as_empty(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question=None,  # type: ignore[arg-type]
            companies=["Apple"],
            intent="retrieval",
            metric="revenue",
        )

        assert result["calculation_requested"] is False
        assert result["comparison_requested"] is False
        assert result["conclusion_requested"] is False
        assert result["reasoning_type"] == "retrieval"

    def test_question_is_case_insensitive(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="CALCULATE APPLE REVENUE GROWTH",
            companies=["Apple"],
            intent="financial_metric",
            metric="revenue",
        )

        assert result["calculation_requested"] is True
        assert result["reasoning_type"] == "calculation"

    def test_question_whitespace_is_removed(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="   Calculate Apple revenue growth.   ",
            companies=["Apple"],
            intent="financial_metric",
            metric="revenue",
        )

        assert result["calculation_requested"] is True

    def test_duplicate_companies_are_removed(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Compare revenue growth.",
            companies=[
                "Apple",
                "apple",
                " Microsoft ",
                "MICROSOFT",
            ],
            intent="comparison",
            metric="revenue",
        )

        assert result["companies"] == [
            "Apple",
            "Microsoft",
        ]

    def test_invalid_company_values_are_removed(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Show revenue.",
            companies=[
                "Apple",
                "",
                "   ",
                None,
                123,
                "Microsoft",
            ],  # type: ignore[list-item]
            intent="retrieval",
            metric="revenue",
        )

        assert result["companies"] == [
            "Apple",
            "Microsoft",
        ]

    def test_non_list_companies_become_empty(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Calculate revenue growth.",
            companies="Apple",  # type: ignore[arg-type]
            intent="financial_metric",
            metric="revenue",
        )

        assert result["companies"] == []
        assert result["calculated_comparison"] is False
        assert result["reasoning_type"] == "calculation"

    def test_metric_is_preserved(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Calculate operating margin growth.",
            companies=["Apple"],
            intent="financial_metric",
            metric="operating_margin",
        )

        assert result["metric"] == "operating_margin"

    def test_none_metric_is_preserved(
        self,
        service: QueryDecompositionService,
    ) -> None:
        result = service.decompose(
            question="Show Apple data.",
            companies=["Apple"],
            intent="retrieval",
            metric=None,
        )

        assert result["metric"] is None


class TestNormalizeCompanies:
    def test_normalizes_company_whitespace(self) -> None:
        result = QueryDecompositionService._normalize_companies(
            [" Apple ", " Microsoft "]
        )

        assert result == [
            "Apple",
            "Microsoft",
        ]

    def test_removes_case_insensitive_duplicates(self) -> None:
        result = QueryDecompositionService._normalize_companies(
            [
                "Apple",
                "APPLE",
                "apple",
                "Microsoft",
            ]
        )

        assert result == [
            "Apple",
            "Microsoft",
        ]

    def test_preserves_first_company_formatting(self) -> None:
        result = QueryDecompositionService._normalize_companies(
            [
                "APPLE Inc.",
                "apple inc.",
            ]
        )

        assert result == [
            "APPLE Inc.",
        ]

    def test_removes_empty_company_names(self) -> None:
        result = QueryDecompositionService._normalize_companies(
            [
                "",
                " ",
                "\n",
                "Apple",
            ]
        )

        assert result == [
            "Apple",
        ]

    def test_removes_non_string_company_values(self) -> None:
        result = QueryDecompositionService._normalize_companies(
            [
                "Apple",
                None,
                10,
                True,
                {},
                [],
            ]
        )

        assert result == [
            "Apple",
        ]

    def test_non_list_input_returns_empty_list(self) -> None:
        assert (
            QueryDecompositionService._normalize_companies(
                "Apple"
            )
            == []
        )

    def test_none_input_returns_empty_list(self) -> None:
        assert (
            QueryDecompositionService._normalize_companies(
                None
            )
            == []
        )

    def test_empty_list_returns_empty_list(self) -> None:
        assert (
            QueryDecompositionService._normalize_companies(
                []
            )
            == []
        )


class TestPatternMatching:
    @pytest.mark.parametrize(
        "question",
        [
            "calculate revenue",
            "compute revenue",
            "show revenue growth",
            "calculate the growth rate",
            "show year-over-year revenue",
            "show year over year revenue",
            "show yoy revenue",
            "show percentage change",
            "show percent change",
        ],
    )
    def test_calculation_patterns_match(
        self,
        question: str,
    ) -> None:
        assert QueryDecompositionService._matches_any(
            question,
            CALCULATION_PATTERNS,
        )

    @pytest.mark.parametrize(
        "question",
        [
            "compare Apple and Microsoft",
            "show a comparison",
            "Apple versus Microsoft",
            "Apple vs Microsoft",
            "Apple vs. Microsoft",
            "which company grew faster",
            "which one is better",
            "which company performed better",
            "which company is stronger",
            "which company is weaker",
            "which has higher growth",
            "which has lower growth",
            "show the difference",
        ],
    )
    def test_comparison_patterns_match(
        self,
        question: str,
    ) -> None:
        assert QueryDecompositionService._matches_any(
            question,
            COMPARISON_PATTERNS,
        )

    @pytest.mark.parametrize(
        "question",
        [
            "explain the result",
            "identify the company",
            "determine the winner",
            "which performed better",
            "which company performed better",
            "identify the stronger performer",
            "identify the best performer",
        ],
    )
    def test_conclusion_patterns_match(
        self,
        question: str,
    ) -> None:
        assert QueryDecompositionService._matches_any(
            question,
            CONCLUSION_PATTERNS,
        )

    def test_patterns_do_not_match_unrelated_question(self) -> None:
        assert (
            QueryDecompositionService._matches_any(
                "What was Apple's revenue?",
                CALCULATION_PATTERNS,
            )
            is False
        )


class TestBuildSubtasks:
    def test_builds_retrieval_subtask_for_single_company(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple"],
            metric="revenue",
            calculation_requested=False,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert result == [
            {
                "step": 1,
                "type": "retrieval",
                "company": "Apple",
                "metric": "revenue",
                "description": (
                    "Retrieve relevant verified report evidence."
                ),
            }
        ]

    def test_retrieval_subtask_has_no_company_for_multiple_companies(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=False,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert result[0]["company"] is None

    def test_builds_one_calculation_subtask_per_company(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=True,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert len(result) == 2
        assert result[0]["step"] == 1
        assert result[0]["type"] == "calculation"
        assert result[0]["company"] == "Apple"

        assert result[1]["step"] == 2
        assert result[1]["type"] == "calculation"
        assert result[1]["company"] == "Microsoft"

    def test_formats_metric_name_with_spaces(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple"],
            metric="operating_margin",
            calculation_requested=True,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert result[0]["description"] == (
            "Calculate Apple's operating margin growth."
        )

    def test_uses_default_metric_description_for_none_metric(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple"],
            metric=None,
            calculation_requested=True,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert result[0]["description"] == (
            "Calculate Apple's requested financial metric growth."
        )

    def test_uses_default_metric_description_for_empty_metric(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple"],
            metric="   ",
            calculation_requested=True,
            comparison_requested=False,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert result[0]["description"] == (
            "Calculate Apple's requested financial metric growth."
        )

    def test_builds_absolute_comparison_subtask(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=False,
            comparison_requested=True,
            conclusion_requested=False,
            calculated_comparison=False,
        )

        assert len(result) == 1
        assert result[0]["step"] == 1
        assert result[0]["type"] == "absolute_comparison"
        assert result[0]["company"] is None
        assert result[0]["metric"] == "revenue"

    def test_builds_calculated_comparison_subtask(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=True,
            comparison_requested=True,
            conclusion_requested=False,
            calculated_comparison=True,
        )

        assert len(result) == 4

        assert result[0]["type"] == "calculation"
        assert result[1]["type"] == "calculation"
        assert result[2]["type"] == "calculated_comparison"
        assert result[3]["type"] == "conclusion"

        assert [
            subtask["step"]
            for subtask in result
        ] == [1, 2, 3, 4]

    def test_calculated_comparison_automatically_adds_conclusion(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=True,
            comparison_requested=True,
            conclusion_requested=False,
            calculated_comparison=True,
        )

        assert result[-1]["type"] == "conclusion"
        assert result[-1]["description"] == (
            "Identify the stronger performer using "
            "only verified deterministic results."
        )

    def test_explicit_conclusion_is_added_after_absolute_comparison(
        self,
    ) -> None:
        result = QueryDecompositionService._build_subtasks(
            companies=["Apple", "Microsoft"],
            metric="revenue",
            calculation_requested=False,
            comparison_requested=True,
            conclusion_requested=True,
            calculated_comparison=False,
        )

        assert len(result) == 2
        assert result[0]["type"] == "absolute_comparison"
        assert result[1]["type"] == "conclusion"
        assert result[1]["step"] == 2


class TestReasoningType:
    @pytest.mark.parametrize(
        (
            "calculation_requested",
            "comparison_requested",
            "calculated_comparison",
            "expected",
        ),
        [
            (
                True,
                True,
                True,
                "calculated_comparison",
            ),
            (
                True,
                False,
                False,
                "calculation",
            ),
            (
                False,
                True,
                False,
                "absolute_comparison",
            ),
            (
                False,
                False,
                False,
                "retrieval",
            ),
        ],
    )
    def test_determines_reasoning_type(
        self,
        calculation_requested: bool,
        comparison_requested: bool,
        calculated_comparison: bool,
        expected: str,
    ) -> None:
        result = (
            QueryDecompositionService
            ._determine_reasoning_type(
                calculation_requested=calculation_requested,
                comparison_requested=comparison_requested,
                calculated_comparison=calculated_comparison,
            )
        )

        assert result == expected


class TestRequiredCapabilities:
    def test_retrieval_capabilities(self) -> None:
        result = (
            QueryDecompositionService
            ._determine_required_capabilities(
                calculation_requested=False,
                comparison_requested=False,
                calculated_comparison=False,
            )
        )

        assert result == [
            "document_retrieval",
        ]

    def test_calculation_capabilities(self) -> None:
        result = (
            QueryDecompositionService
            ._determine_required_capabilities(
                calculation_requested=True,
                comparison_requested=False,
                calculated_comparison=False,
            )
        )

        assert result == [
            "document_retrieval",
            "financial_calculator",
        ]

    def test_absolute_comparison_capabilities(self) -> None:
        result = (
            QueryDecompositionService
            ._determine_required_capabilities(
                calculation_requested=False,
                comparison_requested=True,
                calculated_comparison=False,
            )
        )

        assert result == [
            "document_retrieval",
            "company_comparison",
        ]

    def test_calculated_comparison_capabilities(self) -> None:
        result = (
            QueryDecompositionService
            ._determine_required_capabilities(
                calculation_requested=True,
                comparison_requested=True,
                calculated_comparison=True,
            )
        )

        assert result == [
            "document_retrieval",
            "financial_calculator",
            "multi_step_reasoning",
        ]

    def test_calculated_comparison_does_not_add_company_comparison(
        self,
    ) -> None:
        result = (
            QueryDecompositionService
            ._determine_required_capabilities(
                calculation_requested=True,
                comparison_requested=True,
                calculated_comparison=True,
            )
        )

        assert "company_comparison" not in result