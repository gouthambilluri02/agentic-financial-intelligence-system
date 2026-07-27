"""
Tests for ConfidenceScoringService.

Covered behavior:
- calculate()
- _calculate_relevance_component()
- _calculate_coverage_component()
- _calculate_tool_component()
- _calculate_retry_component()
- _score_to_level()
- _build_reasons()
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.confidence_scoring_service import (
    ConfidenceScoringService,
)


@pytest.fixture
def service() -> ConfidenceScoringService:
    return ConfidenceScoringService()


def make_chunk(
    *,
    distance: Any = 0.2,
    source_file: Any = "report.pdf",
    page: Any = 1,
) -> dict[str, Any]:
    return {
        "distance": distance,
        "metadata": {
            "source_file": source_file,
            "page": page,
        },
    }


class TestCalculate:
    def test_returns_expected_response_keys(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={
                "executed_tools": ["document_retrieval"],
                "successful_tools": ["document_retrieval"],
                "failed_tools": [],
            },
            deterministic_answer_used=True,
        )

        assert set(result) == {
            "score",
            "percentage",
            "level",
            "reasons",
            "components",
            "method",
        }

    def test_returns_expected_component_keys(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={
                "executed_tools": ["document_retrieval"],
                "successful_tools": ["document_retrieval"],
            },
            deterministic_answer_used=True,
        )

        assert set(result["components"]) == {
            "retrieval_relevance",
            "evidence_coverage",
            "retrieval_sufficiency",
            "tool_execution",
            "deterministic_support",
            "retry_stability",
        }

    def test_perfect_inputs_produce_high_confidence(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        chunks = [
            make_chunk(
                distance=0.0,
                source_file=f"report_{index}.pdf",
                page=index,
            )
            for index in range(4)
        ]

        result = service.calculate(
            retrieved_chunks=chunks,
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={
                "executed_tools": [
                    "document_retrieval",
                    "financial_calculator",
                ],
                "successful_tools": [
                    "document_retrieval",
                    "financial_calculator",
                ],
                "failed_tools": [],
            },
            deterministic_answer_used=True,
        )

        assert result["score"] == 1.0
        assert result["percentage"] == 100.0
        assert result["level"] == "high"

    def test_no_chunks_caps_score_at_point_twenty_five(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={
                "executed_tools": ["tool"],
                "successful_tools": ["tool"],
            },
            deterministic_answer_used=True,
        )

        assert result["score"] == 0.25
        assert result["percentage"] == 25.0
        assert result["level"] == "low"

    @pytest.mark.parametrize(
        "retrieved_chunks",
        [None, "invalid", 123, {}, ()],
    )
    def test_invalid_chunks_are_normalized_to_empty_list(
        self,
        service: ConfidenceScoringService,
        retrieved_chunks: Any,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=retrieved_chunks,  # type: ignore[arg-type]
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
        )

        assert result["score"] <= 0.25
        assert result["components"]["retrieval_relevance"] == 0.0
        assert result["components"]["evidence_coverage"] == 0.0
        assert (
            "No relevant report passages were retrieved."
            in result["reasons"]
        )

    @pytest.mark.parametrize(
        "execution_result",
        [None, "invalid", 123, [], ()],
    )
    def test_invalid_execution_result_is_normalized_to_empty_dict(
        self,
        service: ConfidenceScoringService,
        execution_result: Any,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result=execution_result,  # type: ignore[arg-type]
            deterministic_answer_used=True,
        )

        assert result["components"]["tool_execution"] == 0.0

    def test_retrieval_components(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        sufficient = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
        )
        insufficient = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
        )

        assert sufficient["components"]["retrieval_sufficiency"] == 1.0
        assert insufficient["components"]["retrieval_sufficiency"] == 0.45

    def test_deterministic_components(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        deterministic = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=True,
        )
        llm = service.calculate(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
        )

        assert deterministic["components"]["deterministic_support"] == 1.0
        assert llm["components"]["deterministic_support"] == 0.65

    def test_components_are_rounded_to_four_decimal_places(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[make_chunk(distance=0.2)],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={
                "executed_tools": ["a", "b", "c"],
                "successful_tools": ["a"],
            },
            deterministic_answer_used=False,
        )

        assert result["components"]["retrieval_relevance"] == 0.8333
        assert result["components"]["tool_execution"] == 0.3333

    def test_percentage_is_score_times_one_hundred(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[make_chunk(distance=0.5)],
            retrieval_sufficient=False,
            retry_count=1,
            execution_result={
                "executed_tools": ["a"],
                "successful_tools": [],
            },
            deterministic_answer_used=False,
        )

        assert result["percentage"] == round(
            result["score"] * 100,
            1,
        )

    def test_method_description_is_stable(
        self,
        service: ConfidenceScoringService,
    ) -> None:
        result = service.calculate(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
        )

        assert result["method"] == (
            "Evidence-quality heuristic based on retrieval, "
            "tool execution, and deterministic support."
        )


class TestCalculateRelevanceComponent:
    def test_empty_chunks_return_zero(self) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_relevance_component([])
            == 0.0
        )

    @pytest.mark.parametrize(
        "chunk",
        [None, "invalid", 123, [], ()],
    )
    def test_invalid_chunk_items_are_ignored(
        self,
        chunk: Any,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_relevance_component([chunk])
            == 0.0
        )

    @pytest.mark.parametrize(
        "distance",
        [None, "0.2", [], {}, ()],
    )
    def test_invalid_distance_is_ignored(
        self,
        distance: Any,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_relevance_component(
                [{"distance": distance}]
            )
            == 0.0
        )

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (0, 1.0),
            (0.2, 1.0 / 1.2),
            (1, 0.5),
            (3, 0.25),
        ],
    )
    def test_distance_is_converted_to_relevance(
        self,
        distance: float,
        expected: float,
    ) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_relevance_component(
                [{"distance": distance}]
            )
        )

        assert result == pytest.approx(expected)

    def test_negative_distance_is_clamped_to_zero(self) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_relevance_component(
                [{"distance": -5}]
            )
        )

        assert result == 1.0

    def test_uses_average_of_top_five_scores(self) -> None:
        chunks = [
            {"distance": 0},
            {"distance": 0.1},
            {"distance": 0.2},
            {"distance": 0.3},
            {"distance": 0.4},
            {"distance": 100},
        ]

        result = (
            ConfidenceScoringService
            ._calculate_relevance_component(chunks)
        )

        expected = sum(
            [
                1.0,
                1 / 1.1,
                1 / 1.2,
                1 / 1.3,
                1 / 1.4,
            ]
        ) / 5

        assert result == pytest.approx(expected)

    def test_boolean_distance_is_treated_as_numeric(self) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_relevance_component(
                [{"distance": True}]
            )
        )

        assert result == 0.5


class TestCalculateCoverageComponent:
    def test_empty_chunks_return_zero(self) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_coverage_component([])
            == 0.0
        )

    @pytest.mark.parametrize(
        "chunk",
        [None, "invalid", 123, [], ()],
    )
    def test_invalid_chunk_items_are_ignored(
        self,
        chunk: Any,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_coverage_component([chunk])
            == 0.0
        )

    @pytest.mark.parametrize(
        "metadata",
        [None, "invalid", 123, [], ()],
    )
    def test_invalid_metadata_is_ignored(
        self,
        metadata: Any,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_coverage_component(
                [{"metadata": metadata}]
            )
            == 0.0
        )

    @pytest.mark.parametrize(
        ("source_count", "expected"),
        [
            (1, 0.55),
            (2, 0.75),
            (3, 0.9),
            (4, 1.0),
            (6, 1.0),
        ],
    )
    def test_unique_source_thresholds(
        self,
        source_count: int,
        expected: float,
    ) -> None:
        chunks = [
            make_chunk(
                source_file=f"{index}.pdf",
                page=index,
            )
            for index in range(source_count)
        ]

        result = (
            ConfidenceScoringService
            ._calculate_coverage_component(chunks)
        )

        assert result == expected

    def test_duplicate_source_and_page_are_counted_once(self) -> None:
        chunks = [
            make_chunk(source_file="a.pdf", page=1),
            make_chunk(source_file="a.pdf", page=1),
        ]

        assert (
            ConfidenceScoringService
            ._calculate_coverage_component(chunks)
            == 0.55
        )

    def test_missing_metadata_fields_still_form_one_source(self) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_coverage_component(
                [{"metadata": {}}]
            )
            == 0.55
        )


class TestCalculateToolComponent:
    def test_no_executed_tools_returns_zero(self) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_tool_component({})
            == 0.0
        )

    @pytest.mark.parametrize(
        "executed_tools",
        [None, "tool", 123, {}, ()],
    )
    def test_invalid_executed_tools_are_normalized_to_empty(
        self,
        executed_tools: Any,
    ) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_tool_component(
                {
                    "executed_tools": executed_tools,
                    "successful_tools": ["tool"],
                }
            )
        )

        assert result == 0.0

    @pytest.mark.parametrize(
        "successful_tools",
        [None, "tool", 123, {}, ()],
    )
    def test_invalid_successful_tools_are_normalized_to_empty(
        self,
        successful_tools: Any,
    ) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_tool_component(
                {
                    "executed_tools": ["tool"],
                    "successful_tools": successful_tools,
                }
            )
        )

        assert result == 0.0

    @pytest.mark.parametrize(
        ("executed", "successful", "expected"),
        [
            (["a"], [], 0.0),
            (["a"], ["a"], 1.0),
            (["a", "b"], ["a"], 0.5),
            (["a", "b", "c"], ["a", "b"], 2 / 3),
        ],
    )
    def test_success_ratio_is_calculated(
        self,
        executed: list[str],
        successful: list[str],
        expected: float,
    ) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_tool_component(
                {
                    "executed_tools": executed,
                    "successful_tools": successful,
                }
            )
        )

        assert result == pytest.approx(expected)

    def test_success_ratio_is_capped_at_one(self) -> None:
        result = (
            ConfidenceScoringService
            ._calculate_tool_component(
                {
                    "executed_tools": ["a"],
                    "successful_tools": ["a", "b", "c"],
                }
            )
        )

        assert result == 1.0


class TestCalculateRetryComponent:
    @pytest.mark.parametrize(
        "retry_count",
        [None, "1", 1.5, [], {}, ()],
    )
    def test_invalid_retry_count_returns_point_seven(
        self,
        retry_count: Any,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_retry_component(retry_count)
            == 0.7
        )

    @pytest.mark.parametrize(
        ("retry_count", "expected"),
        [
            (-10, 1.0),
            (-1, 1.0),
            (0, 1.0),
            (1, 0.75),
            (2, 0.5),
            (3, 0.5),
            (10, 0.5),
            (True, 0.75),
            (False, 1.0),
        ],
    )
    def test_retry_thresholds(
        self,
        retry_count: int,
        expected: float,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._calculate_retry_component(retry_count)
            == expected
        )


class TestScoreToLevel:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (-1, "low"),
            (0, "low"),
            (0.25, "low"),
            (0.6499, "low"),
            (0.65, "medium"),
            (0.7, "medium"),
            (0.8499, "medium"),
            (0.85, "high"),
            (0.9, "high"),
            (1.0, "high"),
            (5.0, "high"),
        ],
    )
    def test_score_thresholds(
        self,
        score: float,
        expected: str,
    ) -> None:
        assert (
            ConfidenceScoringService
            ._score_to_level(score)
            == expected
        )


class TestBuildReasons:
    def test_deterministic_and_llm_reasons(self) -> None:
        deterministic = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=True,
            relevance_component=0.0,
        )
        llm = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.0,
        )

        assert deterministic[0] == (
            "The answer used a verified deterministic tool result."
        )
        assert llm[0] == (
            "The answer required language-model interpretation."
        )

    def test_retrieval_reasons(self) -> None:
        sufficient = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.0,
        )
        limited = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.0,
        )

        assert (
            "The retrieval evaluator considered the evidence sufficient."
            in sufficient
        )
        assert (
            "The retrieval evaluator identified limited evidence."
            in limited
        )

    def test_retrieved_passage_count_is_added(self) -> None:
        reasons = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[
                make_chunk(),
                make_chunk(page=2),
            ],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.5,
        )

        assert "2 report passages were retrieved." in reasons

    @pytest.mark.parametrize(
        ("relevance", "expected"),
        [
            (
                0.8,
                "The strongest passages had high semantic relevance.",
            ),
            (
                1.0,
                "The strongest passages had high semantic relevance.",
            ),
            (
                0.6,
                "The passages had moderate semantic relevance.",
            ),
            (
                0.79,
                "The passages had moderate semantic relevance.",
            ),
            (
                0.59,
                (
                    "The passages had weak or unavailable "
                    "relevance signals."
                ),
            ),
        ],
    )
    def test_relevance_reason_matches_threshold(
        self,
        relevance: float,
        expected: str,
    ) -> None:
        reasons = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[make_chunk()],
            retrieval_sufficient=True,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=relevance,
        )

        assert expected in reasons

    def test_successful_and_failed_tools_are_listed(self) -> None:
        reasons = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={
                "successful_tools": [
                    "document_retrieval",
                    "financial_calculator",
                ],
                "failed_tools": [
                    "risk_analysis",
                    "company_comparison",
                ],
            },
            deterministic_answer_used=False,
            relevance_component=0.0,
        )

        assert (
            "Successful tools: document_retrieval, "
            "financial_calculator."
            in reasons
        )
        assert (
            "Failed tools reduced confidence: risk_analysis, "
            "company_comparison."
            in reasons
        )

    def test_non_list_tool_fields_are_ignored(self) -> None:
        reasons = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={
                "successful_tools": "tool",
                "failed_tools": "tool",
            },
            deterministic_answer_used=False,
            relevance_component=0.0,
        )

        assert not any(
            reason.startswith("Successful tools:")
            for reason in reasons
        )
        assert not any(
            reason.startswith(
                "Failed tools reduced confidence:"
            )
            for reason in reasons
        )

    def test_retry_reason_is_added_only_for_positive_count(
        self,
    ) -> None:
        positive = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=2,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.0,
        )
        zero = ConfidenceScoringService._build_reasons(
            retrieved_chunks=[],
            retrieval_sufficient=False,
            retry_count=0,
            execution_result={},
            deterministic_answer_used=False,
            relevance_component=0.0,
        )

        assert "Retrieval required 2 retry attempt." in positive
        assert not any(
            reason.startswith("Retrieval required")
            for reason in zero
        )
