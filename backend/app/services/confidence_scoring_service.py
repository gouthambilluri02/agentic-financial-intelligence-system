from __future__ import annotations

from typing import Any


class ConfidenceScoringService:
    """
    Calculate an evidence-quality confidence score for a response.

    The score is based on:
    - retrieval relevance
    - evidence coverage
    - retrieval sufficiency
    - successful tool execution
    - deterministic answer support
    - retry count
    """

    def calculate(
        self,
        retrieved_chunks: list[dict[str, Any]],
        retrieval_sufficient: bool,
        retry_count: int,
        execution_result: dict[str, Any],
        deterministic_answer_used: bool,
    ) -> dict[str, Any]:
        """
        Return a structured confidence assessment.
        """

        safe_chunks = (
            retrieved_chunks
            if isinstance(retrieved_chunks, list)
            else []
        )

        safe_execution_result = (
            execution_result
            if isinstance(execution_result, dict)
            else {}
        )

        relevance_component = (
            self._calculate_relevance_component(
                safe_chunks
            )
        )

        coverage_component = (
            self._calculate_coverage_component(
                safe_chunks
            )
        )

        retrieval_component = (
            1.0
            if retrieval_sufficient
            else 0.45
        )

        tool_component = (
            self._calculate_tool_component(
                safe_execution_result
            )
        )

        deterministic_component = (
            1.0
            if deterministic_answer_used
            else 0.65
        )

        retry_component = (
            self._calculate_retry_component(
                retry_count
            )
        )

        components = {
            "retrieval_relevance": relevance_component,
            "evidence_coverage": coverage_component,
            "retrieval_sufficiency": retrieval_component,
            "tool_execution": tool_component,
            "deterministic_support": deterministic_component,
            "retry_stability": retry_component,
        }

        score = (
            relevance_component * 0.25
            + coverage_component * 0.15
            + retrieval_component * 0.20
            + tool_component * 0.20
            + deterministic_component * 0.15
            + retry_component * 0.05
        )

        if not safe_chunks:
            score = min(
                score,
                0.25,
            )

        score = round(
            min(
                max(score, 0.0),
                1.0,
            ),
            4,
        )

        return {
            "score": score,
            "percentage": round(
                score * 100,
                1,
            ),
            "level": self._score_to_level(
                score
            ),
            "reasons": self._build_reasons(
                retrieved_chunks=safe_chunks,
                retrieval_sufficient=(
                    retrieval_sufficient
                ),
                retry_count=retry_count,
                execution_result=(
                    safe_execution_result
                ),
                deterministic_answer_used=(
                    deterministic_answer_used
                ),
                relevance_component=(
                    relevance_component
                ),
            ),
            "components": {
                key: round(
                    value,
                    4,
                )
                for key, value in components.items()
            },
            "method": (
                "Evidence-quality heuristic based on retrieval, "
                "tool execution, and deterministic support."
            ),
        }

    @staticmethod
    def _calculate_relevance_component(
        retrieved_chunks: list[dict[str, Any]],
    ) -> float:
        """
        Calculate average semantic relevance.
        """

        relevance_scores: list[float] = []

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue

            distance = chunk.get(
                "distance"
            )

            if not isinstance(
                distance,
                (int, float),
            ):
                continue

            normalized_distance = max(
                float(distance),
                0.0,
            )

            relevance = (
                1.0
                / (
                    1.0
                    + normalized_distance
                )
            )

            relevance_scores.append(
                relevance
            )

        if not relevance_scores:
            return 0.0

        top_scores = sorted(
            relevance_scores,
            reverse=True,
        )[:5]

        return (
            sum(top_scores)
            / len(top_scores)
        )

    @staticmethod
    def _calculate_coverage_component(
        retrieved_chunks: list[dict[str, Any]],
    ) -> float:
        """
        Estimate evidence coverage from unique report pages.
        """

        unique_sources: set[
            tuple[Any, Any]
        ] = set()

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                continue

            unique_sources.add(
                (
                    metadata.get(
                        "source_file"
                    ),
                    metadata.get(
                        "page"
                    ),
                )
            )

        source_count = len(
            unique_sources
        )

        if source_count == 0:
            return 0.0

        if source_count == 1:
            return 0.55

        if source_count == 2:
            return 0.75

        if source_count == 3:
            return 0.9

        return 1.0

    @staticmethod
    def _calculate_tool_component(
        execution_result: dict[str, Any],
    ) -> float:
        """
        Calculate the ratio of successful tools.
        """

        executed_tools = execution_result.get(
            "executed_tools",
            [],
        )

        successful_tools = execution_result.get(
            "successful_tools",
            [],
        )

        if not isinstance(executed_tools, list):
            executed_tools = []

        if not isinstance(successful_tools, list):
            successful_tools = []

        if not executed_tools:
            return 0.0

        success_ratio = (
            len(successful_tools)
            / len(executed_tools)
        )

        return min(
            max(success_ratio, 0.0),
            1.0,
        )

    @staticmethod
    def _calculate_retry_component(
        retry_count: int,
    ) -> float:
        """
        Reduce confidence when retrieval required retries.
        """

        if not isinstance(retry_count, int):
            return 0.7

        if retry_count <= 0:
            return 1.0

        if retry_count == 1:
            return 0.75

        return 0.5

    @staticmethod
    def _score_to_level(
        score: float,
    ) -> str:
        """
        Convert the numeric score into a readable level.
        """

        if score >= 0.85:
            return "high"

        if score >= 0.65:
            return "medium"

        return "low"

    @staticmethod
    def _build_reasons(
        retrieved_chunks: list[dict[str, Any]],
        retrieval_sufficient: bool,
        retry_count: int,
        execution_result: dict[str, Any],
        deterministic_answer_used: bool,
        relevance_component: float,
    ) -> list[str]:
        """
        Explain the confidence score.
        """

        reasons: list[str] = []

        if deterministic_answer_used:
            reasons.append(
                "The answer used a verified deterministic tool result."
            )
        else:
            reasons.append(
                "The answer required language-model interpretation."
            )

        if retrieval_sufficient:
            reasons.append(
                "The retrieval evaluator considered the evidence sufficient."
            )
        else:
            reasons.append(
                "The retrieval evaluator identified limited evidence."
            )

        if retrieved_chunks:
            reasons.append(
                f"{len(retrieved_chunks)} report passages were retrieved."
            )
        else:
            reasons.append(
                "No relevant report passages were retrieved."
            )

        if relevance_component >= 0.8:
            reasons.append(
                "The strongest passages had high semantic relevance."
            )
        elif relevance_component >= 0.6:
            reasons.append(
                "The passages had moderate semantic relevance."
            )
        else:
            reasons.append(
                "The passages had weak or unavailable relevance signals."
            )

        successful_tools = execution_result.get(
            "successful_tools",
            [],
        )

        failed_tools = execution_result.get(
            "failed_tools",
            [],
        )

        if isinstance(successful_tools, list) and successful_tools:
            reasons.append(
                "Successful tools: "
                + ", ".join(
                    str(tool)
                    for tool in successful_tools
                )
                + "."
            )

        if isinstance(failed_tools, list) and failed_tools:
            reasons.append(
                "Failed tools reduced confidence: "
                + ", ".join(
                    str(tool)
                    for tool in failed_tools
                )
                + "."
            )

        if retry_count > 0:
            reasons.append(
                f"Retrieval required {retry_count} retry attempt."
            )

        return reasons


if __name__ == "__main__":
    service = ConfidenceScoringService()

    result = service.calculate(
        retrieved_chunks=[
            {
                "distance": 0.2,
                "metadata": {
                    "source_file": "apple_2024_10k.pdf",
                    "page": 37,
                },
            }
        ],
        retrieval_sufficient=True,
        retry_count=0,
        execution_result={
            "executed_tools": [
                "financial_calculator",
                "document_retrieval",
            ],
            "successful_tools": [
                "financial_calculator",
                "document_retrieval",
            ],
            "failed_tools": [],
        },
        deterministic_answer_used=True,
    )

    print(result)