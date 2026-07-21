from __future__ import annotations

from typing import Any


class SourceRankingService:
    """
    Rank retrieved financial-report sources by semantic relevance.

    Smaller retrieval-distance values generally indicate stronger
    semantic similarity.
    """

    def rank_sources(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert retrieved chunks into unique ranked sources.
        """

        if not isinstance(
            retrieved_chunks,
            list,
        ):
            return []

        ranked_sources: list[
            dict[str, Any]
        ] = []

        seen_sources: set[
            tuple[Any, ...]
        ] = set()

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            company = metadata.get(
                "company",
                "Unknown company",
            )

            ticker = metadata.get(
                "ticker",
                "Unknown ticker",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source",
            )

            fiscal_year = metadata.get(
                "fiscal_year",
                "Unknown year",
            )

            document_type = metadata.get(
                "document_type",
                "Unknown document type",
            )

            stored_page = metadata.get(
                "page",
                "Unknown page",
            )

            display_page = stored_page

            if isinstance(stored_page, int):
                display_page = stored_page + 1

            distance = chunk.get(
                "distance"
            )

            relevance_score = (
                self._distance_to_relevance(
                    distance
                )
            )

            source_key = (
                company,
                ticker,
                source_file,
                fiscal_year,
                document_type,
                display_page,
            )

            if source_key in seen_sources:
                continue

            ranked_sources.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "source_file": source_file,
                    "fiscal_year": fiscal_year,
                    "document_type": document_type,
                    "page": display_page,
                    "relevance_score": (
                        relevance_score
                    ),
                    "retrieval_distance": (
                        round(
                            float(distance),
                            4,
                        )
                        if isinstance(
                            distance,
                            (int, float),
                        )
                        else None
                    ),
                }
            )

            seen_sources.add(
                source_key
            )

        ranked_sources.sort(
            key=lambda source: source.get(
                "relevance_score",
                0.0,
            ),
            reverse=True,
        )

        for rank, source in enumerate(
            ranked_sources,
            start=1,
        ):
            source["rank"] = rank

        return ranked_sources

    @staticmethod
    def _distance_to_relevance(
        distance: Any,
    ) -> float:
        """
        Convert retrieval distance into a score between 0 and 1.

        Formula:
        relevance = 1 / (1 + distance)
        """

        if not isinstance(
            distance,
            (int, float),
        ):
            return 0.0

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

        return round(
            min(
                max(
                    relevance,
                    0.0,
                ),
                1.0,
            ),
            4,
        )


if __name__ == "__main__":
    service = SourceRankingService()

    test_sources = service.rank_sources(
        [
            {
                "content": (
                    "Apple revenue information."
                ),
                "metadata": {
                    "company": "Apple",
                    "ticker": "AAPL",
                    "source_file": (
                        "apple_2024_10k.pdf"
                    ),
                    "fiscal_year": 2024,
                    "document_type": "10K",
                    "page": 37,
                },
                "distance": 0.2,
            },
            {
                "content": (
                    "Microsoft revenue information."
                ),
                "metadata": {
                    "company": "Microsoft",
                    "ticker": "MSFT",
                    "source_file": (
                        "microsoft_2024_10k.pdf"
                    ),
                    "fiscal_year": 2024,
                    "document_type": "10K",
                    "page": 79,
                },
                "distance": 0.28,
            },
        ]
    )

    for source in test_sources:
        print(source)