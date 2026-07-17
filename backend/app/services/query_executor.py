from backend.app.services.retrieval_service import RetrievalService


class QueryExecutor:
    """
    Execute retrieval plans created by QueryPlanner.
    """

    def __init__(self) -> None:
        self.retrieval_service = RetrievalService()

    def execute(self, plan: dict) -> list[dict]:
        """
        Execute a retrieval plan and return relevant document chunks.
        """

        retrieval_query = plan["retrieval_query"]
        companies = plan["companies"]
        top_k = plan["top_k"]

        if not companies:
            results = self.retrieval_service.retrieve(
                query=retrieval_query,
                top_k=top_k,
            )

            return self._remove_duplicates(results)

        all_results = []

        for company in companies:
            company_results = self.retrieval_service.retrieve(
                query=retrieval_query,
                top_k=top_k,
                company=company,
            )

            all_results.extend(company_results)

        unique_results = self._remove_duplicates(all_results)

        return self._sort_by_distance(unique_results)

    @staticmethod
    def _remove_duplicates(
        results: list[dict],
    ) -> list[dict]:
        """
        Remove duplicate chunks using source file, page, and content.
        """

        unique_results = []
        seen_results = set()

        for result in results:
            metadata = result["metadata"]
            content = result["content"]

            result_key = (
                metadata.get("source_file"),
                metadata.get("page"),
                content,
            )

            if result_key in seen_results:
                continue

            unique_results.append(result)
            seen_results.add(result_key)

        return unique_results

    @staticmethod
    def _sort_by_distance(
        results: list[dict],
    ) -> list[dict]:
        """
        Sort results from most relevant to least relevant.

        Smaller ChromaDB distance values generally indicate
        stronger semantic similarity.
        """

        return sorted(
            results,
            key=lambda result: result.get(
                "distance",
                float("inf"),
            ),
        )


if __name__ == "__main__":
    executor = QueryExecutor()

    test_plan = {
        "question": "Compare Apple and Microsoft revenue.",
        "retrieval_query": (
            "total revenue net sales consolidated statements "
            "of operations fiscal year"
        ),
        "companies": [
            "Apple",
            "Microsoft",
        ],
        "intent": "comparison",
        "metric": "revenue",
        "top_k": 5,
    }

    results = executor.execute(test_plan)

    print(f"Retrieved chunks: {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        print(f"\n--- Result {index} ---")
        print(
            f"Company: "
            f"{metadata.get('company', 'Unknown')}"
        )
        print(
            f"File: "
            f"{metadata.get('source_file', 'Unknown')}"
        )
        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )
        print(
            f"Distance: "
            f"{result.get('distance')}"
        )
        print(
            f"Content: "
            f"{result['content'][:300]}"
        )