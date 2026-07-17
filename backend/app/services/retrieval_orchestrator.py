from backend.app.services.answer_evaluator import AnswerEvaluator
from backend.app.services.query_executor import QueryExecutor
from backend.app.services.query_planner import QueryPlanner


DEFAULT_MAX_RETRIES = 1
MAX_TOP_K = 20


class RetrievalOrchestrator:
    """
    Coordinate the complete retrieval workflow.

    Responsibilities:
    - Build the initial retrieval plan
    - Execute the plan
    - Evaluate the retrieved chunks
    - Retry with a broader query when necessary
    - Return the final plan and retrieved context
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.query_planner = QueryPlanner()
        self.query_executor = QueryExecutor()
        self.answer_evaluator = AnswerEvaluator()
        self.max_retries = max_retries

    def retrieve(
        self,
        question: str,
        companies: list[str],
        intent: str,
        metric: str | None = None,
    ) -> dict:
        """
        Run the complete retrieval workflow.

        Returns:
        - final execution plan
        - retrieved chunks
        - whether a retry was performed
        - number of retries
        - whether retrieval was sufficient
        """

        plan = self.query_planner.build_plan(
            question=question,
            companies=companies,
            intent=intent,
            metric=metric,
        )

        retrieved_chunks = self.query_executor.execute(
            plan
        )

        retry_count = 0

        retry_required = (
            self.answer_evaluator.should_retry(
                retrieved_chunks=retrieved_chunks,
                plan=plan,
            )
        )

        while (
            retry_required
            and retry_count < self.max_retries
        ):
            retry_count += 1

            plan = self._build_retry_plan(
                current_plan=plan,
                retry_count=retry_count,
            )

            retry_chunks = self.query_executor.execute(
                plan
            )

            if retry_chunks:
                retrieved_chunks = retry_chunks

            retry_required = (
                self.answer_evaluator.should_retry(
                    retrieved_chunks=retrieved_chunks,
                    plan=plan,
                )
            )

        retrieval_sufficient = not retry_required

        return {
            "plan": plan,
            "chunks": retrieved_chunks,
            "retry_performed": retry_count > 0,
            "retry_count": retry_count,
            "retrieval_sufficient": retrieval_sufficient,
        }

    @staticmethod
    def _build_retry_plan(
        current_plan: dict,
        retry_count: int,
    ) -> dict:
        """
        Create a broader plan for another retrieval attempt.

        The retry:
        - keeps the same company filters
        - increases top_k
        - combines the rewritten query with the original question
        """

        retry_plan = current_plan.copy()

        current_top_k = current_plan.get(
            "top_k",
            5,
        )

        retry_plan["top_k"] = min(
            current_top_k * 2,
            MAX_TOP_K,
        )

        original_question = current_plan.get(
            "question",
            "",
        )

        current_retrieval_query = current_plan.get(
            "retrieval_query",
            original_question,
        )

        retry_plan["retrieval_query"] = (
            f"{current_retrieval_query} "
            f"{original_question}"
        ).strip()

        retry_plan["is_retry"] = True
        retry_plan["retry_count"] = retry_count

        return retry_plan


if __name__ == "__main__":
    orchestrator = RetrievalOrchestrator(
        max_retries=1
    )

    test_cases = [
        {
            "question": (
                "Compare Apple and Microsoft revenue."
            ),
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "intent": "comparison",
            "metric": "revenue",
        },
        {
            "question": (
                "What risks did Microsoft disclose?"
            ),
            "companies": [
                "Microsoft",
            ],
            "intent": "risk_analysis",
            "metric": None,
        },
    ]

    for test_case in test_cases:
        result = orchestrator.retrieve(
            question=test_case["question"],
            companies=test_case["companies"],
            intent=test_case["intent"],
            metric=test_case["metric"],
        )

        print("\n" + "=" * 80)
        print(f"Question: {test_case['question']}")

        print("\nFinal execution plan:")

        for key, value in result["plan"].items():
            print(f"{key}: {value}")

        print(
            "\nRetrieved chunks: "
            f"{len(result['chunks'])}"
        )

        print(
            "Retry performed: "
            f"{result['retry_performed']}"
        )

        print(
            "Retry count: "
            f"{result['retry_count']}"
        )

        print(
            "Retrieval sufficient: "
            f"{result['retrieval_sufficient']}"
        )

        print("\nRetrieved results:")

        for index, chunk in enumerate(
            result["chunks"],
            start=1,
        ):
            metadata = chunk.get(
                "metadata",
                {},
            )

            print(
                f"- Result {index}: "
                f"{metadata.get('company', 'Unknown')} | "
                f"{metadata.get('source_file', 'Unknown')} | "
                f"Page {metadata.get('page', 'Unknown')} | "
                f"Distance {chunk.get('distance', 'Unknown')}"
            )