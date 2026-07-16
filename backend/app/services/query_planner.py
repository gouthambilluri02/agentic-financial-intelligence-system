from backend.app.services.query_rewriter import QueryRewriter


DEFAULT_TOP_K = 5


class QueryPlanner:
    """
    Build an execution plan for a financial question.

    The plan determines:
    - Which companies to search
    - Which intent was detected
    - Which financial metric was requested
    - Which query should be sent to ChromaDB
    - How many chunks should be retrieved
    """

    def __init__(self) -> None:
        self.query_rewriter = QueryRewriter()

    def build_plan(
        self,
        question: str,
        companies: list[str],
        intent: str,
        metric: str | None,
    ) -> dict:
        """
        Create a retrieval and execution plan.
        """

        retrieval_query = self.query_rewriter.rewrite(
            question=question,
            intent=intent,
            metric=metric,
        )

        top_k = self._determine_top_k(
            intent=intent,
            metric=metric,
        )

        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "companies": companies,
            "intent": intent,
            "metric": metric,
            "top_k": top_k,
        }

    @staticmethod
    def _determine_top_k(
        intent: str,
        metric: str | None,
    ) -> int:
        """
        Decide how many chunks should be retrieved.

        Different tasks require different amounts of context.
        """

        if intent == "comparison":
            return 10

        if intent == "summary":
            return 8

        if intent == "risk_analysis":
            return 8

        if metric is not None:
            return 6

        return DEFAULT_TOP_K


if __name__ == "__main__":
    planner = QueryPlanner()

    test_plans = [
        planner.build_plan(
            question="Compare Apple and Microsoft revenue.",
            companies=["Apple", "Microsoft"],
            intent="comparison",
            metric="revenue",
        ),
        planner.build_plan(
            question="What risks did Microsoft disclose?",
            companies=["Microsoft"],
            intent="risk_analysis",
            metric=None,
        ),
        planner.build_plan(
            question="What products does Apple offer?",
            companies=["Apple"],
            intent="general_question",
            metric=None,
        ),
    ]

    for plan in test_plans:
        print("\nExecution plan:")

        for key, value in plan.items():
            print(f"{key}: {value}")