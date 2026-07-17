class AnswerEvaluator:
    """
    Evaluate whether the retrieved document chunks are sufficient
    to answer the user's question.

    This is the first version of the evaluation layer.
    It uses simple, rule-based checks instead of another LLM call.
    """

    def should_retry(
        self,
        retrieved_chunks: list[dict],
        plan: dict,
    ) -> bool:
        """
        Return True when retrieval should run again.

        Return False when the current chunks appear sufficient.
        """

        if not retrieved_chunks:
            return True

        companies = plan.get("companies", [])
        intent = plan.get("intent", "general_question")
        metric = plan.get("metric")

        if companies:
            retrieved_companies = {
                chunk.get("metadata", {}).get("company")
                for chunk in retrieved_chunks
            }

            missing_companies = [
                company
                for company in companies
                if company not in retrieved_companies
            ]

            if missing_companies:
                return True

        minimum_chunks = self._get_minimum_chunks(
            intent=intent,
            metric=metric,
            company_count=len(companies),
        )

        if len(retrieved_chunks) < minimum_chunks:
            return True

        if metric and not self._contains_metric_terms(
            retrieved_chunks=retrieved_chunks,
            metric=metric,
        ):
            return True

        return False

    @staticmethod
    def _get_minimum_chunks(
        intent: str,
        metric: str | None,
        company_count: int,
    ) -> int:
        """
        Decide the minimum amount of retrieved context expected
        for different task types.
        """

        if intent == "comparison":
            return max(4, company_count * 2)

        if intent == "summary":
            return 4

        if intent == "risk_analysis":
            return 3

        if metric is not None:
            return 2

        return 1

    @staticmethod
    def _contains_metric_terms(
        retrieved_chunks: list[dict],
        metric: str,
    ) -> bool:
        """
        Check whether at least one retrieved chunk contains useful
        terms related to the requested financial metric.
        """

        metric_keywords = {
            "revenue": [
                "revenue",
                "net sales",
                "total sales",
            ],
            "net_income": [
                "net income",
                "net earnings",
                "profit",
            ],
            "operating_income": [
                "operating income",
                "operating profit",
            ],
            "eps": [
                "earnings per share",
                "diluted earnings per share",
                "basic earnings per share",
                "eps",
            ],
            "cash_flow": [
                "cash flow",
                "net cash provided by operating activities",
                "operating activities",
            ],
            "assets": [
                "total assets",
                "assets",
            ],
            "liabilities": [
                "total liabilities",
                "liabilities",
            ],
        }

        keywords = metric_keywords.get(metric, [])

        if not keywords:
            return True

        combined_content = " ".join(
            chunk.get("content", "").lower()
            for chunk in retrieved_chunks
        )

        return any(
            keyword.lower() in combined_content
            for keyword in keywords
        )


if __name__ == "__main__":
    evaluator = AnswerEvaluator()

    successful_chunks = [
        {
            "content": (
                "Microsoft reported total revenue of "
                "$245,122 million for fiscal year 2024."
            ),
            "metadata": {
                "company": "Microsoft",
                "page": 79,
            },
            "distance": 0.21,
        },
        {
            "content": (
                "Apple reported net sales in its consolidated "
                "statements of operations."
            ),
            "metadata": {
                "company": "Apple",
                "page": 30,
            },
            "distance": 0.24,
        },
        {
            "content": "Additional Apple revenue information.",
            "metadata": {
                "company": "Apple",
                "page": 31,
            },
            "distance": 0.28,
        },
        {
            "content": "Additional Microsoft revenue information.",
            "metadata": {
                "company": "Microsoft",
                "page": 80,
            },
            "distance": 0.29,
        },
    ]

    successful_plan = {
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

    missing_company_chunks = [
        {
            "content": (
                "Microsoft reported total revenue of "
                "$245,122 million."
            ),
            "metadata": {
                "company": "Microsoft",
                "page": 79,
            },
            "distance": 0.21,
        },
        {
            "content": "Additional Microsoft revenue information.",
            "metadata": {
                "company": "Microsoft",
                "page": 80,
            },
            "distance": 0.25,
        },
    ]

    no_metric_chunks = [
        {
            "content": (
                "Apple develops devices, services, and software."
            ),
            "metadata": {
                "company": "Apple",
                "page": 10,
            },
            "distance": 0.31,
        },
        {
            "content": (
                "Microsoft competes across cloud and software markets."
            ),
            "metadata": {
                "company": "Microsoft",
                "page": 20,
            },
            "distance": 0.33,
        },
        {
            "content": "General business information about Apple.",
            "metadata": {
                "company": "Apple",
                "page": 11,
            },
            "distance": 0.35,
        },
        {
            "content": "General business information about Microsoft.",
            "metadata": {
                "company": "Microsoft",
                "page": 21,
            },
            "distance": 0.36,
        },
    ]

    test_cases = [
        {
            "name": "Sufficient comparison results",
            "chunks": successful_chunks,
            "plan": successful_plan,
        },
        {
            "name": "One company is missing",
            "chunks": missing_company_chunks,
            "plan": successful_plan,
        },
        {
            "name": "Metric terms are missing",
            "chunks": no_metric_chunks,
            "plan": successful_plan,
        },
        {
            "name": "No chunks returned",
            "chunks": [],
            "plan": successful_plan,
        },
    ]

    for test_case in test_cases:
        retry_required = evaluator.should_retry(
            retrieved_chunks=test_case["chunks"],
            plan=test_case["plan"],
        )

        print("\n" + "=" * 70)
        print(f"Test: {test_case['name']}")
        print(f"Should retry: {retry_required}")