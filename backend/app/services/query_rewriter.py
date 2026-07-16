METRIC_SEARCH_TERMS = {
    "revenue": (
        "total revenue net sales consolidated statements of operations "
        "fiscal year"
    ),
    "net_income": (
        "net income net earnings consolidated statements of operations "
        "fiscal year"
    ),
    "operating_income": (
        "operating income operating profit consolidated statements "
        "of operations fiscal year"
    ),
    "eps": (
        "earnings per share diluted EPS basic EPS consolidated statements "
        "of operations fiscal year"
    ),
    "cash_flow": (
        "net cash provided by operating activities operating cash flow "
        "consolidated statements of cash flows fiscal year"
    ),
    "assets": (
        "total assets consolidated balance sheets fiscal year end"
    ),
    "liabilities": (
        "total liabilities consolidated balance sheets fiscal year end"
    ),
}


INTENT_SEARCH_TERMS = {
    "risk_analysis": (
        "risk factors cybersecurity legal regulatory operational economic "
        "supply chain competition"
    ),
    "summary": (
        "business overview financial highlights products services "
        "operations risk factors"
    ),
    "comparison": (
        "financial results fiscal year consolidated statements"
    ),
    "general_question": "",
}


class QueryRewriter:
    """
    Rewrite user questions into retrieval-focused search queries.
    """

    def rewrite(
        self,
        question: str,
        intent: str,
        metric: str | None,
    ) -> str:
        """
        Build a search query optimized for document retrieval.
        """

        if metric:
            metric_terms = METRIC_SEARCH_TERMS.get(metric)

            if metric_terms:
                return metric_terms

        intent_terms = INTENT_SEARCH_TERMS.get(
            intent,
            "",
        )

        if intent_terms:
            return f"{question} {intent_terms}".strip()

        return question.strip()


if __name__ == "__main__":
    rewriter = QueryRewriter()

    tests = [
        {
            "question": "Compare Apple and Microsoft revenue.",
            "intent": "comparison",
            "metric": "revenue",
        },
        {
            "question": "What risks did Microsoft disclose?",
            "intent": "risk_analysis",
            "metric": None,
        },
        {
            "question": "Summarize Apple's business.",
            "intent": "summary",
            "metric": None,
        },
        {
            "question": "What products does Microsoft offer?",
            "intent": "general_question",
            "metric": None,
        },
    ]

    for test in tests:
        rewritten_query = rewriter.rewrite(
            question=test["question"],
            intent=test["intent"],
            metric=test["metric"],
        )

        print("\nOriginal:")
        print(test["question"])

        print("Rewritten:")
        print(rewritten_query)