import re


class FinancialMetricDetector:

    METRICS = {
        "revenue": [
            "revenue",
            "net sales",
            "sales",
        ],
        "net_income": [
            "net income",
            "profit",
            "earnings",
        ],
        "operating_income": [
            "operating income",
            "operating profit",
        ],
        "eps": [
            "eps",
            "earnings per share",
        ],
        "cash_flow": [
            "cash flow",
            "operating cash flow",
        ],
        "assets": [
            "assets",
            "total assets",
        ],
        "liabilities": [
            "liabilities",
            "total liabilities",
        ],
    }

    def detect_metric(self, question: str):

        question = question.lower()

        for metric, keywords in self.METRICS.items():

            for keyword in keywords:

                if re.search(rf"\b{re.escape(keyword)}\b", question):

                    return metric

        return None


if __name__ == "__main__":

    detector = FinancialMetricDetector()

    tests = [
        "What was Apple's revenue?",
        "Show Microsoft's net income.",
        "Compare operating income.",
        "What is EPS?",
        "Cash flow?",
        "How many employees does Apple have?",
    ]

    for question in tests:
        print(question)
        print(detector.detect_metric(question))
        print()