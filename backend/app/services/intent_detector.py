import re


INTENT_PATTERNS = {
    "comparison": [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bwhich company\b",
        r"\bdifference between\b",
    ],
    "summary": [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bbriefly explain\b",
    ],
    "risk_analysis": [
        r"\brisk\b",
        r"\brisks\b",
        r"\brisk factors\b",
        r"\bcybersecurity\b",
        r"\blegal proceedings\b",
    ],
    "financial_metric": [
        r"\brevenue\b",
        r"\bnet income\b",
        r"\bearnings per share\b",
        r"\beps\b",
        r"\bcash flow\b",
        r"\boperating margin\b",
        r"\bgross margin\b",
        r"\bdebt\b",
        r"\bassets\b",
        r"\bliabilities\b",
    ],
}


class IntentDetector:
    """
    Detect the primary financial intent of a user question.
    """

    def detect_intent(self, question: str) -> str:
        """
        Return one supported intent for the question.
        """

        normalized_question = question.lower()

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized_question):
                    return intent

        return "general_question"


if __name__ == "__main__":
    detector = IntentDetector()

    test_questions = [
        "Compare Apple and Microsoft revenue.",
        "Summarize Apple's business.",
        "What risks did Microsoft disclose?",
        "What was Apple's revenue?",
        "What products does Microsoft offer?",
    ]

    for question in test_questions:
        intent = detector.detect_intent(question)

        print(f"\nQuestion: {question}")
        print(f"Detected intent: {intent}")