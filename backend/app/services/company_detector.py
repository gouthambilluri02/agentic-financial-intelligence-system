import re


COMPANY_ALIASES = {
    "Apple": [
        "apple",
        "aapl",
        "apple inc",
    ],
    "Microsoft": [
        "microsoft",
        "msft",
        "microsoft corporation",
    ],
    "Tesla": [
        "tesla",
        "tsla",
        "tesla inc",
    ],
    "Amazon": [
        "amazon",
        "amzn",
        "amazon.com",
    ],
    "NVIDIA": [
        "nvidia",
        "nvda",
    ],
    "Meta": [
        "meta",
        "meta platforms",
        "facebook",
    ],
    "Alphabet": [
        "alphabet",
        "google",
        "goog",
        "googl",
    ],
}


class CompanyDetector:
    """
    Detect company names and ticker symbols mentioned in a question.
    """

    def detect_companies(self, question: str) -> list[str]:
        """
        Return all supported companies mentioned in the question.
        """

        normalized_question = question.lower()
        detected_companies = []

        for company, aliases in COMPANY_ALIASES.items():
            for alias in aliases:
                pattern = rf"\b{re.escape(alias)}\b"

                if re.search(pattern, normalized_question):
                    detected_companies.append(company)
                    break

        return detected_companies


if __name__ == "__main__":
    company_detector = CompanyDetector()

    test_questions = [
        "What risks did Microsoft disclose?",
        "Compare Apple and Microsoft revenue.",
        "How is AAPL performing?",
        "What are Meta's major risks?",
        "Compare Google and NVIDIA.",
        "What is the company's revenue?",
    ]

    for question in test_questions:
        companies = company_detector.detect_companies(question)

        print(f"\nQuestion: {question}")
        print(f"Detected companies: {companies}")