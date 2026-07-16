import ollama

from backend.app.services.company_detector import CompanyDetector
from backend.app.services.conversation_memory import ConversationMemory
from backend.app.services.financial_metrics import FinancialMetricDetector
from backend.app.services.intent_detector import IntentDetector
from backend.app.services.prompt_builder import PromptBuilder
from backend.app.services.query_planner import QueryPlanner
from backend.app.services.retrieval_service import RetrievalService


OLLAMA_MODEL = "qwen2.5:7b"


class FinancialAgent:
    """
    Coordinate the complete financial intelligence workflow.

    Responsibilities:
    - Detect companies
    - Detect user intent
    - Detect requested financial metrics
    - Resolve conversation context
    - Build an execution plan
    - Retrieve relevant report chunks
    - Build an intent-specific prompt
    - Generate a grounded answer with Ollama
    - Return verified source references
    - Update conversation memory
    """

    def __init__(self) -> None:
        self.company_detector = CompanyDetector()
        self.intent_detector = IntentDetector()
        self.metric_detector = FinancialMetricDetector()
        self.query_planner = QueryPlanner()
        self.prompt_builder = PromptBuilder()
        self.retrieval_service = RetrievalService()
        self.memory = ConversationMemory()

    def process_question(self, question: str) -> dict:
        """
        Process one financial question through the complete
        agentic RAG workflow.
        """

        # Detect companies explicitly mentioned in the current question.
        newly_detected_companies = (
            self.company_detector.detect_companies(question)
        )

        # Detect the current question's primary task.
        newly_detected_intent = (
            self.intent_detector.detect_intent(question)
        )

        # Detect whether the question asks for a financial metric.
        detected_metric = self.metric_detector.detect_metric(
            question
        )

        # Use current companies when available.
        # Otherwise, reuse companies from conversation memory.
        resolved_companies = self._resolve_companies(
            newly_detected_companies
        )

        # Use the current intent unless the question is vague.
        # For vague follow-ups, reuse the previous intent.
        resolved_intent = self._resolve_intent(
            newly_detected_intent
        )

        # Build a plan that controls retrieval behavior.
        plan = self.query_planner.build_plan(
            question=question,
            companies=resolved_companies,
            intent=resolved_intent,
            metric=detected_metric,
        )

        # Execute the retrieval strategy from the plan.
        retrieved_chunks = self._retrieve_context(
            question=plan["question"],
            companies=plan["companies"],
            top_k=plan["top_k"],
        )

        if not retrieved_chunks:
            self.memory.update(
                question=question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return {
                "answer": (
                    "I could not find relevant information in the "
                    "available financial reports."
                ),
                "sources": [],
                "detected_companies": resolved_companies,
                "detected_intent": resolved_intent,
                "detected_metric": detected_metric,
                "plan": plan,
            }

        context = self._build_context(
            retrieved_chunks
        )

        sources = self._extract_sources(
            retrieved_chunks
        )

        system_prompt = (
            self.prompt_builder.build_system_prompt(
                resolved_intent
            )
        )

        user_prompt = self.prompt_builder.build_user_prompt(
            question=question,
            context=context,
            detected_companies=resolved_companies,
            intent=resolved_intent,
        )

        # Add previous-turn context when available.
        previous_question = self.memory.get_last_question()

        if previous_question:
            user_prompt = (
                "Previous user question:\n"
                f"{previous_question}\n\n"
                "Current request and retrieved context:\n"
                f"{user_prompt}"
            )

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        answer = response["message"]["content"]

        # Save the current turn for future follow-up questions.
        self.memory.update(
            question=question,
            companies=resolved_companies,
            intent=resolved_intent,
        )

        return {
            "answer": answer,
            "sources": sources,
            "detected_companies": resolved_companies,
            "detected_intent": resolved_intent,
            "detected_metric": detected_metric,
            "plan": plan,
        }

    def _resolve_companies(
        self,
        newly_detected_companies: list[str],
    ) -> list[str]:
        """
        Use companies from the current question when available.

        If the user does not mention a company, reuse the companies
        from the previous conversation turn.
        """

        if newly_detected_companies:
            return newly_detected_companies

        return self.memory.get_last_companies()

    def _resolve_intent(
        self,
        newly_detected_intent: str,
    ) -> str:
        """
        Reuse the previous intent when the current question is vague.

        Example:
        Previous: "What risks did Apple disclose?"
        Current: "What about Microsoft?"

        Current detected intent:
        general_question

        Resolved intent:
        risk_analysis
        """

        previous_intent = self.memory.get_last_intent()

        if (
            newly_detected_intent == "general_question"
            and previous_intent
        ):
            return previous_intent

        return newly_detected_intent

    def _retrieve_context(
        self,
        question: str,
        companies: list[str],
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve relevant chunks based on the query plan.

        For one company:
        - Retrieve top_k chunks from that company's report.

        For multiple companies:
        - Retrieve top_k chunks separately for each company.

        When no company is available:
        - Search across the entire vector database.
        """

        if not companies:
            return self.retrieval_service.retrieve(
                query=question,
                top_k=top_k,
            )

        retrieved_chunks = []

        for company in companies:
            company_chunks = (
                self.retrieval_service.retrieve(
                    query=question,
                    top_k=top_k,
                    company=company,
                )
            )

            retrieved_chunks.extend(
                company_chunks
            )

        return retrieved_chunks

    @staticmethod
    def _build_context(
        retrieved_chunks: list[dict],
    ) -> str:
        """
        Convert retrieved chunks into structured context
        for the language model.
        """

        context_sections = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = chunk["metadata"]
            content = chunk["content"]

            company = metadata.get(
                "company",
                "Unknown company",
            )

            ticker = metadata.get(
                "ticker",
                "Unknown ticker",
            )

            fiscal_year = metadata.get(
                "fiscal_year",
                "Unknown year",
            )

            document_type = metadata.get(
                "document_type",
                "Unknown document type",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source",
            )

            page_number = metadata.get(
                "page",
                "Unknown page",
            )

            if isinstance(page_number, int):
                page_number += 1

            context_sections.append(
                f"""
Context {index}

Company: {company}
Ticker: {ticker}
Fiscal year: {fiscal_year}
Document type: {document_type}
File: {source_file}
Page: {page_number}

Content:
{content}
""".strip()
            )

        return "\n\n".join(
            context_sections
        )

    @staticmethod
    def _extract_sources(
        retrieved_chunks: list[dict],
    ) -> list[dict]:
        """
        Extract unique source references directly from
        ChromaDB metadata.
        """

        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]

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

            page_number = metadata.get(
                "page",
                "Unknown page",
            )

            if isinstance(page_number, int):
                page_number += 1

            source_key = (
                company,
                ticker,
                source_file,
                fiscal_year,
                document_type,
                page_number,
            )

            if source_key in seen_sources:
                continue

            sources.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "source_file": source_file,
                    "fiscal_year": fiscal_year,
                    "document_type": document_type,
                    "page": page_number,
                }
            )

            seen_sources.add(
                source_key
            )

        return sources

    def clear_memory(self) -> None:
        """
        Clear the current conversation memory.
        """

        self.memory.clear()


if __name__ == "__main__":
    financial_agent = FinancialAgent()

    test_questions = [
        "Compare Apple and Microsoft revenue.",
        "Which company appears stronger?",
    ]

    for test_question in test_questions:
        result = financial_agent.process_question(
            test_question
        )

        print("\n" + "=" * 80)

        print(
            f"Question:\n{test_question}"
        )

        print(
            "\nDetected companies:\n"
            f"{result['detected_companies']}"
        )

        print(
            "\nDetected intent:\n"
            f"{result['detected_intent']}"
        )

        print(
            "\nDetected metric:\n"
            f"{result['detected_metric']}"
        )

        print(
            "\nExecution plan:\n"
            f"{result['plan']}"
        )

        print(
            f"\nAnswer:\n{result['answer']}"
        )

        print("\nSources:")

        for source in result["sources"]:
            print(
            f"- {source['company']} | "
            f"{source['source_file']} | "
            f"Page {source['page']}"
    )