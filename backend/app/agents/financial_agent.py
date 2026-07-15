import ollama

from backend.app.services.retrieval_service import RetrievalService


OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_TOP_K = 5


class FinancialAgent:
    """
    Coordinate retrieval and grounded LLM response generation.
    """

    def __init__(self) -> None:
        self.retrieval_service = RetrievalService()

    def process_question(self, question: str) -> dict:
        """
        Retrieve relevant report chunks and return an answer
        together with verified document sources.
        """

        retrieved_chunks = self.retrieval_service.retrieve(
            query=question,
            top_k=DEFAULT_TOP_K,
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "available financial reports."
                ),
                "sources": [],
            }

        context = self._build_context(retrieved_chunks)
        sources = self._extract_sources(retrieved_chunks)

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a financial research assistant.

Answer the user's question using only the supplied financial-report context.

Rules:
- Do not invent facts, numbers, or sources.
- If the supplied context is insufficient, clearly say so.
- Explain the answer professionally and clearly.
- Use Markdown headings and bullet points when helpful.
- Separate factual findings from interpretation.
- Do not promise investment outcomes.
- Do not create a separate source list because the application adds it.
""",
                },
                {
                    "role": "user",
                    "content": f"""
Financial-report context:

{context}

User question:

{question}
""",
                },
            ],
        )

        return {
            "answer": response["message"]["content"],
            "sources": sources,
        }

    @staticmethod
    def _build_context(retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into context for the language model.
        """

        context_sections = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]
            content = chunk["content"]

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
Source {index}
File: {source_file}
Page: {page_number}
Content:
{content}
""".strip()
            )

        return "\n\n".join(context_sections)

    @staticmethod
    def _extract_sources(
        retrieved_chunks: list[dict],
    ) -> list[dict]:
        """
        Extract unique source references from retrieval metadata.
        """

        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]

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

            source_key = (source_file, page_number)

            if source_key not in seen_sources:
                sources.append(
                    {
                        "source_file": source_file,
                        "page": page_number,
                    }
                )
                seen_sources.add(source_key)

        return sources


if __name__ == "__main__":
    financial_agent = FinancialAgent()

    test_question = "What risks did Apple disclose?"

    result = financial_agent.process_question(test_question)

    print(f"\nQuestion:\n{test_question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:\n{result['sources']}")