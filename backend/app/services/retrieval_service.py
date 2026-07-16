from backend.app.services.vector_store import VectorStore


DEFAULT_TOP_K = 5


class RetrievalService:
    """
    Retrieve relevant financial-report chunks from ChromaDB.
    """

    def __init__(self) -> None:
        vector_store = VectorStore()
        self.collection = vector_store.get_collection()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company: str | None = None,
    ) -> list[dict]:
        """
        Search the vector database.

        When a company is provided, search only chunks belonging
        to that company.
        """

        query_arguments = {
            "query_texts": [query],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if company:
            query_arguments["where"] = {
                "company": company
            }

        results = self.collection.query(
            **query_arguments
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved_chunks


if __name__ == "__main__":
    retrieval_service = RetrievalService()

    company = "Microsoft"
    query = "What cybersecurity risks did the company disclose?"

    results = retrieval_service.retrieve(
        query=query,
        company=company,
    )

    print(f"Company filter: {company}")
    print(f"Query: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Result {index} ---")
        print(
            f"Company: "
            f"{result['metadata'].get('company')}"
        )
        print(
            f"Page: "
            f"{result['metadata'].get('page')}"
        )
        print(
            f"Distance: "
            f"{result['distance']}"
        )
        print(
            f"Content: "
            f"{result['content'][:400]}"
        )