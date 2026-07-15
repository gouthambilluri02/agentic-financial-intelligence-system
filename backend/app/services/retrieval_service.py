from backend.app.services.vector_store import VectorStore


DEFAULT_TOP_K = 5


class RetrievalService:
    """
    Retrieve the most relevant financial-report chunks for a user query.
    """

    def __init__(self) -> None:
        vector_store = VectorStore()
        self.collection = vector_store.get_collection()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
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

    query = "What risks did Apple disclose?"

    results = retrieval_service.retrieve(query)

    print(f"Query: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Distance: {result['distance']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content: {result['content'][:500]}")