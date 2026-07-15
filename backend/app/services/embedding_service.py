from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingService:
    """
    Generate vector embeddings for document chunks and user queries.
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple document chunks into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Convert one user query into an embedding vector.
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()


if __name__ == "__main__":
    embedding_service = EmbeddingService()

    sample_texts = [
        "Apple reported revenue growth from its services business.",
        "The company disclosed risks related to global supply chains.",
    ]

    document_embeddings = embedding_service.embed_documents(sample_texts)
    query_embedding = embedding_service.embed_query(
        "What risks did the company report?"
    )

    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Number of document embeddings: {len(document_embeddings)}")
    print(f"Document embedding dimension: {len(document_embeddings[0])}")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"First five query values: {query_embedding[:5]}")