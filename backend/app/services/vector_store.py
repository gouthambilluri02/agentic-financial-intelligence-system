from chromadb import PersistentClient
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from backend.app.services.document_loader import load_financial_reports
from backend.app.services.text_splitter import split_documents


CHROMA_PATH = "backend/app/data/chroma_db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)


class VectorStore:
    def __init__(self) -> None:
        self.client = PersistentClient(path=CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )

    def index_documents(self) -> None:
        documents = load_financial_reports()
        chunks = split_documents(documents)

        # Get the IDs of records that may already exist.
        existing_records = self.collection.get()
        existing_ids = existing_records.get("ids", [])

        # Delete old records before rebuilding the index.
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        self.collection.add(
            ids=[f"chunk-{index}" for index in range(len(chunks))],
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

        print(f"Indexed {len(chunks)} chunks successfully.")
        print(f"Total records in collection: {self.collection.count()}")

    def get_collection(self):
        return self.collection


if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.index_documents()