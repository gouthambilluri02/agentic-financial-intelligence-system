from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from backend.app.services.document_metadata import (
    extract_document_metadata,
)


DOCUMENTS_DIR = Path(
    "backend/app/data/documents/financial_reports"
)


def load_financial_reports():
    """
    Load every financial-report PDF and attach metadata
    to each extracted page.
    """

    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        document_metadata = extract_document_metadata(pdf_path)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata.update(document_metadata)

        documents.extend(pages)

    return documents


if __name__ == "__main__":
    loaded_documents = load_financial_reports()

    print(f"Total pages loaded: {len(loaded_documents)}")

    companies = sorted(
        {
            document.metadata.get("company", "Unknown")
            for document in loaded_documents
        }
    )

    print("\nCompanies found:")
    print(companies)

    for company in companies:
        company_documents = [
            document
            for document in loaded_documents
            if document.metadata.get("company") == company
        ]

        print(f"{company}: {len(company_documents)} pages")

    if loaded_documents:
        print("\nSample metadata:")
        print(loaded_documents[0].metadata)
    else:
        print("\nNo PDF documents were found.")