from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


DOCUMENTS_DIR = Path("backend/app/data/documents/financial_reports")


def load_financial_reports():
    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = pdf_path.name

        documents.extend(pages)

    return documents


if __name__ == "__main__":
    loaded_documents = load_financial_reports()

    print(f"Total pages loaded: {len(loaded_documents)}")

    if loaded_documents:
        first_page = loaded_documents[0]

        print("\nFirst page metadata:")
        print(first_page.metadata)

        print("\nFirst 500 characters:")
        print(first_page.page_content[:500])

    else:
        print("No PDF documents were found.")