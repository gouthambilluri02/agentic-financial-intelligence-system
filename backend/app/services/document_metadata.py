from pathlib import Path


COMPANY_METADATA = {
    "apple": {
        "company": "Apple",
        "ticker": "AAPL",
    },
    "microsoft": {
        "company": "Microsoft",
        "ticker": "MSFT",
    },
    "tesla": {
        "company": "Tesla",
        "ticker": "TSLA",
    },
    "amazon": {
        "company": "Amazon",
        "ticker": "AMZN",
    },
    "nvidia": {
        "company": "NVIDIA",
        "ticker": "NVDA",
    },
    "meta": {
        "company": "Meta",
        "ticker": "META",
    },
}


def extract_document_metadata(pdf_path: Path) -> dict:
    """
    Extract company, ticker, fiscal year, and document type
    from a standardized financial-report filename.

    Expected filename format:
        company_year_documenttype.pdf

    Example:
        apple_2024_10k.pdf
    """

    filename_without_extension = pdf_path.stem.lower()
    filename_parts = filename_without_extension.split("_")

    company_key = filename_parts[0]

    company_details = COMPANY_METADATA.get(
        company_key,
        {
            "company": company_key.title(),
            "ticker": "UNKNOWN",
        },
    )

    fiscal_year = (
        int(filename_parts[1])
        if len(filename_parts) > 1
        and filename_parts[1].isdigit()
        else "Unknown"
    )

    document_type = (
        filename_parts[2].upper()
        if len(filename_parts) > 2
        else "Unknown"
    )

    return {
        "company": company_details["company"],
        "ticker": company_details["ticker"],
        "fiscal_year": fiscal_year,
        "document_type": document_type,
        "source_file": pdf_path.name,
    }


if __name__ == "__main__":
    sample_file = Path(
        "backend/app/data/documents/financial_reports/"
        "apple_2024_10k.pdf"
    )

    metadata = extract_document_metadata(sample_file)

    print("Extracted metadata:")
    print(metadata)