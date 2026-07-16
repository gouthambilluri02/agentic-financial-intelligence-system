from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request received from the frontend.
    """

    question: str = Field(
        min_length=3,
        description=(
            "Financial question to ask about the indexed reports."
        ),
    )


class SourceReference(BaseModel):
    """
    Verified document source returned with the answer.
    """

    company: str
    source_file: str
    fiscal_year: int | str
    document_type: str
    page: int | str


class QueryResponse(BaseModel):
    """
    Structured API response.
    """

    question: str
    answer: str
    detected_companies: list[str]
    detected_intent: str
    sources: list[SourceReference]