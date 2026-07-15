from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request model received from the frontend.
    """

    question: str = Field(
        min_length=3,
        description="Financial question to ask about the reports.",
    )


class SourceReference(BaseModel):
    """
    Represents a document source returned with the answer.
    """

    source_file: str
    page: int | str


class QueryResponse(BaseModel):
    """
    Response returned to the frontend.
    """

    question: str
    answer: str
    sources: list[SourceReference]