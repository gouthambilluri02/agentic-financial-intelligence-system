from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Financial question received from the frontend.
    """

    question: str = Field(
        ...,
        min_length=3,
        description="Financial question to ask about the reports.",
        examples=[
            "Calculate Apple's revenue growth.",
        ],
    )


class SourceReference(BaseModel):
    """
    Verified source associated with an answer.
    """

    company: str | None = None
    ticker: str | None = None
    source_file: str
    fiscal_year: int | str | None = None
    document_type: str | None = None
    page: int | str


class QueryResponse(BaseModel):
    """
    Complete structured response returned by the financial agent.
    """

    question: str
    answer: str

    sources: list[SourceReference] = Field(
        default_factory=list
    )

    calculation: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None

    detected_companies: list[str] = Field(
        default_factory=list
    )
    detected_intent: str = "general_question"
    detected_metric: str | None = None

    selected_tool: str = "document_retrieval"

    execution_plan: list[str] = Field(
        default_factory=list
    )
    executed_tools: list[str] = Field(
        default_factory=list
    )
    successful_tools: list[str] = Field(
        default_factory=list
    )
    failed_tools: list[str] = Field(
        default_factory=list
    )

    tool_outputs: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tool_execution_success: bool = False
    tool_execution_duration_ms: float = 0.0

    plan: dict[str, Any] = Field(
        default_factory=dict
    )

    retry_performed: bool = False
    retry_count: int = 0
    retrieval_sufficient: bool = False

    deterministic_answer_used: bool = False