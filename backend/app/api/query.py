from fastapi import APIRouter

from backend.app.agents.financial_agent import FinancialAgent
from backend.app.schemas.query import (
    QueryRequest,
    QueryResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Financial Query"],
)

financial_agent = FinancialAgent()


@router.post(
    "/query",
    response_model=QueryResponse,
)
def ask_financial_question(
    request: QueryRequest,
) -> QueryResponse:
    result = financial_agent.process_question(
        request.question
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
    )