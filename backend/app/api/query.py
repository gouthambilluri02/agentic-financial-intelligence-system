from fastapi import APIRouter
from backend.app.schemas.query import QueryRequest
from backend.app.agents.financial_agent import FinancialAgent

router = APIRouter(
    prefix="/api/v1",
    tags=["Query"],
)


@router.post("/query")
def ask_question(request: QueryRequest):

    agent = FinancialAgent()

    answer = agent.process_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }