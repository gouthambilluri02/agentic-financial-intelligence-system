from fastapi import APIRouter, HTTPException

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
    """
    Process a financial question through the complete
    agentic financial-analysis workflow.
    """

    try:
        result = financial_agent.process_question(
            request.question
        )

        return QueryResponse(
            question=request.question,
            answer=result.get(
                "answer",
                "No answer was generated.",
            ),
            sources=result.get(
                "sources",
                [],
            ),
            calculation=result.get(
                "calculation",
            ),
            comparison=result.get(
                "comparison",
            ),
            detected_companies=result.get(
                "detected_companies",
                [],
            ),
            detected_intent=result.get(
                "detected_intent",
                "general_question",
            ),
            detected_metric=result.get(
                "detected_metric",
            ),
            selected_tool=result.get(
                "selected_tool",
                "document_retrieval",
            ),
            execution_plan=result.get(
                "execution_plan",
                [],
            ),
            executed_tools=result.get(
                "executed_tools",
                [],
            ),
            successful_tools=result.get(
                "successful_tools",
                [],
            ),
            failed_tools=result.get(
                "failed_tools",
                [],
            ),
            tool_outputs=result.get(
                "tool_outputs",
                [],
            ),
            tool_execution_success=result.get(
                "tool_execution_success",
                False,
            ),
            tool_execution_duration_ms=result.get(
                "tool_execution_duration_ms",
                0.0,
            ),
            plan=result.get(
                "plan",
                {},
            ),
            retry_performed=result.get(
                "retry_performed",
                False,
            ),
            retry_count=result.get(
                "retry_count",
                0,
            ),
            retrieval_sufficient=result.get(
                "retrieval_sufficient",
                False,
            ),
            deterministic_answer_used=result.get(
                "deterministic_answer_used",
                False,
            ),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The financial question could not be processed. "
                f"Reason: {error}"
            ),
        ) from error


@router.post(
    "/memory/clear",
)
def clear_financial_memory() -> dict[str, str]:
    """
    Clear the in-memory conversation context.
    """

    financial_agent.clear_memory()

    return {
        "message": "Conversation memory cleared successfully."
    }