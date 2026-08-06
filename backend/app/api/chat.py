from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(request: AskRequest):

    return {
        "answer": f"You asked: {request.question}",
        "sources": []
    }
