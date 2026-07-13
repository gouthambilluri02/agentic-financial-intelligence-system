from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.query import router as query_router

app = FastAPI()
app.include_router(health_router)
app.include_router(query_router)

@app.get("/")
def home():
    return {
        "message": "Agentic Financial Intelligence System API is running!"
    }
