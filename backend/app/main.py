from fastapi import FastAPI
from backend.app.api.health import router as health_router

app = FastAPI()
app.include_router(health_router)

@app.get("/")
def home():
    return {
        "message": "Agentic Financial Intelligence System API is running!"
    }
