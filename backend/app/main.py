from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.query import router as query_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(query_router)

@app.get("/")
def home():
    return {
        "message": "Agentic Financial Intelligence System API is running!"
    }
