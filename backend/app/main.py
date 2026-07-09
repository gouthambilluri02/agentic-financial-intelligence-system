from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Agentic Financial Intelligence System API is running!"
    }