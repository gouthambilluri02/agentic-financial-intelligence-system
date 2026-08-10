# 🚀 Agentic Financial Intelligence System
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

> An enterprise-grade AI-powered financial intelligence platform that combines Retrieval-Augmented Generation (RAG), intelligent tool routing, semantic document search, and local LLM inference to analyze financial reports and answer complex financial questions.

---

## ✨ Overview

The Agentic Financial Intelligence System enables users to interact with company financial reports using natural language.

Instead of relying on a single chatbot response, the system intelligently determines the user's intent, selects the appropriate financial analysis tool, retrieves supporting evidence from financial filings, and generates grounded answers with citations.

The application features a modern React dashboard, a FastAPI backend, ChromaDB vector search, and Ollama-powered local LLM inference.

---

# Features

- Financial report question answering
- Revenue comparison between companies
- Financial ratio calculations
- Risk analysis from SEC filings
- Semantic document retrieval
- Intelligent tool routing
- Retrieval-Augmented Generation (RAG)
- Conversation memory
- Source attribution
- Local LLM inference using Ollama
- Responsive React dashboard
- Docker support

---


# Architecture
📄 [View the editable Draw.io architecture](docs/architecture.drawio)

![System Architecture](docs/images/architecture.png)

```mermaid
flowchart TD

    U["User"] --> FE["React + TypeScript Frontend"]

    FE -->|POST /api/v1/query| API["FastAPI API Layer"]

    API --> AGENT["Financial Intelligence Agent"]

    AGENT --> CD["Company Detector"]
    AGENT --> ID["Intent Detector"]
    AGENT --> MD["Financial Metric Detector"]

    CD --> ROUTER["Tool Router"]
    ID --> ROUTER
    MD --> ROUTER

    ROUTER --> PLAN["Execution Planner"]

    PLAN --> EXEC["Tool Executor"]

    EXEC --> CALC["Financial Calculator"]
    EXEC --> COMP["Company Comparison Tool"]
    EXEC --> RISK["Risk Analysis Tool"]
    EXEC --> RET["Document Retrieval"]

    RET --> ORCH["Retrieval Orchestrator"]

    ORCH --> VECTOR["ChromaDB Vector Database"]

    DOCS["Financial Reports - 10-K PDFs"] --> VECTOR

    VECTOR --> ORCH

    ORCH --> SOURCES["Source Ranking"]

    SOURCES --> AGENT

    CALC --> AGENT
    COMP --> AGENT
    RISK --> AGENT

    AGENT --> DET["Deterministic Response Builder"]

    AGENT --> LLM["Ollama - Qwen2.5:7B"]

    DET --> RESPONSE["Final Response"]
    LLM --> RESPONSE

    RESPONSE --> TRACE["Execution Trace"]

    TRACE --> API
    API --> FE

    FE --> RESULT["Dashboard"]
```

---

# Tech Stack
---


---



## Frontend

- React
- TypeScript
- Vite
- Axios
- CSS

## Backend

- FastAPI
- Python
- Pydantic
- Uvicorn

## AI & Retrieval

- Ollama
- Qwen2.5:7B
- ChromaDB
- Sentence Transformers

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

# Project Structure

```
backend/
│
├── agents/
├── api/
├── core/
├── data/
├── models/
├── schemas/
├── services/
├── tools/
└── main.py

frontend/
│
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/
│   ├── hooks/
│   └── assets/

docker-compose.yml
README.md
requirements.txt
```

---

# Current Capabilities

- Compare company revenues
- Analyze financial risks
- Retrieve evidence from annual reports
- Explain business metrics
- Provide cited answers
- Display retrieved sources
- Show execution details
- Track selected tools

---

# Example Questions

```
Compare Apple and Microsoft revenue

What risks did Microsoft disclose?

What was Apple's operating income?

Compare NVIDIA and AMD revenue.

What cybersecurity risks are mentioned in Microsoft's report?
```

---

# Screenshots

> Screenshots will be added soon.

## Dashboard

![Dashboard](docs/images/landing-page.png)

---

## Revenue Comparison

![Revenue Comparison](docs/images/revenue-comparison.png)

---

## Risk Analysis

![Risk Analysis](docs/images/risk-analysis.png)



## Source Citations

![Sources](docs/images/sources-panel.png)

---

## System Architecture

![Architecture](docs/images/architecture.png)


---

# Installation

Clone the repository

```bash
git clone https://github.com/gouthambilluri02/agentic-financial-intelligence-system.git
```

Install frontend

```bash
cd frontend
npm install
npm run dev
```

Run backend

```bash
docker compose up --build
```

Open

```
http://localhost:5173
```

---

# Future Enhancements

- Multi-agent collaboration
- SQL financial database integration
- PDF upload from UI
- Graph generation
- Financial trend visualization
- Streaming responses
- Authentication
- Cloud deployment (AWS/Azure)
- Portfolio analytics
- Earnings call analysis

---

# License

MIT License

---

## Author

**Goutham Billuri**

AI / ML Engineer

Specializing in Agentic AI, RAG Systems, LLM Applications, and Production AI Platforms.