# Multi-Agent Intelligence System

This repository implements a **Multi-Agent Intelligence System** using **FastAPI**, **LangGraph**, and **Groq LLM**.  
It supports dynamic routing between agents for tasks such as **web search** and **summarization**, making it modular, scalable, and production-ready.

---

## 🚀 Features
- ✅ **Multi-Agent Orchestration** with [LangGraph](https://github.com/langchain-ai/langgraph)  
- ✅ **Web Search Agent** powered by [Tavily](https://tavily.com/)  
- ✅ **Summarizer Agent** using Groq LLM  
- ✅ **Router Agent** that decides workflow dynamically  
- ✅ **FastAPI REST API** with `/multi-agents` endpoint  
- ✅ **Dockerized Deployment** with `docker-compose`  
- ✅ Modular project structure for scalability  

---

### 🔧 Agents Overview

Router Agent: Decides whether to search or summarize.

Search Agent: Performs real-time web search using Tavily.

Summarizer Agent: Summarizes text with Groq LLM.

Workflow is managed with a LangGraph state machine.

---

## 📂 Project Structure


├── endpoints/  
│ └── endpoint.py # FastAPI route for multi-agent queries  
├── schemas/  
│ └── schema.py # Pydantic request schema  
├── services/  
│ ├── agents.py # Agent definitions (search, summarizer, router)  
│ ├── graph.py # Build LangGraph state machine  
│ ├── model.py # Groq LLM integration  
│ └── tools.py # Web search & summarization tools  
├── main.py # FastAPI entry point  
├── requirements.txt # Python dependencies  
├── Dockerfile # Docker image build  
├── docker-compose.yml # Multi-service deployment  
└── README.md # Project documentation  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/imran-sony/Multi-Agent_Intelligence_System.git
cd Multi-Agent_Intelligence_System
```

### 2️⃣ Create virtual environment & install dependencies

python -m venv venv  
pip install -r requirements.txt

### 3️⃣ Configure Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=groq_api_key  
TAVILY_API_KEY=tavily_api_key

---

## ▶️ Running the Application

```bash
python main.py
```
