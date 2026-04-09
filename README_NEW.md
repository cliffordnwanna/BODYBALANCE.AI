# 🏥 BodyBalance AI Concierge: Enterprise Physiotherapy RAG

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.12-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)](https://openai.com/)

**BodyBalance AI Concierge** is a production-grade RAG (Retrieval-Augmented Generation) solution designed for **BodyBalance Physiotherapy Clinic**. It transforms static clinic protocols into an intelligent, clinical-grade conversational agent that assists patients with injury advice, exercise recommendations, and booking.

---

## 🏗️ System Architecture

The platform follows a modern **RAG (Retrieval-Augmented Generation)** architecture:

1.  **Ingestion**: 40+ clinical protocol chunks are loaded from `knowledge_base.jsonl`.
2.  **Vectorization**: Documents are embedded using OpenAI's `text-embedding-3-small` model.
3.  **Storage**: High-speed, in-memory vector search powered by **ChromaDB**.
4.  **Guardrails**: A deterministic regex-based safety layer intercepts medical emergencies before they reach the LLM.
5.  **Reasoning**: **LangChain LCEL** orchestrates the conversation, using `gpt-4o-mini` for context-aware response generation.
6.  **Structured Output**: Pydantic models force the LLM to return validated JSON, ensuring the UI can render interactive exercise cards and CTAs.

---

## 🌟 Key Features

-   **🎯 Semantic Clinic Search**: Understands patient intent beyond keywords (e.g., "my leg is tingling" → Sciatica protocol).
-   **🏃 Interactive Exercise Cards**: Structured recommendations including steps, reps, and clinical cautions.
-   **🚨 Emergency Guardrails**: Automatic detection of "red flag" symptoms with immediate escalation to human services.
-   **📊 Clinical Analytics**: In-memory tracking of patient query types and engagement metrics.
-   **🟢 WhatsApp Integration**: One-click booking CTA tailored for the Nigerian wellness market.

---

## 🛠️ Tech Stack

-   **Orchestration**: LangChain (LCEL)
-   **LLM**: OpenAI GPT-4o-mini
-   **Vector DB**: ChromaDB (In-memory)
-   **Frontend**: Streamlit (Custom Medical Theme)
-   **Governance**: Pydantic (Schema Validation), Custom Regex Guardrails

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key

### 2. Installation
```bash
git clone https://github.com/cliffordnwanna/BODYBALANCE.AI.git
cd BODYBALANCE.AI
pip install -r requirements.txt
```

### 3. Configuration
Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

### 4. Running the App
```bash
streamlit run app.py
```

---

## ⚖️ AI Governance & Safety
This project implements **Medical AI Governance** best practices:
- **PII Protection**: No patient data is stored on disk or used for model training.
- **Scope Limitation**: The AI is restricted to physiotherapy and wellness domains.
- **Human-in-the-loop**: All high-risk queries are redirected to clinical staff.

---

## 👨‍💻 Author
**Chukwuma Clifford Nwanna**  
*AI Engineer | Software Developer*  
[LinkedIn](https://linkedin.com/in/chukwumanwanna) | [GitHub](https://github.com/cliffordnwanna)
