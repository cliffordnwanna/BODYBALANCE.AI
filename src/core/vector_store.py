"""
BODYBALANCE.AI - Vector Store
LangChain InMemoryVectorStore, load+embed on startup, retrieve(query, k=3)
"""
import os
import json
from pathlib import Path
import streamlit as st
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class BodyBalanceVectorStore:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def _ensure_initialized(self):
        if self.embeddings is not None:
            return
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.secrets["OPENAI_API_KEY"]
            )
            self._load_knowledge_base()
        except KeyError:
            st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
            st.stop()

    def _load_knowledge_base(self):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        kb_path = BASE_DIR / "data" / "knowledge_base.jsonl"
        if not kb_path.exists():
            st.error(f"Knowledge base file not found at {kb_path}")
            return

        knowledge_chunks = []
        with kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                knowledge_chunks.append({
                    "content": data["text"],
                    "metadata": {"source": "clinic_manual"}
                })

        if knowledge_chunks:
            self.vectorstore = InMemoryVectorStore.from_texts(
                texts=[chunk["content"] for chunk in knowledge_chunks],
                embedding=self.embeddings,
                metadatas=[chunk.get("metadata", {}) for chunk in knowledge_chunks]
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            self.retriever = None

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        self._ensure_initialized()
        if self.retriever is None:
            return []
        docs = self.retriever.get_relevant_documents(query, k=k)
        return [doc.page_content for doc in docs]
