"""
BODYBALANCE.AI - Vector Store
LangChain InMemoryVectorStore, load+embed on startup, retrieve(query, k=3)
"""
import os
import json
import streamlit as st
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class BodyBalanceVectorStore:
    def __init__(self):
        # Using OpenAI embedding function
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        kb_path = os.path.join(os.path.dirname(__file__), "../../data/knowledge_base.jsonl")
        if not os.path.exists(kb_path):
            st.error(f"Knowledge base file not found at {kb_path}")
            return

        knowledge_chunks = []
        with open(kb_path, "r", encoding="utf-8") as f:
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
        if self.retriever is None:
            return []
        docs = self.retriever.get_relevant_documents(query, k=k)
        return [doc.page_content for doc in docs]
