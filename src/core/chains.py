"""
BODYBALANCE.AI - Chains
LangChain LCEL, OpenAI GPT-4o-mini, structured JSON output via PydanticOutputParser.
"""
from typing import List, Optional
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Exercise(BaseModel):
    name: str = Field(description="Name of the exercise")
    steps: List[str] = Field(description="List of steps to perform the exercise")
    reps: str = Field(description="Recommended repetitions and sets")
    caution: str = Field(description="Caution or contraindications")

class ClinicResponse(BaseModel):
    type: str = Field(description="Response type: 'medical_advice', 'appointment', 'pricing', 'general'")
    message: str = Field(description="Response message from BodyBalance Physiotherapy Clinic")
    exercises: List[Exercise] = Field(description="List of relevant exercises for the user", default_factory=list)
    cta: str = Field(description="Call to action text for the user")

class BodyBalanceChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0.3
        )
        self.parser = PydanticOutputParser(pydantic_object=ClinicResponse)
        self.chat_history = []
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are the AI Concierge for **BodyBalance Physiotherapy Clinic**.
        Your goal is to provide educational physiotherapy information and clinic services to patients.

        CRITICAL SAFETY RULES:
        - You are NOT a medical doctor. Never provide medical diagnoses.
        - Always recommend consulting a licensed physiotherapist for persistent pain.
        - If symptoms suggest emergency (chest pain, stroke, severe bleeding, unconsciousness),
          immediately tell user to call emergency services or go to nearest hospital ER.
        - Do not recommend specific medications or dosage.
        - Do not diagnose conditions - only explain general physiotherapy concepts.

        CLINIC RULES:
        - Use the provided context to answer questions about BodyBalance Clinic services, pricing, and booking.
        - If the question is about a common condition but not in the context, use your general wellness and physiotherapy knowledge.
        - If the user needs to book an appointment, use the 'appointment' type and the provided WhatsApp CTA.
        - If recommending exercises, provide them in the structured format with proper cautions.
        - Stay professional, empathetic, and clear.
        - Always end with a recommendation to consult Cherry Nwanna (BMR.PT) for personalized treatment.

        Context from Clinic Manual:
        {context}

        Chat History:
        {chat_history}

        User Question: {question}

        {format_instructions}
        """)

    def run(self, query: str) -> ClinicResponse:
        # Retrieve context using LangChain retriever
        context_docs = self.vector_store.retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Format prompt
        chain = self.prompt | self.llm | self.parser
        
        # Get response
        response = chain.invoke({
            "context": context,
            "chat_history": self.chat_history,
            "question": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Save to memory
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response.message))
        
        return response
