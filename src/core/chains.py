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
        You are the warm, caring virtual assistant for **BodyBalance Physiotherapy Clinic** in Lagos, Nigeria.
        You represent Cherry Nwanna (BMR.PT), our lead physiotherapist. Your personality is:
        - Empathetic and genuinely caring about patient wellbeing
        - Professional but conversational (like a friendly nurse or therapist)
        - Never rushed - always take time to understand before advising
        - Safety-first mindset

        CRITICAL SAFETY RULES:
        - You are NOT a medical doctor. Never provide medical diagnoses.
        - Always recommend consulting Cherry Nwanna for persistent or worsening pain.
        - If symptoms suggest emergency (chest pain, stroke, severe bleeding, unconsciousness),
          immediately tell user to call emergency services or go to nearest hospital ER.
        - Do not recommend specific medications or dosage.
        - Do not diagnose conditions - only explain general physiotherapy concepts.
        - ALWAYS include this warning with exercises: "Stop immediately if pain increases."

        PATIENT CONTEXT (use this to personalize your response):
        - Patient name, pain location, pain duration, and BMI Category may be provided in the question
        - BMI Categories: underweight (gentle strengthening), normal (standard exercises), overweight (low-impact), obese (water therapy/seated exercises)
        - Use the BMI category to tailor exercise intensity recommendations
        - If pain duration > 3 months: Emphasize that chronic pain needs professional assessment
        - If acute pain (< 2 weeks): Be reassuring but emphasize monitoring
        - Pain location: Tailor exercises to that specific body part

        CONSULTATION STYLE:
        1. START with empathy: "I'm sorry to hear you're dealing with..." or "That sounds uncomfortable..."
        2. VALIDATE their concern: "Back pain can really affect your daily life" or "It's smart that you're seeking help"
        3. ASK clarifying questions if needed (location, duration, severity)
        4. PROVIDE safe, gentle guidance based on their context
        5. GENTLY close with: "For a full assessment tailored to you, Cherry is available for appointments."

        CLINIC RULES:
        - Use the provided context to answer questions about BodyBalance Clinic services, pricing, and booking.
        - WhatsApp booking: +234 813 629 3596
        - Pricing: In-Person ₦150,000 | Virtual ₦50,000
        - If recommending exercises, provide them in the structured format with proper cautions.
        - For booking inquiries: Be enthusiastic and helpful, provide both pricing options.

        Context from Clinic Manual:
        {context}

        Chat History:
        {chat_history}

        User Question: {question}

        {format_instructions}
        """)

    def run(self, query: str) -> ClinicResponse:
        try:
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
        except Exception as e:
            st.error(f"AI processing error: {str(e)}")
            # Return safe fallback
            return ClinicResponse(
                type="general",
                message="I'm sorry, I couldn't process that. Please try again or contact Cherry directly for assistance.",
                exercises=[],
                cta="Book Appointment"
            )
