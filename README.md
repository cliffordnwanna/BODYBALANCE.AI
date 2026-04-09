# BODYBALANCE.AI - RAG-Powered AI Physiotherapy Concierge

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade RAG (Retrieval-Augmented Generation) AI concierge that turns website visitors into paying patients. Built with LangChain LCEL, OpenAI, and Streamlit.

## Why This Matters for Your Practice

**BODYBALANCE.AI** answers patient questions 24/7 and converts curious visitors into booked appointments via WhatsApp. No missed inquiries. No after-hours voicemails. Just qualified leads delivered directly to your phone.

### Business Impact
- **24/7 Patient Support**: Capture inquiries while you sleep, treat patients, or enjoy weekends
- **Direct WhatsApp Conversions**: Every AI response includes instant booking buttons for in-person (₦150k) and virtual sessions (₦50k)
- **Pre-Qualified Leads**: Patients arrive informed about your services and pricing
- **Zero Infrastructure Cost**: Deploy free on Streamlit Cloud, pay only for OpenAI API usage (~$0.01 per conversation)
- **Medical Safety First**: Built-in emergency detection redirects critical cases to hospital ER immediately

### For Other Healthcare Professionals
This architecture works for any appointment-based practice:
- **Physiotherapists** (current implementation)
- **Dentists, Optometrists, Chiropractors**: Swap the knowledge base, keep the booking flow
- **Psychologists, Nutritionists**: Modify pricing tiers and session types
- **Veterinary Clinics, Wellness Centers**: Adapt the guardrails for your specialty

### Key Features
- **RAG Architecture**: LangChain LCEL with InMemoryVectorStore for semantic document retrieval
- **AI Models**: OpenAI GPT-4o-mini (chat) + text-embedding-3-small (embeddings)
- **Safety Guardrails**: Emergency red-flag detection with human escalation
- **Structured Outputs**: Pydantic-based response formatting with exercises and CTAs
- **WhatsApp Integration**: Direct booking links for in-person and virtual consultations
- **Session Limits**: 3-question limit per session to prevent abuse

### Clinic Information
- **Name**: BodyBalance Physiotherapy Clinic
- **Location**: Lagos, Nigeria
- **Lead Therapist**: Cherry Nwanna (BMR.PT)
- **WhatsApp**: +234 813 629 3596
- **Services**: In-Person Session (₦150,000) | Virtual Consultation (₦50,000)

---

## Architecture

### Tech Stack
| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | LangChain InMemoryVectorStore |
| Framework | LangChain LCEL (LangChain Expression Language) |
| UI | Streamlit |
| Output Parsing | Pydantic v2 |

### Project Structure
```
BODYBALANCE.AI/
├── src/
│   ├── core/
│   │   ├── vector_store.py     # InMemoryVectorStore with OpenAI embeddings
│   │   ├── chains.py           # LangChain LCEL + GPT-4o-mini
│   │   └── guardrails.py       # Emergency red-flag detection
│   ├── ui/
│   │   ├── components.py       # UI render functions
│   │   └── styles.py           # Custom CSS styling
│   └── main.py                 # Streamlit entry point
├── data/
│   └── knowledge_base.jsonl    # Physiotherapy clinic knowledge base
├── .streamlit/
│   ├── config.toml             # Streamlit configuration
│   └── secrets.toml            # API keys (OpenAI)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/cliffordnwanna/BODYBALANCE.AI.git
cd BODYBALANCE.AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up OpenAI API key
echo 'OPENAI_API_KEY = "your-openai-key-here"' > .streamlit/secrets.toml

# 5. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

### 1. Document Ingestion
- Knowledge base stored in `data/knowledge_base.jsonl`
- Documents embedded using OpenAI text-embedding-3-small
- Vector store: LangChain InMemoryVectorStore (in-memory, zero config)

### 2. Query Flow
```
User Query → Guardrails Check → Vector Retrieval → 
RAG Chain (GPT-4o-mini) → Structured Response → WhatsApp CTA
```

### 3. Safety Features
- **Red Flag Detection**: Automatic escalation for emergency keywords
- **3-Question Limit**: Users must refresh after 3 questions
- **Disclaimer**: AI guidance is not a substitute for professional care

### 4. Response Format
```json
{
  "type": "medical_advice|appointment|pricing|general",
  "message": "Professional response text",
  "exercises": [
    {
      "name": "Exercise name",
      "steps": ["Step 1", "Step 2"],
      "reps": "3 sets of 10",
      "caution": "Avoid if pain increases"
    }
  ],
  "cta": "Book your session today"
}
```

---

## Configuration

### Environment Variables
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-openai-key-here"
```

### Knowledge Base
Update `data/knowledge_base.jsonl` with your clinic's information:
```json
{"text": "BodyBalance Clinic offers physiotherapy services in Lagos..."}
```

### Customization
- **Clinic Hours**: Edit `src/main.py` (sidebar section)
- **Pricing**: Edit `src/ui/components.py` (`render_clinic_cta_card`)
- **WhatsApp Number**: Edit `src/ui/components.py` (update +2348136293596)

---

## Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set `OPENAI_API_KEY` in Secrets management
4. Deploy `app.py`

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## RAG Implementation Details

### Why InMemoryVectorStore?
- Zero configuration (no external database)
- Fast startup for small knowledge bases (<1000 docs)
- Perfect for single-tenant clinic deployments

### Why OpenAI?
- GPT-4o-mini: Cost-effective, medical-grade reasoning
- text-embedding-3-small: 1536-dim embeddings, excellent semantic search
- Reliable API with consistent model availability

### Why LangChain LCEL?
- Composable pipeline: `prompt | llm | parser`
- Easy to extend with additional steps
- Production-ready with proper error handling

---

## Safety & Compliance

### Medical AI Guardrails
1. **Emergency Detection**: Hardcoded red-flag keywords trigger immediate escalation
2. **Structured Responses**: Pydantic schema prevents free-form medical advice
3. **Human-in-the-Loop**: WhatsApp booking links for all clinical decisions
4. **Session Limits**: Prevents infinite loops and API abuse

### Data Privacy
- No PII stored in vector store
- No conversation logs retained
- OpenAI API: Zero data retention policy
- WhatsApp: End-to-end encrypted

---

## Customization for Your Practice

### Step 1: Fork and Clone
```bash
git clone https://github.com/cliffordnwanna/BODYBALANCE.AI.git
cd BODYBALANCE.AI
```

### Step 2: Update Knowledge Base
Edit `data/knowledge_base.jsonl` with your clinic's information:
```json
{"text": "Your Clinic Name offers [your services] in [your location]..."}
```

### Step 3: Configure Branding
Edit `src/main.py`:
- Clinic name and tagline (line 54-55)
- Operating hours (line 60)
- Location (line 64)

### Step 4: Set Pricing & WhatsApp
Edit `src/ui/components.py`:
- WhatsApp number in `render_clinic_cta_card()` (line 112-113)
- Service pricing in the green card (line 106-107)

### Step 5: Deploy
```bash
streamlit run app.py
```

### Cost Estimation
- **OpenAI API**: ~$0.01-0.03 per conversation (GPT-4o-mini is cheap)
- **Streamlit Cloud**: Free tier sufficient for small clinics
- **WhatsApp Business**: Free for standard messaging

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Contact

**Clifford Nwanna**  
Email: [nwannachumaclifford@gmail.com](mailto:nwannachumaclifford@gmail.com)

**BodyBalance Clinic**  
WhatsApp: [+234 813 629 3596](https://wa.me/2348136293596)

---

## Changelog

### v3.0.0 (April 2025)
- Complete RAG architecture migration (TF-IDF → LangChain + OpenAI)
- Added InMemoryVectorStore for zero-config vector search
- Implemented GPT-4o-mini with Pydantic structured outputs
- Added emergency red-flag detection guardrails
- Integrated WhatsApp booking CTAs
- Added 3-question session limits
- Removed: Google Sheets, Twilio, ChromaDB dependencies

