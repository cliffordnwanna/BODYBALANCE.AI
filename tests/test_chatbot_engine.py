"""
BODYBALANCE.AI - Unit Tests for Chatbot Engine
Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.chatbot_engine import ChatbotEngine, Session, ESCALATION_KEYWORDS


# Sample Q&A pairs for testing
SAMPLE_QA_PAIRS = {
    "What is BodyBalance?": "BodyBalance is a wellness brand offering innovative pain relief solutions.",
    "What products do you offer?": "We offer ergonomic supports, back stretchers, orthopedic pillows, and knee braces.",
    "How long does delivery take?": "Delivery times vary by location. Estimated times are displayed during checkout.",
    "What is your return policy?": "Products can be returned within 7 days for a full refund or exchange.",
    "How can I contact support?": "Contact our support team via email at support@bodybalance.com"
}


class TestChatbotEngine:
    """Tests for the ChatbotEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a chatbot engine with sample data."""
        return ChatbotEngine(qa_pairs=SAMPLE_QA_PAIRS, similarity_threshold=0.3)
    
    def test_initialization(self, engine):
        """Test engine initializes correctly."""
        assert len(engine.questions) == 5
        assert engine.similarity_threshold == 0.3
        assert engine.tfidf_matrix is not None
    
    def test_empty_qa_pairs(self):
        """Test engine handles empty Q&A pairs."""
        engine = ChatbotEngine(qa_pairs={}, similarity_threshold=0.3)
        assert len(engine.questions) == 0
        assert engine.tfidf_matrix is None
    
    def test_exact_match(self, engine):
        """Test exact question matching."""
        answer, confidence, matched = engine.find_answer("What is BodyBalance?")
        assert answer is not None
        assert confidence > 0.8
        assert matched == "What is BodyBalance?"
    
    def test_similar_match(self, engine):
        """Test similar question matching."""
        answer, confidence, matched = engine.find_answer("Tell me about BodyBalance")
        assert answer is not None
        assert confidence > 0.3
        assert "BodyBalance" in answer
    
    def test_no_match(self, engine):
        """Test handling of unmatched queries."""
        answer, confidence, matched = engine.find_answer("What is the weather today?")
        # Should return None if below threshold
        assert matched is None or confidence < 0.3
    
    def test_empty_input(self, engine):
        """Test handling of empty input."""
        answer, confidence, matched = engine.find_answer("")
        assert answer is None
        assert confidence == 0.0
        assert matched is None
        
        answer, confidence, matched = engine.find_answer("   ")
        assert answer is None
    
    def test_escalation_detection(self, engine):
        """Test human escalation detection."""
        escalation_phrases = [
            "I want to speak to someone",
            "Can I talk to a human?",
            "Get me an agent",
            "Help me please",
            "Call me back"
        ]
        
        for phrase in escalation_phrases:
            answer, confidence, matched = engine.find_answer(phrase)
            assert matched == "ESCALATION_REQUEST"
            assert "support" in answer.lower() or "contact" in answer.lower()
    
    def test_suggestions(self, engine):
        """Test suggestion generation."""
        suggestions = engine.get_suggestions("product information", n=3)
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
    
    def test_update_qa_pairs(self, engine):
        """Test updating Q&A pairs."""
        new_pairs = {
            "New question?": "New answer!"
        }
        engine.update_qa_pairs(new_pairs)
        
        assert len(engine.questions) == 1
        answer, _, _ = engine.find_answer("New question?")
        assert answer == "New answer!"
    
    def test_preprocess_text(self, engine):
        """Test text preprocessing."""
        tokens = engine._preprocess_text("What is the BEST product?")
        assert "what" not in tokens  # stopword removed
        assert "best" in tokens
        assert "product" in tokens


class TestSession:
    """Tests for the Session class."""
    
    @pytest.fixture
    def session(self):
        """Create a test session."""
        return Session(session_id="test123", max_history=5)
    
    def test_initialization(self, session):
        """Test session initializes correctly."""
        assert session.session_id == "test123"
        assert session.max_history == 5
        assert len(session.history) == 0
        assert session.exchange_count == 0
        assert session.feedback_collected is False
    
    def test_add_exchange(self, session):
        """Test adding exchanges to history."""
        session.add_exchange(
            user_input="Hello",
            bot_response="Hi there!",
            confidence=1.0,
            matched_question="Greeting"
        )
        
        assert len(session.history) == 1
        assert session.exchange_count == 1
        assert session.history[0]["user_input"] == "Hello"
    
    def test_max_history_limit(self, session):
        """Test history doesn't exceed max limit."""
        for i in range(10):
            session.add_exchange(f"Question {i}", f"Answer {i}", 0.5)
        
        assert len(session.history) == 5  # max_history
        assert session.exchange_count == 10
        assert session.history[0]["user_input"] == "Question 5"  # oldest kept
    
    def test_get_context(self, session):
        """Test getting conversation context."""
        session.add_exchange("Q1", "A1", 0.5)
        session.add_exchange("Q2", "A2", 0.5)
        session.add_exchange("Q3", "A3", 0.5)
        
        context = session.get_context(n=2)
        assert len(context) == 2
        assert context == ["Q2", "Q3"]
    
    def test_should_ask_feedback(self, session):
        """Test feedback timing logic."""
        assert session.should_ask_feedback() is False
        
        # Add 3 exchanges
        for i in range(3):
            session.add_exchange(f"Q{i}", f"A{i}", 0.5)
        
        assert session.should_ask_feedback() is True
        
        # Mark feedback collected
        session.mark_feedback_collected()
        assert session.should_ask_feedback() is False
    
    def test_clear_session(self, session):
        """Test clearing session."""
        session.add_exchange("Q", "A", 0.5)
        session.mark_feedback_collected()
        
        session.clear()
        
        assert len(session.history) == 0
        assert session.exchange_count == 0
        assert session.feedback_collected is False


class TestPreprocessing:
    """Tests for text preprocessing."""
    
    @pytest.fixture
    def engine(self):
        return ChatbotEngine(qa_pairs=SAMPLE_QA_PAIRS, similarity_threshold=0.3)
    
    def test_lowercase(self, engine):
        """Test text is lowercased."""
        tokens = engine._preprocess_text("HELLO WORLD")
        assert all(t.islower() for t in tokens)
    
    def test_stopword_removal(self, engine):
        """Test stopwords are removed."""
        tokens = engine._preprocess_text("What is the best way to do this?")
        stopwords = ["what", "is", "the", "to", "this"]
        for sw in stopwords:
            assert sw not in tokens
    
    def test_punctuation_removal(self, engine):
        """Test punctuation is handled."""
        tokens = engine._preprocess_text("Hello, world! How are you?")
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens
    
    def test_alphanumeric_only(self, engine):
        """Test only alphanumeric tokens are kept."""
        tokens = engine._preprocess_text("Product #123 costs $50")
        assert "#123" not in tokens
        assert "$50" not in tokens
        assert "123" in tokens or "product" in tokens


class TestEscalation:
    """Tests for escalation keyword detection."""
    
    def test_escalation_keywords_exist(self):
        """Test escalation keywords are defined."""
        assert len(ESCALATION_KEYWORDS) > 0
        assert "human" in ESCALATION_KEYWORDS
        assert "agent" in ESCALATION_KEYWORDS
    
    def test_escalation_case_insensitive(self):
        """Test escalation detection is case-insensitive."""
        engine = ChatbotEngine(qa_pairs=SAMPLE_QA_PAIRS, similarity_threshold=0.3)
        
        # Test various cases
        for phrase in ["SPEAK TO SOMEONE", "Speak To Someone", "speak to someone"]:
            assert engine._check_escalation(phrase) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
