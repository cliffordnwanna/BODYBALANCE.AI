"""
BODYBALANCE.AI - Logging and Audit Trail
Handles application logging and audit trail for queries/responses.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

# Audit log file
AUDIT_LOG_FILE = "audit_log.jsonl"
FEEDBACK_LOG_FILE = "feedback_log.json"


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured at {log_level} level")


def log_query(
    session_id: str,
    user_input: str,
    bot_response: str,
    confidence: float,
    matched_question: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a query/response exchange to the audit trail.
    
    Args:
        session_id: Unique session identifier
        user_input: User's query
        bot_response: Bot's response
        confidence: Confidence score of the match
        matched_question: The question that was matched (if any)
        metadata: Additional metadata to log
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": "query",
        "session_id": session_id,
        "query": user_input,
        "response": bot_response[:500],  # Truncate long responses
        "confidence": round(confidence, 4),
        "matched_question": matched_question,
        "metadata": metadata or {}
    }
    
    _append_audit_log(log_entry)


def log_feedback(
    session_id: str,
    helpful: str,
    suggestion: Optional[str] = None,
    exchange_count: int = 0
):
    """
    Log user feedback.
    
    Args:
        session_id: Unique session identifier
        helpful: User's response (Yes/No/Somewhat)
        suggestion: Optional user suggestion
        exchange_count: Number of exchanges in the session
    """
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "helpful": helpful,
        "suggestion": suggestion,
        "exchange_count": exchange_count
    }
    
    _append_feedback_log(feedback_entry)
    
    # Also log to audit trail
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": "feedback",
        "session_id": session_id,
        "data": feedback_entry
    }
    _append_audit_log(log_entry)


def log_admin_action(
    action: str,
    details: Optional[Dict[str, Any]] = None
):
    """
    Log an admin action.
    
    Args:
        action: Description of the action
        details: Additional details
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": "admin_action",
        "action": action,
        "details": details or {}
    }
    
    _append_audit_log(log_entry)


def _append_audit_log(entry: Dict[str, Any]):
    """Append an entry to the audit log file (JSONL format)."""
    try:
        with open(AUDIT_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logging.error(f"Failed to write audit log: {e}")


def _append_feedback_log(entry: Dict[str, Any]):
    """Append feedback to the feedback log file."""
    try:
        # Load existing feedback
        feedback_list = []
        if os.path.exists(FEEDBACK_LOG_FILE):
            with open(FEEDBACK_LOG_FILE, 'r', encoding='utf-8') as f:
                feedback_list = json.load(f)
        
        # Append new feedback
        feedback_list.append(entry)
        
        # Save back
        with open(FEEDBACK_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logging.error(f"Failed to write feedback log: {e}")


def get_feedback_logs() -> list:
    """
    Get all feedback logs.
    
    Returns:
        List of feedback entries
    """
    try:
        if os.path.exists(FEEDBACK_LOG_FILE):
            with open(FEEDBACK_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Failed to read feedback log: {e}")
    return []


def get_audit_logs(limit: int = 100) -> list:
    """
    Get recent audit log entries.
    
    Args:
        limit: Maximum number of entries to return
        
    Returns:
        List of audit log entries (most recent first)
    """
    entries = []
    try:
        if os.path.exists(AUDIT_LOG_FILE):
            with open(AUDIT_LOG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            # Return most recent entries
            return entries[-limit:][::-1]
    except Exception as e:
        logging.error(f"Failed to read audit log: {e}")
    return entries


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]


def get_analytics_summary() -> Dict[str, Any]:
    """
    Generate analytics summary from logs.
    
    Returns:
        Dictionary with analytics data
    """
    audit_logs = get_audit_logs(limit=1000)
    feedback_logs = get_feedback_logs()
    
    # Count queries
    query_logs = [log for log in audit_logs if log.get("event_type") == "query"]
    total_queries = len(query_logs)
    
    # Count matched vs unmatched
    matched = sum(1 for log in query_logs if log.get("matched_question"))
    unmatched = total_queries - matched
    
    # Top questions
    question_counts: Dict[str, int] = {}
    for log in query_logs:
        q = log.get("matched_question")
        if q and q != "ESCALATION_REQUEST":
            question_counts[q] = question_counts.get(q, 0) + 1
    
    top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Unanswered queries
    unanswered = [log.get("query") for log in query_logs if not log.get("matched_question")][:10]
    
    # Feedback summary
    helpful_yes = sum(1 for f in feedback_logs if f.get("helpful") == "Yes")
    helpful_no = sum(1 for f in feedback_logs if f.get("helpful") == "No")
    helpful_somewhat = sum(1 for f in feedback_logs if f.get("helpful") == "Somewhat")
    
    return {
        "total_queries": total_queries,
        "matched_queries": matched,
        "unmatched_queries": unmatched,
        "match_rate": round(matched / total_queries * 100, 1) if total_queries > 0 else 0,
        "top_questions": top_questions,
        "unanswered_queries": unanswered,
        "feedback": {
            "yes": helpful_yes,
            "no": helpful_no,
            "somewhat": helpful_somewhat,
            "total": len(feedback_logs)
        }
    }
