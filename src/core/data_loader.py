"""
BODYBALANCE.AI - Data Loader
Handles loading Q&A pairs from various sources (Google Sheets, local files, JSON).
"""

import json
import os
import time
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache settings
CACHE_FILE = "qa_cache.json"
CACHE_TIMESTAMP_FILE = "qa_cache_timestamp.txt"


def load_qa_from_text_file(file_path: str) -> Dict[str, str]:
    """
    Load Q&A pairs from a plain text file (legacy format).
    
    Format:
        question: What is BodyBalance?
        answer: BodyBalance is...
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Dictionary mapping questions to answers
    """
    qa_pairs = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.lower().startswith("question:"):
                    question = line.split(':', 1)[1].strip()
                    i += 1
                    if i < len(lines):
                        answer_line = lines[i].strip()
                        if answer_line.lower().startswith("answer:"):
                            answer = answer_line.split(':', 1)[1].strip()
                            qa_pairs[question] = answer
                i += 1
                
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {file_path}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading text file: {e}")
    
    return qa_pairs


def load_qa_from_json(file_path: str) -> Dict[str, str]:
    """
    Load Q&A pairs from a JSON file.
    
    Expected format:
    {
        "qa_pairs": [
            {"question": "...", "answer": "...", "category": "...", "active": true},
            ...
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary mapping questions to answers
    """
    qa_pairs = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        pairs = data.get("qa_pairs", [])
        for pair in pairs:
            # Only load active pairs
            if pair.get("active", True):
                question = pair.get("question", "").strip()
                answer = pair.get("answer", "").strip()
                if question and answer:
                    qa_pairs[question] = answer
                    
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from JSON")
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
    
    return qa_pairs


def load_qa_from_google_sheets(sheet_id: str, credentials_path: str, cache_ttl_minutes: int = 5) -> Dict[str, str]:
    """
    Load Q&A pairs from Google Sheets with caching.
    
    Sheet columns expected: Question | Answer | Category | Tags | Active
    
    Args:
        sheet_id: Google Sheets document ID
        credentials_path: Path to Google service account credentials JSON
        cache_ttl_minutes: Cache time-to-live in minutes
        
    Returns:
        Dictionary mapping questions to answers
    """
    # Check cache first
    cached_data, cache_valid = _check_cache(cache_ttl_minutes)
    if cache_valid and cached_data:
        logger.info("Using cached Q&A data from Google Sheets")
        return cached_data
    
    qa_pairs = {}
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # Define scopes
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        # Authenticate
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open sheet and get all records
        sheet = client.open_by_key(sheet_id).sheet1
        records = sheet.get_all_records()
        
        for record in records:
            # Check if row is active (default to True if column missing)
            active = str(record.get("Active", "TRUE")).upper() in ["TRUE", "YES", "1", ""]
            
            if active:
                question = str(record.get("Question", "")).strip()
                answer = str(record.get("Answer", "")).strip()
                
                if question and answer:
                    qa_pairs[question] = answer
        
        # Update cache
        _save_cache(qa_pairs)
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from Google Sheets")
        
    except ImportError:
        logger.error("gspread or google-auth not installed. Run: pip install gspread google-auth")
    except FileNotFoundError:
        logger.error(f"Credentials file not found: {credentials_path}")
    except Exception as e:
        logger.error(f"Error loading from Google Sheets: {e}")
        # Fall back to cache if available
        if cached_data:
            logger.info("Falling back to cached data")
            return cached_data
    
    return qa_pairs


def _check_cache(ttl_minutes: int) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Check if cache exists and is still valid.
    
    Args:
        ttl_minutes: Cache time-to-live in minutes
        
    Returns:
        Tuple of (cached_data, is_valid)
    """
    try:
        if not os.path.exists(CACHE_FILE) or not os.path.exists(CACHE_TIMESTAMP_FILE):
            return None, False
        
        # Check timestamp
        with open(CACHE_TIMESTAMP_FILE, 'r') as f:
            timestamp = float(f.read().strip())
        
        age_minutes = (time.time() - timestamp) / 60
        is_valid = age_minutes < ttl_minutes
        
        if is_valid:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            return cached_data, True
        else:
            return None, False
            
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")
        return None, False


def _save_cache(qa_pairs: Dict[str, str]):
    """
    Save Q&A pairs to cache.
    
    Args:
        qa_pairs: Dictionary of Q&A pairs to cache
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        with open(CACHE_TIMESTAMP_FILE, 'w') as f:
            f.write(str(time.time()))
            
        logger.info("Cache updated successfully")
        
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def clear_cache():
    """Clear the Q&A cache files."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        if os.path.exists(CACHE_TIMESTAMP_FILE):
            os.remove(CACHE_TIMESTAMP_FILE)
        logger.info("Cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")


def load_qa_pairs(config: dict) -> Dict[str, str]:
    """
    Load Q&A pairs based on configuration.
    
    Tries sources in order:
    1. Google Sheets (if configured)
    2. JSON file (if exists)
    3. Text file (legacy fallback)
    
    Args:
        config: Configuration dictionary with data source settings
        
    Returns:
        Dictionary mapping questions to answers
    """
    qa_pairs = {}
    
    # Try Google Sheets first
    sheet_id = config.get("google_sheet_id", "")
    credentials_path = config.get("google_credentials_path", "")
    
    if sheet_id and credentials_path and os.path.exists(credentials_path):
        qa_pairs = load_qa_from_google_sheets(
            sheet_id=sheet_id,
            credentials_path=credentials_path,
            cache_ttl_minutes=config.get("cache_ttl_minutes", 5)
        )
        if qa_pairs:
            return qa_pairs
    
    # Try JSON file
    json_path = config.get("qa_json_path", "qa_data.json")
    if os.path.exists(json_path):
        qa_pairs = load_qa_from_json(json_path)
        if qa_pairs:
            return qa_pairs
    
    # Fall back to text file
    text_path = config.get("qa_text_path", "training_data.txt")
    if os.path.exists(text_path):
        qa_pairs = load_qa_from_text_file(text_path)
    
    if not qa_pairs:
        logger.warning("No Q&A data loaded from any source!")
    
    return qa_pairs
