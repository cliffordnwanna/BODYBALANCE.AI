"""
BODYBALANCE.AI - Guardrails
keyword regex check runs BEFORE chain; red flags return hardcoded emergency escalation.
"""
import re

RED_FLAGS = [
    r"\bchest\s+pain\b",
    r"\bstroke\b",
    r"\bparalysis\b",
    r"\bcan't\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bsevere\s+bleeding\b",
    r"\bunconscious\b",
    r"\bsudden\s+weakness\b",
    r"\bfainting\b",
    r"\bseizure\b",
    r"\bheart\s+attack\b",
    r"\bdifficulty\s+breathing\b",
    r"\ballergic\s+reaction\b",
    r"\banaphylaxis\b",
    r"\bcyanosis\b",
    r"\bblue\s+lips\b"
]

def check_red_flags(text: str) -> str:
    """Check for red flags and return an emergency message if found."""
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if re.search(flag, text_lower):
            return (
                "EMERGENCY NOTICE\n\n"
                "Your symptoms indicate a potential medical emergency. "
                "Please do not wait. Call emergency services immediately or head to the nearest hospital ER.\n\n"
                "For urgent clinic follow-ups, contact BodyBalance Physiotherapy Clinic at +234 813 629 3596."
            )
    return None
