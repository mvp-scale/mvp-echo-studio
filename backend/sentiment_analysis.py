"""Per-paragraph sentiment analysis using VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is optimized
for conversational text — a good fit for transcribed audio. Returns
a label (positive/neutral/negative) and compound score per paragraph.
Runs on CPU, no model download required.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_analyzer = None


def _get_analyzer():
    """Lazy-load VADER on first use."""
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"Failed to load VADER: {e}")
    return _analyzer


def _classify(compound: float) -> str:
    """Map compound score to label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def annotate_paragraphs_with_sentiment(paragraphs: List[Dict]) -> List[Dict]:
    """Add sentiment to each paragraph dict.

    Stores ``sentiment: {"label": "positive", "score": 0.42}`` on each
    paragraph dict that has non-empty text.

    Args:
        paragraphs: List of paragraph dicts (with at least a "text" key).

    Returns:
        The same list with "sentiment" added to each dict.
    """
    analyzer = _get_analyzer()
    if analyzer is None:
        return paragraphs

    for para in paragraphs:
        text = para.get("text", "")
        if not text.strip():
            continue
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        para["sentiment"] = {
            "label": _classify(compound),
            "score": round(compound, 3),
        }

    return paragraphs
