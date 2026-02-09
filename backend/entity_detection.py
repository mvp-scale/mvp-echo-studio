"""Entity detection and topic extraction using spaCy.

Provides NER-based entity extraction and noun-phrase topic extraction
from transcription segments. Runs on CPU, lightweight.
"""

import logging
from typing import List, Dict, Optional, Set
from collections import Counter

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model on first use."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
    return _nlp


# Entity types relevant for transcription analysis
ENTITY_TYPES = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Times",
    "MONEY": "Money",
    "CARDINAL": "Numbers",
    "ORDINAL": "Ordinals",
    "PRODUCT": "Products",
    "EVENT": "Events",
    "WORK_OF_ART": "Works",
    "LAW": "Legal",
    "NORP": "Groups",
}


def extract_entities(segments: list) -> Optional[List[Dict]]:
    """Extract named entities from transcription segments.

    Args:
        segments: List of WhisperSegment objects with text.

    Returns:
        List of entity dicts with text, label, start_segment, count,
        or None if spaCy is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return None

    # Build full text from segments for better NER context
    full_text = " ".join(seg.text.strip() for seg in segments)
    doc = nlp(full_text)

    # Deduplicate and count entities
    entity_counts: Counter = Counter()
    entity_labels: Dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        key = ent.text.strip().lower()
        if not key:
            continue
        entity_counts[key] += 1
        # Keep the original casing from first occurrence
        if key not in entity_labels:
            entity_labels[key] = (ent.text.strip(), ent.label_)

    # Build results sorted by frequency
    results = []
    for key, count in entity_counts.most_common():
        text, label = entity_labels[key]
        results.append({
            "text": text,
            "label": label,
            "category": ENTITY_TYPES.get(label, label),
            "count": count,
        })

    return results


def redact_entities(
    segments: list,
    entity_types: Set[str],
) -> list:
    """Redact entities of specified types from segment text.

    Args:
        segments: List of WhisperSegment objects.
        entity_types: Set of spaCy entity labels to redact (e.g. {"PERSON", "ORG"}).

    Returns:
        The same segments list with text modified in-place.
    """
    nlp = _get_nlp()
    if nlp is None:
        return segments

    for seg in segments:
        doc = nlp(seg.text)
        new_text = seg.text
        # Replace entities in reverse order to preserve character positions
        for ent in reversed(doc.ents):
            if ent.label_ in entity_types:
                replacement = f"[{ent.label_}]"
                new_text = new_text[:ent.start_char] + replacement + new_text[ent.end_char:]
        seg.text = new_text

    return segments


def annotate_paragraphs_with_entities(paragraphs: List[Dict]) -> List[Dict]:
    """Add entity_counts to each paragraph dict.

    Runs spaCy NER on each paragraph's text and stores per-type counts.
    GPE and LOC are merged into a single "Locations" bucket.

    Args:
        paragraphs: List of paragraph dicts (with at least a "text" key).

    Returns:
        The same list with "entity_counts" added to each dict.
    """
    nlp = _get_nlp()
    if nlp is None:
        return paragraphs

    # Map spaCy labels to canonical display keys (merge GPE+LOC)
    label_map = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "GPE",
        "LOC": "GPE",  # merge LOC into GPE bucket
        "DATE": "DATE",
        "TIME": "TIME",
        "MONEY": "MONEY",
        "CARDINAL": "CARDINAL",
        "ORDINAL": "ORDINAL",
        "PRODUCT": "PRODUCT",
        "EVENT": "EVENT",
        "WORK_OF_ART": "WORK_OF_ART",
        "LAW": "LAW",
        "NORP": "NORP",
    }

    for para in paragraphs:
        text = para.get("text", "")
        if not text.strip():
            continue
        doc = nlp(text)
        counts: Counter = Counter()
        for ent in doc.ents:
            canonical = label_map.get(ent.label_)
            if canonical:
                counts[canonical] += 1
        if counts:
            para["entity_counts"] = dict(counts)

    return paragraphs


def extract_topics(segments: list, top_n: int = 10) -> Optional[List[Dict]]:
    """Extract topics from transcription via noun phrase frequency.

    Uses spaCy's noun chunk extraction to find the most-discussed
    subjects in the transcript.

    Args:
        segments: List of WhisperSegment objects with text.
        top_n: Maximum number of topics to return.

    Returns:
        List of topic dicts with text and count,
        or None if spaCy is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return None

    full_text = " ".join(seg.text.strip() for seg in segments)
    doc = nlp(full_text)

    # Extract noun chunks, normalize to lowercase, filter short/stopword-only
    chunk_counts: Counter = Counter()
    chunk_display: Dict[str, str] = {}
    for chunk in doc.noun_chunks:
        # Strip leading determiners/pronouns for cleaner topics
        tokens = [t for t in chunk if not t.is_stop and not t.is_punct and len(t.text) > 1]
        if not tokens:
            continue
        key = " ".join(t.lemma_.lower() for t in tokens)
        if len(key) < 2:
            continue
        chunk_counts[key] += 1
        if key not in chunk_display:
            chunk_display[key] = " ".join(t.text for t in tokens)

    results = []
    for key, count in chunk_counts.most_common(top_n):
        if count < 2:
            break  # Only include topics mentioned more than once
        results.append({
            "text": chunk_display[key],
            "count": count,
        })

    return results
