from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class WhisperSegment(BaseModel):
    """Represents a segment in the transcription"""
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.1
    speaker: Optional[str] = None


class SentimentResult(BaseModel):
    """Sentiment analysis result for a paragraph."""
    label: str  # "positive", "neutral", "negative"
    score: float  # VADER compound score: -1.0 to +1.0


class Paragraph(BaseModel):
    """A group of consecutive same-speaker segments."""
    speaker: Optional[str] = None
    start: float
    end: float
    text: str
    segment_count: int
    entity_counts: Optional[Dict[str, int]] = None
    sentiment: Optional[SentimentResult] = None


class SpeakerStatistics(BaseModel):
    """Per-speaker talk time and word count."""
    duration: float
    percentage: float
    word_count: int


class Statistics(BaseModel):
    """Aggregate speaker statistics for the transcription."""
    speakers: Dict[str, SpeakerStatistics]
    total_speakers: int


class Entity(BaseModel):
    """A named entity detected in the transcript."""
    text: str
    label: str
    category: str
    count: int


class Topic(BaseModel):
    """A topic extracted from the transcript."""
    text: str
    count: int


class TranscriptionResponse(BaseModel):
    """Response format for transcription"""
    text: str
    segments: Optional[List[WhisperSegment]] = None
    language: Optional[str] = None
    task: str = "transcribe"
    duration: Optional[float] = None
    model: Optional[str] = None
    paragraphs: Optional[List[Paragraph]] = None
    statistics: Optional[Statistics] = None
    entities: Optional[List[Entity]] = None
    topics: Optional[List[Topic]] = None

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        if not self.segments:
            result.pop("segments", None)
        if not self.paragraphs:
            result.pop("paragraphs", None)
        if not self.statistics:
            result.pop("statistics", None)
        if not self.entities:
            result.pop("entities", None)
        if not self.topics:
            result.pop("topics", None)
        return result


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
