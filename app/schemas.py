"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DetectorResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="1.0 = fully authentic, 0.0 = fully fake/manipulated")
    label: str
    method: str
    details: dict = Field(default_factory=dict)


class UnifiedVerdict(BaseModel):
    verdict: str
    unified_score: float = Field(..., ge=0.0, le=1.0)
    image: Optional[DetectorResult] = None
    text: Optional[DetectorResult] = None
    metadata: Optional[DetectorResult] = None
    explanation: str


class HealthResponse(BaseModel):
    status: str
    use_pretrained_models: bool
