"""FastAPI application exposing the authentication engine over REST.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    GET  /health
    POST /v1/authenticate/image     (multipart file upload)
    POST /v1/authenticate/metadata  (multipart file upload)
    POST /v1/authenticate/text      (JSON: {"text": "...", image optional as multipart in /v1/authenticate)
    POST /v1/authenticate           (multipart file upload + optional `caption` form field) -> full pipeline
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.detectors.image_detector import ImageAuthDetector
from app.detectors.metadata_detector import MetadataAuthDetector
from app.detectors.text_detector import TextConsistencyDetector
from app.pipeline.consistency_pipeline import CrossModalConsistencyPipeline
from app.schemas import DetectorResult, HealthResponse, UnifiedVerdict

app = FastAPI(
    title="Multimodal Synthetic Content Authentication Engine",
    description=(
        "Learning-scale reimplementation of a cross-modal authenticity pipeline: "
        "ViT-backed image analysis, CLIP-based image/text consistency checking, and "
        "rule-based EXIF metadata auditing, fused into a single verdict."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = CrossModalConsistencyPipeline()
_image_detector = ImageAuthDetector()
_metadata_detector = MetadataAuthDetector()
_text_detector = TextConsistencyDetector()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", use_pretrained_models=settings.use_pretrained_models)


@app.post("/v1/authenticate/image", response_model=DetectorResult)
async def authenticate_image(file: UploadFile = File(...)) -> DetectorResult:
    image_bytes = await file.read()
    return _image_detector.analyze(image_bytes)


@app.post("/v1/authenticate/metadata", response_model=DetectorResult)
async def authenticate_metadata(file: UploadFile = File(...)) -> DetectorResult:
    image_bytes = await file.read()
    return _metadata_detector.analyze(image_bytes)


@app.post("/v1/authenticate/text", response_model=DetectorResult)
async def authenticate_text(text: str = Form(...), file: Optional[UploadFile] = File(None)) -> DetectorResult:
    image_bytes = await file.read() if file is not None else None
    return _text_detector.analyze(text, image_bytes=image_bytes)


@app.post("/v1/authenticate", response_model=UnifiedVerdict)
async def authenticate(file: UploadFile = File(...), caption: Optional[str] = Form(None)) -> UnifiedVerdict:
    image_bytes = await file.read()
    return _pipeline.run(image_bytes, caption=caption)
