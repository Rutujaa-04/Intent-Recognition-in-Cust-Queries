"""
FastAPI REST API Service for Customer Intent Classification.
Usage:
    uvicorn src.api.app:app --reload --port 8000
"""
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.models.intent_classifier import CustomerIntentClassifier

app = FastAPI(
    title="Customer Query Intent Recognition API",
    description="Production-ready REST API for classifying customer support queries into business intents.",
    version="1.0.0"
)

# Global classifier instance
classifier: Optional[CustomerIntentClassifier] = None


def get_classifier() -> Optional[CustomerIntentClassifier]:
    """Retrieves or lazily initializes the model classifier instance."""
    global classifier
    if classifier is None:
        try:
            classifier = CustomerIntentClassifier()
        except Exception as e:
            print(f"Warning: Model could not be loaded: {e}")
            classifier = None
    return classifier


@app.on_event("startup")
def load_model_on_startup():
    get_classifier()


class QueryRequest(BaseModel):
    query: str = Field(..., example="Can I get a refund for my damaged shoes?", description="The raw customer text query.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top candidates to return.")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Custom fallback confidence threshold.")


class CandidateIntent(BaseModel):
    intent: str
    probability: float


class PredictionResponse(BaseModel):
    query: str
    cleaned_query: str
    intent: str
    raw_intent: str
    confidence: float
    requires_human_escalation: bool
    top_intents: List[CandidateIntent]
    explanation: List[str]


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=100)


@app.get("/")
def root():
    return {
        "service": "Customer Intent Recognition API",
        "status": "running",
        "documentation": "/docs",
        "supported_intents_count": len(config.INTENT_LABELS)
    }


@app.get("/health")
def health_check():
    model = get_classifier()
    model_loaded = model is not None and model.pipeline is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "active_threshold": config.CONFIDENCE_THRESHOLD
    }


@app.get("/intents")
def list_intents():
    return {
        "intents": config.INTENT_DEFINITIONS
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_intent(request: QueryRequest):
    model = get_classifier()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure train.py has been executed."
        )

    if request.threshold is not None:
        model.confidence_threshold = request.threshold
    else:
        model.confidence_threshold = config.CONFIDENCE_THRESHOLD

    result = model.predict(request.query, top_k=request.top_k)
    return result


@app.post("/predict/batch")
def predict_batch(request: BatchQueryRequest):
    model = get_classifier()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    results = model.predict_batch(request.queries)
    return {"predictions": results, "count": len(results)}
