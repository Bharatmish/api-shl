from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

# Replace this import with your actual smart_recommend path
from src.combined_recommender import smart_recommend

app = FastAPI(title="SHL Assessment Recommender")

# CORS setup for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Request model
class RecommendRequest(BaseModel):
    query: str

# Response model for a single assessment
class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

# Final response model
class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]

# Recommend endpoint
@app.post("/recommend", response_model=RecommendResponse)
def recommend(data: RecommendRequest):
    raw_results = smart_recommend(data.query, top_k=10)

    formatted_results = []
    for r in raw_results:
        # Handle duration string like "18 mins" safely
        duration_raw = str(r.get("duration", ""))
        duration_match = re.search(r"\d+", duration_raw)
        duration_int = int(duration_match.group()) if duration_match else 0

        formatted_results.append({
            "url": r.get("url"),
            "adaptive_support": r.get("adaptive", "No"),
            "description": r.get("description", ""),
            "duration": duration_int,
            "remote_support": r.get("remote", "No"),
            "test_type": r.get("type") if isinstance(r.get("type"), list) else [r.get("type")]
        })

    return {"recommended_assessments": formatted_results}
