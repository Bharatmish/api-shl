from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from combined_recommender import smart_recommend

app = FastAPI(title="SHL Assessment Recommender")

# Optional: Allow frontend to access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationResponse(BaseModel):
    name: str
    url: str
    duration: str
    type: str
    remote: str
    adaptive: str
    description: str

@app.get("/recommend", response_model=list[RecommendationResponse])
def recommend(query: str = Query(..., description="Natural language job query")):
    results = smart_recommend(query, top_k=10)
    return results
