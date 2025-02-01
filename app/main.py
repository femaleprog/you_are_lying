from fastapi import FastAPI
from pydantic import BaseModel
from app.services.ml_model import MistralService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
mistral_service = MistralService()

# Request body
class StoryRequest(BaseModel):
    story: str

#api route to analyse a story 
@app.post("/analyze-story")
def analyze_story(request: StoryRequest):
    story = request.story
    is_coherent, feedback = mistral_service.get_llm_feedback(story)
    return {"is_coherent": is_coherent, "feedback": feedback}

# health check route
@app.get("/")
def read_root():
    return {"status": "ok"}