# app/core/models.py
from pydantic import BaseModel
from typing import Dict

class StoryRequest(BaseModel):
    text: str

class CoherenceResponse(BaseModel):
    is_coherent: bool
    analysis_results: Dict[str, bool]
    feedback: str

# app/services/ml_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple

class MistralService:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def get_llm_feedback(self, story: str, analysis_results: Dict[str, bool]) -> Tuple[bool, str]:
        prompt = f"""
        Analyze the following story for signs of fabrication. Consider these aspects:
        - Personal Context: {"Present" if analysis_results['has_personal_context'] else "Missing"}
        - Sensory Details: {"Present" if analysis_results['has_sensory_details'] else "Missing"}
        - Specificity: {"Present" if analysis_results['is_specific'] else "Missing"}
        - Internal Consistency: {"Consistent" if analysis_results.get('is_consistent', True) else "Inconsistent"}
        - Emotional Alignment: {"Aligned" if analysis_results.get('emotion_alignment', True) else "Misaligned"}

        Story: {story}

        Determine if the story might be fabricated or misleading. Provide a detailed explanation focusing on:
        1. Logical inconsistencies or contradictions.
        2. Lack of verifiable personal context.
        3. Inconsistencies in emotional cues.
        4. Any missing or vague details that may indicate fabrication.

        Offer specific feedback on what parts of the story might be untruthful and why.
        """


        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        is_coherent = "coherent" in response.lower() and "not coherent" not in response.lower()
        
        return is_coherent, response

# app/api/endpoints.py
from fastapi import APIRouter, HTTPException
from app.core.models import StoryRequest, CoherenceResponse
from app.services.ml_model import MistralService
from sentiment_analysis.personal_context import has_personal_context
from sentiment_analysis.sensory_details import has_sensory_details
from sentiment_analysis.specificity import specificity_spacy
from sentiment_analysis.causal_coherence import is_causally_coherent

router = APIRouter()
ml_service = MistralService()

@router.post("/analyze", response_model=CoherenceResponse)
async def analyze_story(request: StoryRequest):
    try:
        analysis_results = {
            "has_personal_context": has_personal_context(request.text),
            "has_sensory_details": has_sensory_details(request.text),
            "is_specific": specificity_spacy(request.text),
            "is_causally_coherent": is_causally_coherent(request.text)
            
        }
        
        is_coherent, feedback = ml_service.get_llm_feedback(request.text, analysis_results)
        
        return CoherenceResponse(
            is_coherent=is_coherent,
            analysis=analysis_results,
            feedback=feedback
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

# app/main.py
from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Story Coherence API")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)