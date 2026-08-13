from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils

app = FastAPI(title="AI Knowledge Retention Predictor API")

class RetentionRequest(BaseModel):
    topic: str
    last_revised_days: int
    quiz_score: float
    difficulty: str

class QuizRequest(BaseModel):
    topic: str
    difficulty: str
    num_questions: int = 3

@app.get("/")
def read_root():
    return {"message": "AI Knowledge Retention Predictor API is running."}

@app.post("/predict-retention")
def predict_retention(data: RetentionRequest):
    try:
        result = utils.analyze_retention_risk(
            topic=data.topic,
            last_revised_days=data.last_revised_days,
            quiz_score=data.quiz_score,
            difficulty=data.difficulty
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz")
def generate_quiz(data: QuizRequest):
    try:
        result = utils.generate_topic_quiz(
            topic=data.topic,
            difficulty=data.difficulty,
            num_questions=data.num_questions
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))