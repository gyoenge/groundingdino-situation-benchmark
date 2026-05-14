from pydantic import BaseModel 
from typing import Literal, List, Optional 


PromptType = Literal["object", "composition", "relation", "situation"]
ExpectedType = Literal["positive", "negative", "ambiguous"]


class PromptItem(BaseModel):
    prompt_id: str 
    prompt: str
    type: PromptType 
    expected: ExpectedType


class BBoxAnnotation(BaseModel):
    label: str
    bbox: List[float] # [x1, y1, x2, y2]


class BenchmarkSample(BaseModel):
    image_id: str
    image_path: str 
    category: str 
    prompts: List[PromptItem]
    annotations: List[BBoxAnnotation]


class PredictionItem(BaseModel):
    bbox: List[float]
    score: float 
    phrase: Optional[str] = None 


class PromptPrediction(BaseModel):
    image_id: str 
    prompt_id: str 
    prompt: str
    predictions: List[PredictionItem]

