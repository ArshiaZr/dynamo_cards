from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor


class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_processor = GeminiProcessor("gemini-pro", "dynamo-cards-421318")

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    youtube_prcessor = YoutubeProcessor(genai_processor=gemini_processor)

    documents = youtube_prcessor.retrieve_youtube_documents(str(request.youtube_link))
    raw_concepts = youtube_prcessor.get_key_concepts(documents, verbose=True)
    
    # Deconstruct
    unique_concepts = {}
    for concept_dict in raw_concepts:
        for key, value in concept_dict.items():
            unique_concepts[key] = value
    
    # Reconstruct
    key_concepts_list = [{"concept": key, "description": value} for key, value in concept_dict.items()]
    
    return {
        "key_concepts": key_concepts_list
    }