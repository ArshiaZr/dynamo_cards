from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor


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

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    processor = YoutubeProcessor()
    result = processor.retrieve_youtube_documents(str(request.youtube_link), True)


    return { "result": result }