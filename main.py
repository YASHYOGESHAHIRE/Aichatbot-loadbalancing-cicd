
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import AsyncGroq
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = AsyncGroq(api_key=api_key) if api_key else None

client = AsyncGroq(api_key=api_key)
app = FastAPI(title="AWS AI Chatbot")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "instance": os.getenv("INSTANCE_ID", "local")
    }



@app.post("/chat")
async def chat(request: ChatRequest):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured.",
        )

    try:
        result = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
        )

        return {
            "reply": result.choices[0].message.content
        }

    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")

        raise HTTPException(
            status_code=502,
            detail="AI service request failed. Please try again.",
        )

