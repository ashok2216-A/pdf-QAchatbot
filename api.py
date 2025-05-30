from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional, List
import base64
from mistralai import Mistral
import json

app = FastAPI()

# Configuration
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

class ChatRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    api_key: str

class ImageRequest(BaseModel):
    image: str
    prompt: str = "What's in this image?"
    api_key: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {request.api_key}"
    }
    
    full_prompt = f"Context: {request.context}\n\nQuestion: {request.prompt}" if request.context else request.prompt
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(MISTRAL_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision")
async def vision_endpoint(request: ImageRequest):
    try:
        # Create Mistral client with user's API key
        client = Mistral(api_key=request.api_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": request.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{request.image}"
                    }]}]
        
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",  # Model that supports vision
            messages=messages
        )
        return {"response": chat_response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
