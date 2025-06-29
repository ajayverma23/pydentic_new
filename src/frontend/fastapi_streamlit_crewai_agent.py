'''
how to run: fastapi and streamlit together
first run streamlit code : (pydentic-new) PS D:\Aj\GenAI\pydentic_new\frontend> streamlit run D:\Aj\GenAI\pydentic_new\frontend\streamlit_crewai_agent.py
second run fastapi code : (pydentic-new) PS D:\Aj\GenAI\pydentic_new\frontend> uvicorn fastapi_streamlit_crewai_agent:app --reload # make sure code from same folder
where is code file there, here folder "frontend"
'''

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="FastAPI Proxy to Streamlit")

# Change this to the actual URL/port where your Streamlit backend is running
STREAMLIT_API_URL = "http://localhost:8501/generate-content"

@app.post("/generate-content")
async def proxy_generate_content(request: Request):
    try:
        # Read the incoming JSON
        body = await request.json()
        # Forward the request to the Streamlit backend
        async with httpx.AsyncClient() as client:
            resp = await client.post(STREAMLIT_API_URL, json=body)
        # Return the response from Streamlit as is
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

