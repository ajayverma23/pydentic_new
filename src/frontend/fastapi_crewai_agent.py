'''
how to run: uvicorn fastapi_crewai_agent:app --reload #make sure code from same folder
where is code file there, here folder "frontend"
'''

from fastapi import FastAPI, HTTPException

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app_schema.data_validation import ContentRequest, ContentResult
from agents.crewai_multiple_agents import MultiLevelCrewAI


app = FastAPI(title="Content Creation API")
    
@app.post("/generate-content", response_model=ContentResult)
async def generate_content(request: ContentRequest):
    try:
        crew = MultiLevelCrewAI()
        result = crew.run(request)
        # result is likely a CrewOutput object, not a string
        # Extract the string content
        if hasattr(result, "raw"):
            final_content = result.raw
        else:
            final_content = str(result)
        return ContentResult(content_created=final_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



