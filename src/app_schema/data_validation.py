from pydantic import BaseModel, Field
from typing import Literal, Annotated, Optional
import os


# --- Pydantic Models ---
class ContentRequest(BaseModel):
    user_input: Annotated[str, Field(..., description='Details of content to be created')]
    reference_data: Annotated[str, Field(..., description='Reference for content creation')]
    context: Annotated[str, Field(..., description='Context for content creation')]
    subject: Annotated[str, Field(..., description='Subject of the content')]
    topic: Annotated[str, Field(..., description='Content topic')]
    domain: Annotated[str, Field(..., description='Domain of the content (e.g., healthcare, cloud)')]
    system_prompt: Optional[str] = Field(None, description='Custom system prompt (optional)')
    user_prompt: Optional[str] = Field(None, description='Custom user prompt (optional)')
    content_size: Annotated[int, Field(..., gt=100, lt=1000000, description='Content size in words')]
    target_audience: Annotated[Literal["Doctors", "General Public", "Engineers"], Field(..., description='Target audience')]
    audience_level: Annotated[Literal["Beginner", "Intermediate", "Expert"], Field(..., description='Audience expertise level')]
    content_type: Annotated[Literal["Blog", "Training Contents", "Test cases", "Requirement", "Process"], Field(..., description='Type of content')]
    role: Annotated[Literal["Blog expert", "Tester", "Training Content Creator"], Field(..., description='Agent role')]

class ImprovedInputs(BaseModel):
    user_input: Annotated[str, Field(..., description='Details of content to be created')]
    reference_data: Annotated[str, Field(..., description='Reference for content creation')]
    context: Annotated[str, Field(..., description='Context for content creation')]
    subject: Annotated[str, Field(..., description='Subject of the content')]
    topic: Annotated[str, Field(..., description='Content topic')]
    domain: Annotated[str, Field(..., description='Domain of the content (e.g., healthcare, cloud)')]
    system_prompt: Annotated[str, Field(..., description='Custom system prompt')]
    user_prompt: Annotated[str, Field(...,  description='Custom user prompt')]

class ContentResult(BaseModel):
    content_created: str = Field(..., description="Generated content")

#testing
# testing_input_data = {"user_input": "user_input",
# "reference_data": "www.xyz.com", 
# "context": "context",
# "subject":"GenAI",
# "content_size": 1001,
# "target_audience": "Doctors",
# "audience_level": "Beginner",
# "content_type": "Blog",
# "topic": "GenAI",
# "role": "Blog expert"}

# testing_content_details = ContentRequest(**testing_input_data)
#print("testing_content_details:", testing_content_details)