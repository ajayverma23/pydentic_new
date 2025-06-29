
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import Any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app_schema.data_validation import ContentRequest, ContentResult
from agents.crewai_multiple_agents import MultiLevelCrewAI

from agents.app_pdf_utils import serialize_result

google_sheet_URL = "https://docs.google.com/spreadsheets/d/1yiN8eO4-1alxsQgrgkinbX_CckOZfjgb/export?format=csv&gid=1525010287"

import pandas as pd

sheet_id = "1yiN8eO4-1alxsQgrgkinbX_CckOZfjgb"
gid = "1525010287"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

df = pd.read_csv(csv_url)
print("DataFrame loaded from Google Sheet:")
#print(df)

# Convert DataFrame to JSON (list of dicts)
json_data = df.to_dict(orient='records')
#print("JSON data:")
#print(json_data)

# --- Batch Process Each Record ---
results = []

for idx, record in enumerate(json_data):
    print(f"\nProcessing record {idx+1}...")
    try:
        # Validate & coerce types
        validated = ContentRequest(**record)
        #print(f"Validated input: {validated}")

        # Generate content using your CrewAI logic
        crew = MultiLevelCrewAI()
        result = crew.run(validated)
        
        #print(f"Result for record {idx+1}:\n{result}\n{'-'*40}")
        results.append({"input": validated.model_dump(), "result": serialize_result(result)})
        #results.append({"input": validated.dict(), "result": result})

        # Optionally, save or export result here (e.g., to CSV, JSON, or PDF)

    except ValidationError as ve:
        print(f"Validation error for record {idx+1}: {ve}")
        results.append({"input": record, "error": str(ve)})
    except Exception as e:
        print(f"Error processing record {idx+1}: {e}")
        results.append({"input": record, "error": str(e)})

# --- Optionally, save all results to a file ---
import json
with open("batch_content_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Batch processing complete. Results saved to batch_content_results.json.")