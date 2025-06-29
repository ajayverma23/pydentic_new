# === BUILDER STAGE ===
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# === RUNTIME STAGE ===
FROM python:3.11-slim

WORKDIR /app

# Copy application code from builder stage
COPY --from=builder /app .

# Expose the port your app runs on
EXPOSE 8501

# Set the default command to run your app
CMD ["streamlit", "run", "frontend/fastapi_streamlit_crewai_agent.py"]