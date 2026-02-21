# Cats vs Dogs inference API
FROM python:3.10-slim

WORKDIR /app

# Install dependencies (pinned for reproducibility)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/

# Model is mounted at runtime: -v $(pwd)/models:/app/models
ENV MODEL_PATH=/app/models/baseline.pt
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
