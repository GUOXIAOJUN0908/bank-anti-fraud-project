FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user (bank container hardening baseline).
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs && chown appuser /app/logs

COPY src ./src
COPY models ./models

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# One worker: card history lives in process memory. Scale out only after moving it to Redis.
CMD ["uvicorn", "fraud.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
