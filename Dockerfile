# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code and committed reference data. The trained model (models/), the
# risk overlay (tiles/), and processed datasets (data/processed/) are large build
# artifacts kept out of git — mount them at runtime, e.g.:
#   docker run -p 8080:8080 \
#     -v "$PWD/models:/app/models" -v "$PWD/tiles:/app/tiles" \
#     -v "$PWD/data/processed:/app/data/processed" traffic-safety
# Without them the service still boots and serves discovery/validation/reference
# endpoints; /v1/health reports model_ready=false until the model is present.
COPY src/ ./src/
COPY data/reference/ ./data/reference/
COPY source_catalog.json ./

ENV PYTHONPATH=/app/src
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/health')"

CMD ["python", "-m", "uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8080"]
