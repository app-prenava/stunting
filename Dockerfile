FROM python:3.11-slim

# System deps for numpy/pandas/sklearn (avoids build-time failures)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY main.py .
COPY models/ models/

EXPOSE 8000

# Gunicorn + UvicornWorker: multi-process, production-grade
# --workers 2: safe for 1-2 vCPU VPS. Increase to 4 on 2+ vCPU.
# --timeout 120: allows SHAP + inference to complete within 2 min
# --preload: loads model ONCE in master process, forks to workers (memory efficient)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--preload", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "main:app"]
