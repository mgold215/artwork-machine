FROM python:3.11-slim

# Install ffmpeg and build tools
RUN apt-get update && \
    apt-get install -y ffmpeg libcairo2 libcairo2-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 7860
# gunicorn: production WSGI server with 10min timeout for large uploads + long pipeline runs
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "--worker-class", "sync"]
