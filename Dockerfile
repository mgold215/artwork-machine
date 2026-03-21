FROM python:3.11-slim

# Install ffmpeg and build tools
RUN apt-get update && \
    apt-get install -y ffmpeg libcairo2 libcairo2-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["python", "app.py"]
