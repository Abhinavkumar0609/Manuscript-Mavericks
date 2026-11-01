FROM python:3.12-slim

# Tesseract 5 engine (the models in models/tessdata are loaded by pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces uses 7860; other hosts set PORT themselves
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 2 --timeout 180 app:app"]
