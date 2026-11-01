FROM python:3.12-slim

# Tesseract 5 engine (the models in models/tessdata are loaded by pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If the files were uploaded flat (no folders)
RUN mkdir -p templates data models/tessdata \
    && if [ -f index.html ]; then mv index.html templates/; fi \
    && if [ -f results.json ]; then mv results.json data/; fi \
    && if [ -f manuscript_san.traineddata ]; then mv manuscript_san.traineddata models/tessdata/; fi \
    && if [ -f san.traineddata ]; then mv san.traineddata models/tessdata/; fi

# Hugging Face Spaces uses 7860; other hosts set PORT themselves
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 2 --timeout 180 app:app"]
