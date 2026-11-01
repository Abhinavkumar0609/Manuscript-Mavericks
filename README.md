---
title: Manuscript Mavericks
emoji: 📜
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Manuscript to Text (web app)

Upload a scanned Sanskrit/Devanagari page; the app enhances it and returns searchable text
using the fine-tuned model from the project (`manuscript_san`).

## Run
1. Install the Tesseract engine (version 5): `sudo apt install tesseract-ocr` (Linux) or
   `brew install tesseract` (macOS); on Windows use the UB Mannheim installer and add it to PATH.
2. `pip install -r requirements.txt`
3. `python app.py` and open http://127.0.0.1:5000

## Using it
Upload a page image (drag and drop or click). The app enhances it and reads it with both models straight away, then shows the recognised text, the output of the pretrained Tesseract model next to the fine-tuned model, and the CER and WER comparison. For CER and WER on your own page, open "Correct text (optional)", add the page's correct text and click "Recognise again".

## Models (`models/tessdata/`)
- `manuscript_san.traineddata` - fine-tuned model
- `san.traineddata` - pretrained tessdata_best baseline

## Files
`pipeline.py` preprocessing + OCR (same steps as notebook Section 3) · `app.py` Flask server · `templates/index.html` interface.
Reported accuracy (30 held-out pages): CER 0.31 -> 0.16, WER 0.85 -> 0.58 after fine-tuning. Output can still contain errors; check it against the page.

## Deploy
A `Dockerfile` is included. On Hugging Face Spaces: create a Space with SDK "Docker", upload the contents of this folder, and wait for the build. On Render, Railway or Fly.io: create a web service from this folder with the Dockerfile (they provide the PORT variable).
