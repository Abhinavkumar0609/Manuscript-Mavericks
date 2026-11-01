import json, os
from flask import Flask, jsonify, render_template, request
import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB per upload


@app.get("/")
def index():
    return render_template("index.html", models=pipeline.available_models())


@app.post("/api/ocr")
def ocr():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="Choose an image of a manuscript page first."), 400
    try:
        return jsonify(pipeline.recognise(f.read(), request.form.get("model", "finetuned")))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except pipeline.pytesseract.TesseractNotFoundError:
        return jsonify(error="Tesseract was not found. Install Tesseract 5 and restart the app."), 500
    except Exception as e:
        app.logger.exception("OCR failed")
        return jsonify(error=f"Recognition failed: {e}"), 500


@app.post("/api/compare")
def compare():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="Choose an image of a manuscript page first."), 400
    try:
        return jsonify(pipeline.compare(f.read(), request.form.get("reference", "")))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        app.logger.exception("Compare failed")
        return jsonify(error=f"Comparison failed: {e}"), 500


@app.get("/api/results")
def results():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.json"), encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That image is over 20 MB. Resize or compress it and try again."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
