import base64, os, time
import cv2
import numpy as np
import pytesseract
import shutil

if not shutil.which("tesseract"):
    for _p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
               r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
               os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

BASE = os.path.dirname(os.path.abspath(__file__))
TESSDATA = os.path.join(BASE, "models", "tessdata")
os.environ["TESSDATA_PREFIX"] = TESSDATA

# display name -> Tesseract language file in models/tessdata
MODELS = {"finetuned": "manuscript_san", "baseline": "san"}
TARGET_WIDTH = 1200


def available_models():
    return [k for k, lang in MODELS.items()
            if os.path.exists(os.path.join(TESSDATA, lang + ".traineddata"))]


def load_normalized(img, target_w=TARGET_WIDTH):
    if img.shape[1] > 1.5 * target_w:  # very wide scans are scaled to 1200 px
        s = target_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def deskew(img):
    coords = np.column_stack(np.where(img < 255))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 0:        # OpenCV >= 4.5 reports (0, 90]; notebook logic expects [-90, 0)
        angle -= 90
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_page(img):
    img = load_normalized(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, blockSize=31, C=15)
    return deskew(binar)


def recognise(image_bytes, model="finetuned"):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("This file could not be read as an image. Use a JPG, PNG, TIFF or WebP scan.")
    if model not in available_models():
        raise ValueError(f"Model '{model}' is not installed in models/tessdata.")
    t0 = time.time()
    enhanced = preprocess_page(img)
    text = pytesseract.image_to_string(enhanced, lang=MODELS[model], config="--oem 1 --psm 6")
    ok, png = cv2.imencode(".png", enhanced)
    return {
        "text": text.strip(),
        "enhanced": "data:image/png;base64," + base64.b64encode(png).decode(),
        "model": model,
        "seconds": round(time.time() - t0, 1),
    }


def _edit(a, b):
    """Levenshtein distance between two sequences (rapidfuzz if installed, else pure Python)."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compare(image_bytes, reference=""):
    """Read one page with both models; score each against the correct text if given."""
    if not {"baseline", "finetuned"} <= set(available_models()):
        raise ValueError("Both models (san and manuscript_san) must be in models/tessdata.")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("This file could not be read as an image. Use a JPG, PNG, TIFF or WebP scan.")
    ref = " ".join((reference or "").split())
    t0 = time.time()
    enhanced = preprocess_page(img)
    out = {}
    for m in ("baseline", "finetuned"):
        text = pytesseract.image_to_string(enhanced, lang=MODELS[m], config="--oem 1 --psm 6")
        hyp = " ".join(text.split())
        out[m] = {"text": text.strip()}
        if ref:
            out[m]["cer"] = round(_edit(ref, hyp) / len(ref), 4)
            out[m]["wer"] = round(_edit(ref.split(), hyp.split()) / len(ref.split()), 4)
    ok, png = cv2.imencode(".png", enhanced)
    return {"models": out, "has_reference": bool(ref), "ref_chars": len(ref),
            "ref_words": len(ref.split()), "seconds": round(time.time() - t0, 1),
            "enhanced": "data:image/png;base64," + base64.b64encode(png).decode()}
