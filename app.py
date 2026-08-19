import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image, UnidentifiedImageError

from disease_data import DISEASE_DB, CROP_CHOICES
from inference import diagnose, load_image_from_bytes

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html", crops=CROP_CHOICES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file was sent."}), 400

    file = request.files["image"]
    crop_key = request.form.get("crop", "tomato")

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, or WEBP image."}), 400
    if crop_key not in DISEASE_DB:
        return jsonify({"error": "Unknown crop type."}), 400

    raw = file.read()
    try:
        image = load_image_from_bytes(raw)
        image.verify()
        image = load_image_from_bytes(raw)  # reopen after verify() invalidates the handle
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "That file doesn't look like a valid image."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    image.convert("RGB").save(saved_path, quality=88)

    ranked, engine = diagnose(image, crop_key)
    top = ranked[0]
    cond = top["condition"]

    result = {
        "engine": engine,
        "crop": DISEASE_DB[crop_key]["label"],
        "image_url": url_for("static", filename=f"uploads/{saved_name}"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "top_result": {
            "id": cond["id"],
            "name": cond["name"],
            "severity": cond["severity"],
            "confidence": round(top["confidence"] * 100, 1),
            "summary": cond["summary"],
            "symptoms": cond["symptoms"],
            "cause": cond["cause"],
            "treatment": cond["treatment"],
            "prevention": cond["prevention"],
        },
        "alternatives": [
            {
                "name": r["condition"]["name"],
                "confidence": round(r["confidence"] * 100, 1),
                "severity": r["condition"]["severity"],
            }
            for r in ranked[1:4]
        ],
        "features": {k: round(v, 3) for k, v in top["features"].items()},
    }
    return jsonify(result)


@app.route("/api/crops")
def crops():
    return jsonify(CROP_CHOICES)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
