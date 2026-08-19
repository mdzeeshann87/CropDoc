"""
CropDoc analysis engine.

This module extracts colour/texture features from a leaf photo (green
coverage, browning, chlorosis/yellowing, and lesion "spottiness") and scores
them against the signature of each known condition in disease_data.py. It is
a deterministic, explainable stand-in for a trained CNN so the app works
fully offline with zero external model downloads.

To upgrade to a real deep-learning classifier: train the architecture in
train_model.py on a labeled leaf-image dataset (e.g. PlantVillage), export
it to models/crop_disease_model.h5, and this module will automatically defer
to it — see `_try_load_deep_model()` below.
"""

import io
import os
import colorsys
import numpy as np
from PIL import Image

from disease_data import DISEASE_DB

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_disease_model.h5")

LEVELS = {"low": 0, "medium": 1, "high": 2}


def _try_load_deep_model():
    """Load a trained Keras model if one has been placed in models/. Returns None otherwise."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from tensorflow import keras  # noqa: local import, optional dependency
        return keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


_DEEP_MODEL = _try_load_deep_model()


def _bucket(value, low_hi, hi_hi):
    if value < low_hi:
        return "low"
    if value < hi_hi:
        return "medium"
    return "high"


def extract_features(image: Image.Image):
    """Return a dict of interpretable 0-1 feature scores for a leaf photo."""
    img = image.convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    v = maxc
    s = np.where(maxc == 0, 0, (maxc - minc) / np.where(maxc == 0, 1, maxc))

    # Hue in degrees (vectorized HSV hue calc)
    delta = maxc - minc + 1e-8
    hue = np.zeros_like(maxc)
    mask_r = (maxc == r)
    mask_g = (maxc == g) & ~mask_r
    mask_b = (maxc == b) & ~mask_r & ~mask_g
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    leaf_mask = (s > 0.12) & (v > 0.08)  # exclude near-white/near-black background
    leaf_pixel_count = max(leaf_mask.sum(), 1)

    green_mask = leaf_mask & (hue >= 70) & (hue <= 170) & (v > 0.25)
    yellow_mask = leaf_mask & (hue >= 35) & (hue < 70)
    brown_mask = leaf_mask & (hue >= 8) & (hue < 40) & (v < 0.55)
    dark_brown_mask = leaf_mask & (v < 0.25) & (s > 0.15)

    green_ratio = float(green_mask.sum() / leaf_pixel_count)
    yellow_ratio = float(yellow_mask.sum() / leaf_pixel_count)
    brown_ratio = float((brown_mask | dark_brown_mask).sum() / leaf_pixel_count)

    # Texture / spottiness proxy: local variance of luminance over 8x8 blocks.
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    h, w = lum.shape
    bs = 8
    block_view = lum[: h - h % bs, : w - w % bs].reshape(h // bs, bs, w // bs, bs)
    block_std = block_view.std(axis=(1, 3))
    spotty_blocks = (block_std > 0.06).mean()
    spot_ratio = float(np.clip(spotty_blocks * 1.3, 0, 1))

    return {
        "green": green_ratio,
        "yellow": yellow_ratio,
        "brown": brown_ratio,
        "spots": spot_ratio,
        "leaf_coverage": float(leaf_pixel_count / (256 * 256)),
    }


def _feature_levels(features):
    return {
        "green": _bucket(features["green"], 0.35, 0.65),
        "yellow": _bucket(features["yellow"], 0.08, 0.22),
        "brown": _bucket(features["brown"], 0.06, 0.18),
        "spots": _bucket(features["spots"], 0.15, 0.35),
    }


def _score_condition(levels, signature):
    """Lower distance = better match. Compares categorical levels per channel."""
    dist = 0
    for key, target_level in signature.items():
        dist += abs(LEVELS[levels[key]] - LEVELS[target_level])
    return dist


def diagnose(image: Image.Image, crop_key: str):
    """
    Returns a ranked list of {condition, confidence, features} for the given
    crop, using the trained deep model if present, otherwise the rule-based
    color/texture analyzer.
    """
    crop = DISEASE_DB[crop_key]
    features = extract_features(image)

    if _DEEP_MODEL is not None:
        # Real CNN path (active once a trained .h5 model is placed in models/)
        img = image.convert("RGB").resize((224, 224))
        x = np.expand_dims(np.asarray(img).astype(np.float32) / 255.0, axis=0)
        preds = _DEEP_MODEL.predict(x, verbose=0)[0]
        conditions = crop["conditions"]
        ranked = sorted(zip(conditions, preds), key=lambda t: -t[1])
        return [
            {"condition": c, "confidence": float(p), "features": features}
            for c, p in ranked
        ], "deep_model"

    levels = _feature_levels(features)
    conditions = crop["conditions"]
    scored = []
    max_dist = 8  # 4 channels * max per-channel distance of 2
    for cond in conditions:
        dist = _score_condition(levels, cond["signature"])
        raw_score = max(max_dist - dist, 0)
        scored.append((cond, raw_score))

    total = sum(s for _, s in scored) or 1
    # Softmax-ish sharpening so the top match reads as a confident diagnosis
    exp_scores = [(c, np.exp(s * 1.6)) for c, s in scored]
    exp_total = sum(s for _, s in exp_scores) or 1
    ranked = sorted(
        [{"condition": c, "confidence": float(s / exp_total), "features": features} for c, s in exp_scores],
        key=lambda d: -d["confidence"],
    )
    return ranked, "rule_based"


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
