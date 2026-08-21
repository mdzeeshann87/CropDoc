# CropDoc — AI-Based Crop Disease Prediction Website

A Flask web app that diagnoses crop leaf diseases from a photo: upload a leaf
image, pick the crop species, and get a ranked diagnosis with symptoms,
likely cause, treatment, and prevention steps.

## Live Demo

Try it here: **[https://cropdoc-yj8t.onrender.com](https://cropdoc-yj8t.onrender.com)**

> Note: hosted on Render's free tier — the app may take ~50 seconds to wake up if it's been idle.

## Run it locally 

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**.

## How the diagnosis works

There's no bundled pretrained CNN (large model downloads aren't available in
every environment), so out of the box the app uses a **deterministic,
explainable rule-based analyzer** (`inference.py`):

1. The uploaded photo is converted to HSV color space.
2. It computes four interpretable signals: chlorophyll (green) coverage,
   browning/necrosis, yellowing (chlorosis), and lesion "spottiness" (local
   texture variance).
3. Those signals are compared against each disease's known visual signature
   in `disease_data.py`, and scored into a ranked, confidence-weighted list.

This gives a fully working, offline diagnostic pipeline with real, useful
agronomy content — but its accuracy is naturally limited compared to a
trained deep-learning model.

### Upgrading to a real trained CNN

`inference.py` will automatically use a trained model if you place one at
`models/crop_disease_model.h5` — no other code changes required.

To train one:

```bash
pip install tensorflow
python train_model.py --data_dir /path/to/PlantVillage --crop tomato --epochs 15
```

`train_model.py` builds a MobileNetV2-based transfer-learning classifier.
Good public datasets to train on: **PlantVillage** (~54k labeled leaf
images across 14 crop species) or **PlantDoc**.

## Project structure

```
app.py              Flask routes: page + /api/analyze endpoint
inference.py         Feature extraction + rule-based / CNN diagnosis engine
disease_data.py       Crop & disease knowledge base (symptoms, treatment, etc.)
train_model.py        Optional script to train a real CNN on your own dataset
templates/index.html  UI markup
static/css/style.css  Styling
static/js/script.js   Upload, scan animation, results rendering
static/uploads/       Uploaded leaf photos are saved here
models/                Drop a trained crop_disease_model.h5 here to activate it
```

## Supported crops (demo knowledge base)

Tomato, Potato, Corn/Maize, Apple, Grape, Wheat — each with a healthy class
plus 2–3 common diseases. Add more by extending `DISEASE_DB` in
`disease_data.py`.

## Notes

- Max upload size: 8MB. Accepted formats: JPG, PNG, WEBP.
- This is a demo/educational tool, not a substitute for a licensed plant
  pathologist or agricultural extension service.
