# New Lantern Priors

This project predicts whether a prior imaging study is relevant to a current study.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The prediction endpoint is:

```text
POST /predict
```

## Test locally

With the API running:

```bash
python3 scripts/test_api_local.py
python3 scripts/test_api_public_eval.py
```

The public eval test sends all public cases in one request and checks that every prior study gets one prediction.

## Model

The model is a logistic regression classifier using:

- separate TF-IDF features for current and prior study descriptions
- cosine similarity
- token overlap count and ratio
- exact-description and same-body-region features

The API uses threshold `0.51`. It applies one rule before the model output: left/right side mismatch returns `False`.

## Current Validation Result

Validation split result:

```text
Accuracy: 0.9468
Precision: 0.9271
Recall: 0.8423
False positives: 87
False negatives: 207
```
