# Experiments

## Problem Summary

The goal is to predict which prior studies are relevant to a current study.
Each case has one current study and a list of prior studies. The output is a
boolean prediction for every prior study.

## Approach Overview

The final solution uses a Logistic Regression model with engineered text
features from the study descriptions.

Features used:

- TF-IDF features for current study descriptions
- TF-IDF features for prior study descriptions
- cosine similarity between current and prior TF-IDF vectors
- token overlap count
- token overlap ratio
- exact description match
- body-region signal

The API also applies one minimal rule before the model output:

- left/right mismatch returns `False`

Everything else uses the model probability with threshold `0.51`.

## Experiments and Results

| Experiment | Accuracy |
| --- | ---: |
| Always false baseline | 0.7622 |
| Exact match | 0.7949 |
| Body-region rules | 0.9151 |
| Initial TF-IDF model | 0.8224 |
| Structured ML model | 0.9435 |
| Final model, case-level split | 0.9368 |

The deployed Railway endpoint also passed the New Lantern quick API check with
a score of 96.53 on 10 public smoke-test cases. It returned 173 predictions for
173 priors in 347ms.

The always-false baseline was strong because most priors are not relevant.

Exact description matching improved accuracy slightly, mostly by catching
repeated studies with the same name.

Body-region rules helped a lot by grouping related studies such as breast,
chest, cardiac, abdomen/pelvis, and spine. This raised recall, but the rules
were hard to tune safely.

The first ML model used TF-IDF over combined current/prior text. It was better
than simple matching, but not better than the body-region rules.

The structured ML model worked best. Separating current and prior TF-IDF
features and adding similarity, overlap, exact-match, and body-region features
gave the model much better signal.

The final model keeps the structured ML model, validates by `case_id`, and adds
only a small left/right mismatch rule. This reduced clear false positives
without adding broad filters.

## What Worked

- Combining ML with lightweight domain features
- Feature engineering around similarity and token overlap
- Keeping rules small and conservative
- Batch prediction instead of per-study API calls

## What Didn't Work

- Heavy rule-based logic reduced recall
- Strict filters were easy to overfit to the public data
- Broad modality and region rules created too many false negatives

## Design Decisions

No LLM is used.

Reasons:

- latency constraints
- batch processing requirement
- cost concerns
- reliability and repeatability concerns

This task is mostly structured text matching and ranking, so a small local model
is a better fit for the endpoint.

The deployed API loads the model once at startup and performs batch prediction
for all prior studies in a request. On the full public eval request, it returned
27,614 predictions in under 2 seconds during local-to-deployed testing.

## Error Analysis and Workflow Considerations

False negatives are more concerning in radiology workflow because a missing
relevant prior may hide disease progression or interval change. False positives
are less dangerous but add reading burden. Because of that, I prioritized full
prediction coverage, low latency, and a balanced precision/recall tradeoff
instead of using aggressive filters that reduced recall.

Remaining errors mostly came from body-region overlaps, laterality edge cases,
and cases where different modalities were clinically related. In a production
system, I would surface priors with relevance scores rather than only a binary
label, allowing radiologists to quickly scan the most likely relevant exams
first.

## Future Improvements

- Clinical ontologies such as RadLex or UMLS
- Better modality-aware features
- Learned embeddings for medical text
- Temporal modeling of study dates
