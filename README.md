# ImageLab · Interactive Image Analytics

A Streamlit dashboard exploring the scikit-learn handwritten digits dataset: image montages, grayscale intensity histograms, class comparisons, held-out logistic regression occlusion saliency, and image browser tools.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Methods

- Images: 1,797 grayscale 8 × 8 images, values 0–16.
- Classifier: logistic regression; stratified 70/30 train/test split (seed 0).
- Saliency: occlude overlapping 3×3 regions, then compute Δp = P(true class | original) − P(true class | occluded).
- Positive Δp indicates masking lowers the true-class probability; negative Δp indicates masking raises it.
- Limitations: occlusion maps are local sensitivity views, not proof of causal understanding; blanking already-zero pixels cannot reveal background reliance.

## Acknowledgment

Development involved AI assistance for app scaffolding, styling, debugging, and code review. Interpret scientific findings against observed figures and model outputs before submission.
