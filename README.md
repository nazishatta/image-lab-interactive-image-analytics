# ImageLab — Interactive Image Analytics

Interactive Streamlit dashboard for investigating the scikit-learn handwritten-digits dataset (1,797 grayscale 8×8 images across ten labels).

## Features

- Labeled image montage with class and sample-count controls.
- Normalized pixel-intensity histograms and cross-class comparison.
- Occlusion-saliency inspection of held-out test examples using a 3×3 mask.
- Individual image browser, channel-based image upload explorer, and processing comparisons.
- Data-driven research insights and model diagnostics.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

## Deploy using Streamlit Community Cloud

Connect this repository at https://share.streamlit.io, select branch `main`, and choose `app.py` as the entry point. The deployment reads `requirements.txt` and `.streamlit/config.toml` automatically.

## Methodology

Image data come from `sklearn.datasets.load_digits`. The app uses a stratified 70%/30% training/test split with random seed 0 for logistic regression. Saliency visualizes local changes in the target-class probability after masking patches. The technique is a sensitivity diagnostic, not proof that the model reasons causally.

## Academic transparency

This coursework project uses AI-assisted application scaffolding, styling, and code review. The owner is responsible for independently reviewing the analyses, verifying functionality, and documenting their own interpretation as required by their course.
