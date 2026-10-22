"""ImageLab — interactive image dataset exploration and occlusion diagnostics."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ImageLab Dropdown Fixed v7 | DATS 6401", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

BG = "#101318"
PANEL = "#191e26"
FG = "#ebedf1"
MUTED = "#9ea9b8"
ACCENT = "#80c9bb"
ACCENT2 = "#dbb77c"

st.markdown('''<style>
:root {color-scheme:dark;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label {color:#edf1f5 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#bac5d1 !important;}

.stApp, [data-testid="stAppViewContainer"] {background:#101318;color:#ebedf1;}
[data-testid="stSidebar"] {background:#151a21;border-right:1px solid #323946;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {color:#e6ebf1 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#b9c5d1 !important;}
/* Contrast-safe light select fields on a dark sidebar. BaseWeb uses an INPUT
   for the selected value; styling SPAN/DIV alone leaves the digit invisible. */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select"] {
  background-color:#f4f6f9 !important; border:1px solid #8395a9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
  color:#17212d !important; -webkit-text-fill-color:#17212d !important;
  opacity:1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
  color:#17212d !important; fill:#17212d !important;
}
[data-baseweb="popover"] [role="listbox"] {background:#f4f6f9 !important;}
[data-baseweb="popover"] [role="option"] {color:#17212d !important; -webkit-text-fill-color:#17212d !important;}
[data-baseweb="popover"] [role="option"]:hover {background:#dce8ed !important;}

[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label {color:#e2e7ee;}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {color:#b4bfce;}
[data-testid="stAppViewContainer"] [data-baseweb="textarea"] textarea {color:#171d25;}
[data-testid="stHeader"] {background:rgba(16,19,24,.85);}
.block-container {max-width:1370px;padding-top:2rem;padding-bottom:4rem;}
[data-testid="stTabs"] button {font-weight:650;}
button[data-baseweb="tab"] {color:#a9b4c1;}
button[data-baseweb="tab"][aria-selected="true"] {color:#80c9bb;}
[data-testid="stMetric"] {background:#191e26;border:1px solid #313844;padding:14px 16px;border-radius:12px;}
[data-testid="stMetricLabel"] {color:#9ea9b8;}
[data-testid="stMetricValue"] {color:#e9f1f0;}
div[data-testid="stAlert"] {border-radius:10px;}
.stMarkdown a {color:#80c9bb;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {background:linear-gradient(110deg,#202a32,#171d25 65%,#18202a);border:1px solid #39414c;padding:30px 34px;border-radius:18px;margin:0 0 16px;}
.eyebrow {font-size:11px;color:#80c9bb;font-weight:800;letter-spacing:.15em;}
.hero-title {font-size:37px;font-weight:760;letter-spacing:-.045em;margin:8px 0;line-height:1.12;color:#f6f8fa;}
.hero-desc {color:#b9c3ce;font-size:15px;max-width:850px;line-height:1.65;}
.section-caption {color:#9ea9b8;font-size:14px;margin-bottom:14px;}
.small-chip {display:inline-block;border:1px solid #3b5155;background:#1c3133;color:#aee6dc;border-radius:999px;padding:5px 12px;margin:8px 8px 0 0;font-size:12px;}
[data-testid="stPlotlyChart"], [data-testid="stImage"] {border-radius:10px;}
</style>''', unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": "#54606d", "font.size": 10,
    "grid.color": "#333d48", "axes.titlecolor": FG,
})

@st.cache_data
def dataset():
    return load_digits()

@st.cache_resource
def fitted_model():
    ds = load_digits()
    all_ids = np.arange(len(ds.target))
    train_ids, test_ids = train_test_split(all_ids, stratify=ds.target, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=5000, random_state=0).fit(ds.data[train_ids], ds.target[train_ids])
    accuracy = model.score(ds.data[test_ids], ds.target[test_ids])
    return model, test_ids, accuracy

@st.cache_data(show_spinner=False)
def saliency_for(digit_id: int, cls: int) -> tuple[np.ndarray, float]:
    ds = load_digits()
    model, _, _ = fitted_model()
    image = ds.images[digit_id]
    base = float(model.predict_proba(image.reshape(1, -1))[0, cls])
    sal = np.zeros_like(image, dtype=float)
    for y in range(8):
        for x in range(8):
            altered = image.copy()
            altered[max(y - 1, 0):min(y + 2, 8), max(x - 1, 0):min(x + 2, 8)] = 0
            sal[y, x] = base - model.predict_proba(altered.reshape(1, -1))[0, cls]
    return sal, base


def render(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def section(title: str, desc: str):
    st.subheader(title)
    st.markdown(f'<div class="section-caption">{desc}</div>', unsafe_allow_html=True)


digits = dataset()
with st.sidebar:
    st.markdown("### ◈ ImageLab")
    st.caption("INTERACTIVE EXPLORATION")
    selected = st.selectbox("1 · Digit class", list(range(10)), index=3, key="digit_class")
    montage_count = st.select_slider("2 · Montage images", options=[12, 18, 24, 30], value=30)
    comparison = st.selectbox("3 · Compare intensity with", ["None"] + [str(k) for k in range(10) if k != selected], index=1)
    saliency_choice = st.slider("4 · Test-image position", min_value=1, max_value=int((load_digits().target[train_test_split(np.arange(len(load_digits().target)), stratify=load_digits().target, test_size=0.3, random_state=0)[1]] == selected).sum()), value=1)
    st.caption("The controls update the montage, intensity comparison, and held-out saliency example.")


st.markdown('''<div class="hero"><div class="eyebrow">GEORGE WASHINGTON UNIVERSITY · DATS 6401</div>
<div class="hero-title">IMAGE<span style="color:#80c9bb">LAB</span> / Visual Image Analysis</div>
<div class="hero-desc">An interactive study of handwritten digit images, pixel intensity, and model behavior. Examine data diversity and the evidence behind a classifier's predictions.</div>
<div><span class="small-chip">01 · Collections</span><span class="small-chip">02 · Structure</span><span class="small-chip">03 · Explainability</span></div></div>''', unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
c1.metric("Dataset images", f"{len(digits.target):,}")
c2.metric("Classes", "10")
c3.metric("Image dimensions", "8 × 8")
c4.metric(f"Class {selected} samples", f"{int((digits.target==selected).sum()):,}")
st.write("")

t_montage, t_structure, t_sal, t_browser, t_findings = st.tabs(["01  Montage", "02  Pixel structure", "03  Saliency", "04  Image browser", "05  Research insights"])

with t_montage:
    section("Collection explorer", "Compare handwriting styles within one labeled class. Look for variation, faint strokes and images that differ from most of their class.")
    count=montage_count
    ids=np.flatnonzero(digits.target==selected)
    fig, axes=plt.subplots(count//6,6,figsize=(11.3,(count//6)*1.53), facecolor=PANEL)
    for ax, idx in zip(np.asarray(axes).ravel(),ids[:count]):
        ax.imshow(digits.images[idx],cmap="gray",vmin=0,vmax=16,interpolation="nearest")
        ax.set_title(f"ID {idx}",fontsize=9,color=MUTED,pad=3)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.025,right=.99,top=.96,bottom=.04,wspace=.22,hspace=.42)
    render(fig)
    st.caption("Figure 1. First selected class examples in dataset order; each tile uses the same grayscale scale (0–16). IDs refer to original dataset positions.")
    with st.expander("Optional exploration · class balance and mean image",expanded=False):
        left,right=st.columns([1.6,1])
        with left:
            counts=np.bincount(digits.target,minlength=10)
            fig,ax=plt.subplots(figsize=(7.5,3))
            ax.bar(range(10), counts, color=[ACCENT if d==selected else '#667b90' for d in range(10)])
            ax.set_xticks(range(10));ax.set_xlabel("Class label");ax.set_ylabel("Number of images")
            ax.set_title("Images per digit class",fontsize=12);ax.grid(axis="y",alpha=.28)
            render(fig)
        with right:
            mean_image=digits.images[ids].mean(axis=0)
            fig,ax=plt.subplots(figsize=(3,3))
            ax.imshow(mean_image,cmap="gray",vmin=0,vmax=16)
            ax.set_title(f"Mean image · digit {selected}");ax.axis("off")
            render(fig)
        st.caption("A mean image blends all examples. It can show the typical stroke location but hide within-class variation.")

with t_structure:
    section("Pixel intensity and structure", "Grayscale images have one intensity channel. A histogram shows the full distribution that a single average cannot capture.")
    compare=comparison != "None"
    other=int(comparison) if compare else None
    pixels=digits.images[digits.target==selected].ravel()
    fig,ax=plt.subplots(figsize=(10,4.3))
    bins=np.arange(-.5,17.5,1)
    ax.hist(pixels,bins=bins,density=True,alpha=.78,label=f"Digit {selected}",color=ACCENT)
    if compare:
        alt=digits.images[digits.target==other].ravel()
        ax.hist(alt,bins=bins,density=True,alpha=.5,label=f"Digit {other}",color=ACCENT2)
    ax.set_xlabel("Pixel intensity (0 = black; 16 = brightest)")
    ax.set_ylabel("Proportion of pixels per intensity value")
    ax.set_xticks(range(17,2));ax.grid(axis="y",alpha=.28)
    ax.legend(facecolor=PANEL,edgecolor="#424e5b",labelcolor=FG)
    ax.set_title("Grayscale intensity distribution",fontsize=15,pad=17)
    fig.tight_layout();render(fig)
    a,b,c=st.columns(3)
    a.metric("Mean pixel intensity",f"{pixels.mean():.2f}")
    b.metric("Black pixels (intensity 0)",f"{100*np.mean(pixels==0):.1f}%")
    c.metric("Bright pixels (12–16)",f"{100*np.mean(pixels>=12):.1f}%")
    st.caption("Figure 2. Histograms are density-normalized so class sample counts do not drive the height of the bars.")

with t_sal:
    section("Occlusion saliency · held-out data", "A logistic regression model classifies digits. Hide a 3 × 3 patch, re-predict, and measure the probability change for the true class.")
    model,test_ids,accuracy=fitted_model()
    eligible=np.sort(test_ids[digits.target[test_ids]==selected])
    image_idx=int(eligible[saliency_choice - 1])
    st.caption(f"Selected held-out test image: ID {image_idx} · example {saliency_choice} of {len(eligible)}")
    sal,base=saliency_for(int(image_idx),int(selected))
    image=digits.images[image_idx]
    prediction=int(model.predict(image.reshape(1,-1))[0])
    m1,m2,m3=st.columns(3)
    m1.metric("Model test accuracy",f"{accuracy:.1%}")
    m2.metric("Predicted class",str(prediction))
    m3.metric("P(true class)",f"{base:.3f}")
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    axes[0].imshow(image,cmap="gray",vmin=0,vmax=16,interpolation="nearest")
    axes[0].set_title("01 · Original image",fontsize=12)
    limit=max(float(np.max(np.abs(sal))),1e-6)
    hm=axes[1].imshow(sal,cmap="coolwarm",vmin=-limit,vmax=limit,interpolation="nearest")
    axes[1].set_title("02 · Signed probability change",fontsize=12)
    cb=fig.colorbar(hm,ax=axes[1],fraction=.045,pad=.03);cb.ax.tick_params(labelsize=8,colors=MUTED)
    axes[2].imshow(image,cmap="gray",vmin=0,vmax=16,interpolation="nearest")
    axes[2].imshow(np.maximum(sal,0),cmap="inferno",vmin=0,vmax=max(float(sal.max()),1e-6),alpha=.65,interpolation="nearest")
    axes[2].set_title("03 · Positive-drop overlay",fontsize=12)
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout(w_pad=2);render(fig)
    st.caption("Figure 3. Red regions indicate that masking increases evidence for the true class (a probability drop); blue regions indicate probability increases when masked. The right overlay highlights positive drops with inferno.")
    if prediction!=selected:
        st.warning("This is a misclassified test image. The heatmap still shows sensitivity of the TRUE class probability, not necessarily the predicted class.")
    with st.expander("How this experiment is controlled"):
        st.write("Dataset split: 70% training / 30% held-out testing, stratified by class, random seed 0. Each occlusion blanks a 3 × 3 region (clipped at image edges). Model and split are cached. The same test image is used across all patches.")

with t_browser:
    section("Image browser · supporting exploration", "Explore individual digit statistics or upload your own image for channel and processing comparisons. Optional tools; the graded analysis uses the digits dataset above.")
    b1,b2=st.tabs(["Single digit", "Upload and compare"])
    with b1:
        image_ids=np.flatnonzero(digits.target==selected)
        one=st.selectbox("Image ID",image_ids.tolist(),format_func=lambda x:f"ID {x}",key="browse_id")
        img=digits.images[one]
        col1,col2=st.columns([1,2])
        with col1:
            fig,ax=plt.subplots(figsize=(4,4));ax.imshow(img,cmap="gray",vmin=0,vmax=16,interpolation="nearest");ax.axis("off");render(fig)
        with col2:
            st.metric("Mean intensity",f"{img.mean():.2f}")
            st.metric("Nonzero pixels (ink)",f"{int(np.count_nonzero(img))} / 64")
            st.caption("A nonzero pixel contains some digit stroke intensity. This does not indicate image quality by itself.")
    with b2:
        upload=st.file_uploader("Upload PNG or JPG",type=["png","jpg","jpeg"])
        if upload is not None:
            original=Image.open(upload).convert("RGB")
        else:
            sample=(digits.images[0]/16*255).astype(np.uint8)
            original=Image.fromarray(sample).convert("RGB").resize((256,256),Image.Resampling.NEAREST)
            st.caption("No image uploaded — using an enlarged digits sample as a demo.")
        method=st.selectbox("Processing method",["Grayscale","Blur","Edges","Invert"])
        if method=="Grayscale":processed=ImageOps.grayscale(original).convert("RGB")
        elif method=="Blur":processed=original.filter(ImageFilter.GaussianBlur(radius=3))
        elif method=="Edges":processed=original.filter(ImageFilter.FIND_EDGES)
        else:processed=ImageOps.invert(original)
        a,b=st.columns(2)
        a.image(original,caption="Original",use_container_width=True)
        b.image(processed,caption=method,use_container_width=True)
        st.caption(f"Image size: {original.width} × {original.height} pixels. Mean RGB: {np.asarray(original).mean(axis=(0,1)).round(1).tolist()}")
        if upload is not None:
            arr=np.asarray(original)
            fig,ax=plt.subplots(figsize=(10,3.5))
            for channel,name,color in [(0,"Red","#f48c86"),(1,"Green","#8cdeaa"),(2,"Blue","#91a7f6")]:
                ax.hist(arr[...,channel].ravel(),bins=40,alpha=.48,density=True,label=name,color=color)
            ax.set_title("RGB channel distributions",fontsize=12);ax.set_xlabel("Channel intensity (0–255)");ax.set_ylabel("Density")
            ax.legend(facecolor=PANEL,edgecolor="#424e5b",labelcolor=FG)
            fig.tight_layout();render(fig)

with t_findings:
    section("Research insights", "A compact, reproducible summary of the selected class, its pixel structure, and the classifier's behavior. Values update when you change the selected digit.")
    class_ids = np.flatnonzero(digits.target == selected)
    class_imgs = digits.images[class_ids]
    flat = class_imgs.ravel()
    counts = np.bincount(digits.target, minlength=10)
    model, test_ids, accuracy = fitted_model()
    selected_test = np.sort(test_ids[digits.target[test_ids] == selected])
    preds = model.predict(digits.data[selected_test])
    class_recall = float(np.mean(preds == selected))
    mean_img = class_imgs.mean(axis=0)
    deviations = np.linalg.norm((class_imgs - mean_img).reshape(len(class_imgs), -1), axis=1)
    exemplar_id = int(class_ids[np.argmax(deviations)])
    a, b, c = st.columns(3)
    a.metric("Selected-class images", f"{len(class_ids):,}")
    b.metric("Zero-intensity pixels", f"{100*np.mean(flat == 0):.1f}%")
    c.metric("Held-out class recall", f"{class_recall:.1%}")

    st.markdown("### 01 / Collection diversity")
    left, right = st.columns([1, 2])
    with left:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(digits.images[exemplar_id], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(f"Most distant from class mean · ID {exemplar_id}", fontsize=10)
        ax.axis("off")
        render(fig)
    with right:
        st.write(f"Class **{selected}** contains **{len(class_ids)}** images (the dataset spans {int(counts.min())}–{int(counts.max())} examples per class).")
        st.write(f"Image **ID {exemplar_id}** has the largest Euclidean pixel-space distance from this class's mean image. It is a candidate for visual inspection, **not proof of a wrong label**.")
        st.caption("Explore the montage to compare stroke thickness, shape, position, and unusual handwriting.")

    st.markdown("### 02 / Pixel structure")
    st.write(f"Mean grayscale intensity is **{flat.mean():.2f} / 16**; **{100*np.mean(flat == 0):.1f}%** of pixels are black (0), and **{100*np.mean(flat >= 12):.1f}%** have brightness ≥ 12.")
    st.write("These values summarize the amount and strength of visible ink, but do not capture stroke geometry. Use the distribution chart to see information hidden by the mean.")

    st.markdown("### 03 / Model sensitivity")
    st.write(f"A logistic-regression classifier achieves **{accuracy:.1%}** overall accuracy on a stratified held-out split and **{class_recall:.1%}** recall for digit {selected} on its held-out examples (**{len(selected_test)}** images).")
    st.write("The saliency view applies a 3 × 3 zero-valued occlusion patch to a held-out image, one location at a time. Positive probability drops reveal regions that support the target prediction in this particular test.")
    st.caption("Interpretation boundary: occlusion indicates model sensitivity under this intervention, not causal understanding. Blanking an already-black region cannot reveal its effect; inspect potential dataset artifacts separately.")

    st.divider()
    st.markdown("### Methodology & reproducibility")
    st.write("**Data:** scikit-learn handwritten digits, 1,797 grayscale 8 × 8 images; intensities 0–16. **Training:** logistic regression (max_iter=5000); 70/30 stratified split, random_state=0. **Explanation:** overlapping 3 × 3 patch occlusion, probability change for the true class.")
    st.caption("Analysis figures and metrics are computed from the local dataset; no remote service or uploaded images are required for the primary analysis.")

st.divider()
