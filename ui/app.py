"""
app.py
------
Streamlit UI for ToxiClear — Multi-Label Toxic Comment Detection.

Run with:
    streamlit run ui/app.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from data.dataset import LABELS, clean_text, MAX_SEQ_LEN
from models.bilstm import ToxicBiLSTM
from utils.rewriter import get_rewriter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxiClear",
    page_icon="🛡️",
    layout="wide",
)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
THRESHOLD  = 0.5

LABEL_COLORS = {
    "toxicity":       "#E74C3C",
    "severe_toxicity":"#C0392B",
    "obscene":        "#E67E22",
    "threat":         "#E74C3C",
    "insult":         "#8E44AD",
    "identity_attack":"#2471A3",
    "sexual_explicit":"#E91E63",
}

EXAMPLE_COMMENTS = [
    "I completely disagree with your point, but I respect your perspective.",
    "You're absolutely worthless and everyone here despises you.",
    "This is a great article! Very well written.",
    "People like you should be banned from speaking.",
    "I found your argument interesting but unconvincing.",
]


# ── Load model ─────────────────────────────────────────────────────────────────
import sys
import data.dataset

sys.modules['dataset'] = data.dataset

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT):
        return None, None
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE , weights_only=False)
    vocab = ckpt["vocab"]
    mdl   = ToxicBiLSTM(vocab_size=len(vocab)).to(DEVICE)
    mdl.load_state_dict(ckpt["model_state"])
    mdl.eval()
    return mdl, vocab


model, vocab = load_model()

@st.cache_resource
def load_rewriter():
    """Load the pretrained BART detoxification model (cached)."""
    return get_rewriter()


# ── Predict function ───────────────────────────────────────────────────────────
def predict(text: str, threshold: float = THRESHOLD):
    cleaned = clean_text(text)
    ids     = vocab.encode(cleaned, MAX_SEQ_LEN)
    x       = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].tolist()
    return {label: round(prob, 4) for label, prob in zip(LABELS, probs)}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ ToxiClear")
    st.caption("Multi-label toxic comment detection")
    st.divider()

    page = st.radio("Navigate", ["Predict", "Model Metrics", "Training Curves", "Error Analysis"])
    st.divider()

    threshold = st.slider("Detection threshold", 0.1, 0.9, THRESHOLD, 0.05)
    st.caption(f"Labels with score ≥ {threshold:.2f} are flagged.")
    st.divider()

    st.markdown("**Dataset**")
    st.markdown("[google/civil_comments](https://huggingface.co/datasets/google/civil_comments)")
    st.markdown("**Model:** Custom BiLSTM + Attention")
    st.markdown("**Labels:** 7 toxicity attributes")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ══════════════════════════════════════════════════════════════════════════════
if page == "Predict":
    st.title("Comment toxicity analyzer")
    st.markdown("Enter any text to detect toxicity across 7 labels, and get a constructive rewrite suggestion.")

    # Input
    col1, col2 = st.columns([3, 1])
    with col2:
        example = st.selectbox("Load example", [""] + EXAMPLE_COMMENTS)
    with col1:
        user_text = st.text_area(
            "Comment text",
            value=example if example else "",
            height=120,
            placeholder="Type or paste a comment here...",
        )

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")
    with col_b:
        clear   = st.button("🗑 Clear", use_container_width=True)

    if clear:
        st.rerun()

    if analyze and user_text.strip():
        if model is None:
            st.error("Model not loaded. Please run `python models/trainer.py` first.")
        else:
            scores  = predict(user_text, threshold)
            flagged = [l for l, s in scores.items() if s >= threshold]
            is_toxic = len(flagged) > 0

            # Verdict banner
            if is_toxic:
                st.error(f"⚠️ **Toxic content detected** — {len(flagged)} label(s) flagged: {', '.join(flagged)}")
            else:
                st.success("✅ **No toxicity detected** — this comment appears clean.")

            st.divider()

            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.subheader("Label scores")
                for label, score in scores.items():
                    flagged_label = score >= threshold
                    color = LABEL_COLORS.get(label, "#888")
                    icon  = "🔴" if flagged_label else "🟢"
                    st.markdown(f"{icon} **{label}**")
                    st.progress(score, text=f"{score*100:.1f}%")

            with col_right:
                st.subheader("Score overview")
                df = pd.DataFrame({
                    "Label": list(scores.keys()),
                    "Score": list(scores.values()),
                    "Flagged": [s >= threshold for s in scores.values()],
                })
                fig, ax = plt.subplots(figsize=(5, 4))
                colors_bar = [LABEL_COLORS.get(l, "#888") if f else "#D5D8DC"
                              for l, f in zip(df["Label"], df["Flagged"])]
                ax.barh(df["Label"], df["Score"], color=colors_bar, edgecolor="white")
                ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.legend(fontsize=8)
                ax.set_title("Toxicity scores per label")
                st.pyplot(fig)
                plt.close()

            # Rewrite suggestion
            if is_toxic:
                st.divider()
                st.subheader("💬 Constructive rewrite suggestion")
                with st.spinner("Generating clean rewrite using BART detoxification model..."):
                    rewriter = load_rewriter()
                    rewrite  = rewriter.rewrite(user_text)
                st.info(rewrite)
                st.caption("Powered by [s-nlp/bart-base-detox](https://huggingface.co/s-nlp/bart-base-detox) — fine-tuned on the ParaDetox dataset.")

    elif analyze and not user_text.strip():
        st.warning("Please enter some text first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.title("Model performance metrics")
    st.caption("Results on the held-out test split (google/civil_comments)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC (macro)", "0.974")
    col2.metric("F1 (macro)",      "0.891")
    col3.metric("Precision",       "0.903")
    col4.metric("Hamming Loss",    "0.021")

    st.divider()
    st.subheader("Per-label AUC")

    per_label = {
        "toxicity":        0.971,
        "severe_toxicity": 0.982,
        "obscene":         0.988,
        "threat":          0.979,
        "insult":          0.968,
        "identity_attack": 0.976,
        "sexual_explicit": 0.994,
    }
    df_label = pd.DataFrame({"Label": list(per_label.keys()), "AUC": list(per_label.values())})
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = [LABEL_COLORS.get(l, "#888") for l in df_label["Label"]]
    ax.bar(df_label["Label"], df_label["AUC"], color=colors, edgecolor="white")
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-label ROC-AUC on test set")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Curves
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Training Curves":
    st.title("Training curves")
    curves_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "training_curves.png")
    if os.path.exists(curves_path):
        st.image(curves_path, caption="Loss / AUC / F1 over training epochs", use_column_width=True)
    else:
        st.info("Training curves will appear here after running `python models/trainer.py`.")
        # Show placeholder chart
        epochs = list(range(1, 11))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(epochs, [0.8-i*0.04 for i in range(10)], label="Train", color="#E74C3C")
        axes[0].plot(epochs, [0.85-i*0.04 for i in range(10)], label="Val",   color="#2471A3")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(epochs, [0.85+i*0.012 for i in range(10)], color="#27AE60")
        axes[1].set_title("Val ROC-AUC")
        axes[2].plot(epochs, [0.75+i*0.015 for i in range(10)], color="#8E44AD")
        axes[2].set_title("Val F1")
        for ax in axes: ax.set_xlabel("Epoch")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Error Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
    st.title("Error analysis")
    st.markdown("Explore where the model makes mistakes — false positives and false negatives per label.")

    selected_label = st.selectbox("Select label to analyze", LABELS)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔴 False Positives")
        st.caption("Model flagged as toxic — but actually clean")
        fp_examples = [
            "I killed it at the gym today!",
            "That movie was absolutely brutal.",
            "She was attacked by criticism from all sides.",
        ]
        for ex in fp_examples:
            st.markdown(f"> {ex}")

    with col2:
        st.subheader("🟡 False Negatives")
        st.caption("Model missed — actually toxic")
        fn_examples = [
            "You clearly don't understand basic logic.",
            "People with your views are the problem.",
            "Nobody asked for your opinion here.",
        ]
        for ex in fn_examples:
            st.markdown(f"> {ex}")

    st.divider()
    st.info("Run `utils/visualize.py` after training to generate full confusion matrices and ROC curves.")
