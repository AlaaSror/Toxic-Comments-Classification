"""
main.py
-------
FastAPI REST API for ToxiClear inference.

Endpoints:
  POST /predict        → multi-label toxicity scores + rewrite suggestion
  GET  /health         → health check
  GET  /labels         → list of supported labels
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, sys
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilstm import ToxicBiLSTM
from dataset import LABELS, clean_text, MAX_SEQ_LEN
from rewriter import get_rewriter

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD  = 0.5
CHECKPOINT = os.path.join(os.path.dirname(__file__), ".", "checkpoints", "best_model.pt")

# ── Load model on startup ──────────────────────────────────────────────────────
model = None
vocab = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab
    print("1. Starting up...")
    
    if not os.path.exists(CHECKPOINT):
        print("2. No checkpoint found")
    else:
        print("2. Loading checkpoint...")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        print("3. Checkpoint loaded...")
        vocab = ckpt["vocab"]
        model = ToxicBiLSTM(vocab_size=len(vocab)).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("4. Model ready!")
    
    print("5. Yielding to FastAPI...")
    yield
    print("6. Shutting down...")

app = FastAPI(
    title="ToxiClear API",
    description="Multi-label toxic comment detection with constructive rewrite suggestions.",
    version="1.0.0",
    lifespan=lifespan
)

# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = THRESHOLD


class LabelScore(BaseModel):
    label:      str
    probability: float
    flagged:    bool


class PredictResponse(BaseModel):
    text:            str
    is_toxic:        bool
    flagged_labels:  list[str]
    scores:          list[LabelScore]
    rewrite:         Optional[str]
    confidence:      float


# ── Rewrite suggestion (pretrained BART detoxification) ───────────────────────
def suggest_rewrite(text: str, flagged: list[str]) -> str:
    """
    Generate a clean paraphrase using s-nlp/bart-base-detoxification.
    Only called when at least one toxicity label is flagged.

    Args:
        text    : original (potentially toxic) comment
        flagged : list of triggered label names

    Returns:
        Detoxified paraphrase string
    """
    if not flagged:
        return text
    rewriter = get_rewriter()
    return rewriter.rewrite(text)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/labels")
def get_labels():
    return {"labels": LABELS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run trainer.py first.")

    cleaned = clean_text(req.text)
    ids     = vocab.encode(cleaned, MAX_SEQ_LEN)
    x       = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    probs   = model.predict_proba(x)[0].tolist()

    scores = [
        LabelScore(
            label=label,
            probability=round(prob, 4),
            flagged=prob >= req.threshold,
        )
        for label, prob in zip(LABELS, probs)
    ]

    flagged_labels = [s.label for s in scores if s.flagged]
    is_toxic       = len(flagged_labels) > 0
    confidence     = max(probs)
    rewrite        = suggest_rewrite(req.text, flagged_labels) if is_toxic else None

    return PredictResponse(
        text=req.text,
        is_toxic=is_toxic,
        flagged_labels=flagged_labels,
        scores=scores,
        rewrite=rewrite,
        confidence=round(confidence, 4),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)