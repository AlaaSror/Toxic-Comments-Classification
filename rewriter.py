"""
rewriter.py
-----------
Constructive rewrite module using a pretrained seq2seq detoxification model.

Model: s-nlp/bart-base-detoxification
  - Fine-tuned BART-base on ParaDetox dataset (toxic → clean paraphrase pairs)
  - HuggingFace: https://huggingface.co/s-nlp/bart-base-detoxification
  - Task: given a toxic sentence, generate a semantically similar but non-toxic version

Usage:
    from utils.rewriter import Rewriter
    rewriter = Rewriter()
    clean = rewriter.rewrite("You are absolutely worthless.")
    # → "You have different priorities."
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

MODEL_NAME = "s-nlp/bart-base-detoxification"


class Rewriter:
    """
    Wraps the pretrained BART detoxification model for inference.

    Args:
        model_name  : HuggingFace model ID (default: s-nlp/bart-base-detoxification)
        device      : torch device — auto-detected if not provided
        max_length  : max tokens in generated output
        num_beams   : beam search width (higher = better quality, slower)
    """

    def __init__(
        self,
        model_name: str  = MODEL_NAME,
        device:     Optional[torch.device] = None,
        max_length: int  = 128,
        num_beams:  int  = 4,
    ):
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams  = num_beams

        print(f"Loading detoxification model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Rewriter model ready.")

    def rewrite(self, text: str) -> str:
        """
        Generate a detoxified (clean) version of the input text.

        Args:
            text : potentially toxic input sentence

        Returns:
            Clean paraphrase string
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        clean_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return clean_text

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        """
        Rewrite a batch of texts in one forward pass (faster for bulk inference).

        Args:
            texts : list of potentially toxic strings

        Returns:
            List of clean paraphrases
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]


# ── Lazy singleton — load once, reuse everywhere ───────────────────────────────
_rewriter_instance: Optional[Rewriter] = None


def get_rewriter() -> Rewriter:
    """Return a cached singleton Rewriter instance."""
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = Rewriter()
    return _rewriter_instance
