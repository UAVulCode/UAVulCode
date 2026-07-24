"""
╔══════════════════════════════════════════════════════════════════════╗
║   UAVulDetect — Multi-Round Vulnerability Detection Framework        ║
║   GUI Application — Python / Tkinter                   V16          ║
║                                                                      ║
║   Pipeline Steps (corrected in V16):                                 ║
║   STEP 1  — Load dataset (UAVulDB01 CSV/XLSX)                       ║
║   STEP 2  — Deduplication (SHA-256 snippet hash)                    ║
║   STEP 3  — Split 80% train / 20% test  (Mtag-stratified, shared)  ║
║   STEP 4  — GraphCodeBERT embedding extraction (if R2/R3/R4 needed) ║
║             microsoft/graphcodebert-base → [CLS] 768-dim            ║
║             + 12-dim handcrafted features = 780-dim (for R4)        ║
║             Skipped automatically when only Round 1 is selected.    ║
║   STEP 5  — Code snippet data note for fine-tuned Rounds 1 & 2     ║
║             CB / GCB loaded on-demand inside run_bert_round()        ║
║             (static pre-extraction is incompatible with fine-tuning) ║
║   STEP 6  — Round 1: CodeBERT Fine-tuned  (from raw snippets)      ║
║             [CLS] 768 → Linear(768→69) → Softmax                    ║
║             1A: Binary  |  1B: Multi-class (69 classes)             ║
║   STEP 7  — Round 2: GraphCodeBERT Fine-tuned  (from raw snippets) ║
║             [CLS] 768 → Linear(768→69) → Softmax                    ║
║             2A: Binary  |  2B: Multi-class (69 classes)             ║
║   STEP 8  — Round 3: GraphCodeBERT + XGBoost (STEP 4, 768-dim)     ║
║             No handcrafted features — GCB [CLS] 768-dim only        ║
║             3A: Binary standalone  |  3B: Multi-class flat (69 cls) ║
║   STEP 9  — Round 4: Two-Stage Hierarchical XGBoost (STEP 4, 780d) ║
║             Stage-1 binary gate → Stage-2 multi-class (Vuln only)  ║
║             4A: Binary standalone  |  4B: Multi-class hierarchical  ║
║   STEP 10 — Generate PDF report + comparison charts                 ║
║                                                                      ║
║   Key V12 fixes:                                                     ║
║     C2: Redundant CB static-extraction step removed — fine-tuned    ║
║         rounds load backbone on-demand inside run_bert_round().      ║
║     C3: Global padding removed from extractor (was padding=True).   ║
║     S2: Alignment assertion added to run_bert_round().              ║
║     S3: Step 4 GCB extraction guarded by round-selection flags.     ║
║     M1: gc.collect() + VRAM logging after model cleanup.            ║
║     M2: Snippet slicing uses NumPy object-array indexing.           ║
║     M3: Explicit ValueError when SNIPPET column absent in Step 4.   ║
║     M4: Progress budget re-calibrated (rounds start at 32%).        ║
║                                                                      ║
║   Default hyperparameters:                                           ║
║     Max tokens = 512  ·  Epochs = 5  ·  Batch = 32  ·  LR = 2e-5  ║
║                                                                      ║
║   Requirements:                                                      ║
║     pip install numpy pandas scikit-learn imbalanced-learn xgboost   ║
║     pip install reportlab matplotlib torch transformers accelerate   ║
║     pip install openpyxl                                             ║
║                                                                      ║
║   Run:  python script_V16.py                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import time
import re
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
#  Scientific / ML
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

# ─────────────────────────────────────────────────────────────────────
#  XGBoost (two-stage hierarchical classifier)
# ─────────────────────────────────────────────────────────────────────
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────
#  SMOTE (applied to training partition only — post-split)
# ─────────────────────────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────────────────────────────
#  GraphCodeBERT embedding
# ─────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────────────
#  Excel I/O (precomputed feature-dataset export / import)
# ─────────────────────────────────────────────────────────────────────
import openpyxl

GRAPHCODEBERT_MODEL_NAME = "microsoft/graphcodebert-base"
CODEBERT_MODEL_NAME      = "microsoft/codebert-base"
EMBEDDING_DIM = 768
HC_DIM = 12              # handcrafted feature dimensions
FEATURE_DIM = EMBEDDING_DIM + HC_DIM   # 780

# Fine-tuning defaults (Rounds 1 & 2) — updated V08
FT_EPOCHS_DEFAULT    = 5
FT_LR_DEFAULT        = 2e-5
FT_BATCH_DEFAULT     = 32    # updated from 16 → 32
FT_MAX_LEN_DEFAULT   = 512   # updated from 128 → 512

# Column names used when exporting/importing the concatenated
# (768-dim GraphCodeBERT + 12-dim handcrafted) feature dataset to/from
# Excel (.xlsx). Keeping these as module-level constants guarantees the
# export (Step 3) and import (precomputed-dataset checkbox) paths stay
# perfectly in sync.
EMB_COLS = [f"gcb_emb_{i}" for i in range(EMBEDDING_DIM)]

# ─────────────────────────────────────────────────────────────────────
#  PDF + charting
# ─────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, PageBreak,
                                 HRFlowable, Image as RLImage)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════
DARK_BG      = "#1A1D2E"
PANEL_BG     = "#252840"
CARD_BG      = "#2E3250"
ACCENT_BLUE  = "#4F8EF7"
ACCENT_GREEN = "#3EC97A"
ACCENT_AMBER = "#F5A623"
ACCENT_RED   = "#E95B5B"
ACCENT_TEAL  = "#3DD6C8"
TEXT_PRIMARY = "#E8EAF6"
TEXT_MUTED   = "#8892B0"
BORDER       = "#3A3F5C"
BTN_EXEC     = "#3A7EF7"
BTN_STOP     = "#E95B5B"
BTN_BROWSE   = "#3EC97A"

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
#  HANDCRAFTED STRUCTURAL FEATURES  (12-dim per snippet)
#
#  These complement GraphCodeBERT [CLS] embeddings with lightweight
#  structural signals that transformers can under-represent when
#  processing short or heavily truncated code snippets.
#
#  Features (all normalised to [0,1]):
#    [0]  Snippet length     (chars / 3000)
#    [1]  Token count        (whitespace tokens / 500)
#    [2]  Brace-depth proxy  |{count − }count| / 30
#    [3]  Pointer ops        (* + & + ->) / 50
#    [4]  Memory func hits   (malloc/free/strcpy…) / 10
#    [5]  Numerical literals / 100
#    [6]  Comment density    (// + /* occurrences) / 20
#    [7-11] Language one-hot (C | C++/CPP | Java | PHP | other)
# ═══════════════════════════════════════════════════════════════════
# Column names matching the order above 1:1 — used for Excel export
# (Step 3) and precomputed-dataset import (Dataset card checkbox).
HC_FEATURE_NAMES = [
    "hc_snippet_len", "hc_token_count", "hc_brace_depth",
    "hc_pointer_ops", "hc_mem_func_hits", "hc_num_literals",
    "hc_comment_density",
    "hc_lang_C", "hc_lang_CPP", "hc_lang_Java", "hc_lang_PHP",
    "hc_lang_other",
]
assert len(HC_FEATURE_NAMES) == HC_DIM

_MEM_RE = re.compile(
    r"\b(malloc|calloc|realloc|free|strcpy|strncpy|memcpy|"
    r"memmove|sprintf|gets)\b")
_LANG_MAP = {
    "C": 0, "CPP": 1, "C++": 1,
    "JAVA": 2, "Java": 2,
    "PHP": 3,
}  # everything else → 4 ("other")


def extract_handcrafted(snippets: list[str],
                        file_types: list[str]) -> np.ndarray:
    """Return (N, 12) float32 structural feature matrix."""
    out = np.zeros((len(snippets), HC_DIM), dtype=np.float32)
    for i, (s, ft) in enumerate(zip(snippets, file_types)):
        lv = [0] * 5
        lv[_LANG_MAP.get(str(ft).strip(), 4)] = 1
        out[i] = [
            min(len(s), 3000) / 3000,
            min(len(s.split()), 500) / 500,
            min(abs(s.count("{") - s.count("}")), 30) / 30,
            min(s.count("*") + s.count("&") + s.count("->"), 50) / 50,
            min(len(_MEM_RE.findall(s)), 10) / 10,
            min(len(re.findall(r"\b\d+\b", s)), 100) / 100,
            min(s.count("//") + s.count("/*"), 20) / 20,
        ] + lv
    return out


# ═══════════════════════════════════════════════════════════════════
#  GRAPHCODEBERT EMBEDDER
#
#  Produces 768-dim [CLS]-token embeddings per snippet.
#  Embeddings are L2-normalised after extraction.
#  Supports three pooling strategies (selectable via GUI):
#    cls  — [CLS] token from last hidden state (default, best for
#            function-level snippets in vulnerability detection)
#    mean — mean-pool over non-padding tokens
#    max  — max-pool over non-padding tokens
# ═══════════════════════════════════════════════════════════════════
class GraphCodeBERTEmbedder:
    def __init__(self, device, model_name=GRAPHCODEBERT_MODEL_NAME):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self, log_fn=None):
        if self.tokenizer is None:
            t0 = time.time()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except (OSError, EnvironmentError) as exc:
                local_name = self.model_name.replace("/", "_")
                msg = (
                    f"Cannot load '{self.model_name}' from HuggingFace Hub.\n"
                    f"Cause: {exc}\n\n"
                    "Fix: pre-download the model on an internet-connected "
                    "machine and copy it locally:\n"
                    f"  python -c \"from transformers import AutoTokenizer, "
                    f"AutoModel; "
                    f"AutoTokenizer.from_pretrained('{self.model_name}')"
                    f".save_pretrained('./models/{local_name}'); "
                    f"AutoModel.from_pretrained('{self.model_name}')"
                    f".save_pretrained('./models/{local_name}')\"\n"
                    "Then enter the local folder path as the model name, or "
                    "set TRANSFORMERS_OFFLINE=1 if a local cache already "
                    "exists."
                )
                if log_fn:
                    log_fn(f"❌  {msg}", "error")
                raise RuntimeError(msg) from exc
            self.model.to(self.device)
            self.model.eval()
            if log_fn:
                log_fn(f"   Model loaded      : {self.model_name} "
                       f"on {self.device}  ({time.time()-t0:.1f}s)")

    @torch.no_grad()
    def embed(self, snippets: list[str], max_len: int,
              batch_size: int, pooling: str,
              progress_fn=None, check_stop_fn=None,
              pct_start=14, pct_end=30) -> np.ndarray:
        n = len(snippets)
        X = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        for start in range(0, n, batch_size):
            if check_stop_fn:
                check_stop_fn()
            batch = snippets[start: start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc).last_hidden_state   # (B, T, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()

            if pooling == "cls":
                pooled = out[:, 0, :]
            elif pooling == "max":
                masked = out.masked_fill(mask == 0, float("-inf"))
                pooled = masked.max(dim=1).values
            else:  # mean
                summed = (out * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            X[start: start + len(batch)] = pooled.cpu().numpy()

            if progress_fn and (start // batch_size) % 20 == 0:
                pct = pct_start + int(
                    (pct_end - pct_start) * start / max(n, 1))
                progress_fn(pct,
                            f"GraphCodeBERT embedding {start:,}/{n:,} …")

        return normalize(X, norm="l2").astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  BERT FINE-TUNED CLASSIFIER  (Rounds 4 & 5)
#
#  Architecture:
#    Code snippets → Transformer (GCB or CB) → [CLS] 768-dim →
#    Linear(768, n_classes) → Softmax
#
#  Two sub-rounds per backbone:
#    a) Binary  : n_classes=2  (Benign / Vulnerable)
#    b) Multi   : n_classes=N  (all CWE-ID classes)
#
#  Training:  AdamW + linear warmup + cross-entropy loss.
#  Evaluation: accuracy, MCC, macro-F1, weighted-F1.
# ═══════════════════════════════════════════════════════════════════
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import AdamW

try:
    from transformers import get_linear_schedule_with_warmup
    _HAS_LR_SCHEDULER = True
except ImportError:
    _HAS_LR_SCHEDULER = False


class _CodeDataset(TorchDataset):
    """Minimal PyTorch Dataset wrapping tokenised code snippets."""
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels    = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTClassifier(nn.Module):
    """Transformer backbone + single linear classification head.

    Architecture:
        backbone (GCB or CB) → [CLS] → Linear(768, n_classes) → logits
    Softmax is applied by CrossEntropyLoss during training and
    explicitly at inference time via torch.softmax.
    """

    def __init__(self, model_name: str, n_classes: int, device):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)
        self.n_classes  = n_classes
        self.device     = device
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out    = self.backbone(**kwargs)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits


def run_bert_round(snippets_tr, snippets_te,
                   y_tr, y_te,
                   model_name: str,
                   n_classes: int,
                   le,
                   tag: str,
                   device,
                   epochs: int,
                   lr: float,
                   batch_size: int,
                   max_len: int,
                   log_fn,
                   progress_fn,
                   check_stop_fn,
                   pct_start: int = 0,
                   pct_end: int   = 20) -> dict:
    """
    Fine-tune a Transformer backbone with a linear classification head
    and return a standard metrics dict compatible with V09 round results.

    Fixes applied vs V09
    --------------------
    C1  Dynamic per-batch tokenisation via collate_fn — eliminates
        whole-corpus pre-tokenisation at max_len=512 and global padding.
    C2  Weighted CrossEntropyLoss — class weights computed from training
        label distribution so minority CWE classes are not dominated by
        Benign in both binary and multi-class sub-rounds.
    S1  Validation split + best-checkpoint selection — 10% of the training
        fold is held out; the checkpoint with the lowest validation loss is
        saved and reloaded before evaluation.
    S2  Correct Benign/Vuln count for multi-class sub-rounds — Benign
        index is looked up from the LabelEncoder rather than assumed to
        be label 0.
    S3  Seeded DataLoader and deterministic PyTorch state — reproducible
        across runs with SEED=42.

    Parameters
    ----------
    snippets_tr / snippets_te : list[str]  — raw code text
    y_tr / y_te               : np.ndarray — integer-encoded labels
    model_name                : HuggingFace model identifier
    n_classes                 : number of output classes
    le                        : fitted LabelEncoder for inverse-transform
    tag                       : display name for logging
    device                    : torch.device
    epochs, lr, batch_size, max_len : fine-tuning hyperparameters
    """
    import tempfile, os as _os, gc as _gc
    from sklearn.metrics import (accuracy_score, f1_score,
                                  matthews_corrcoef,
                                  classification_report,
                                  confusion_matrix)
    from sklearn.model_selection import train_test_split as _tts

    check_stop_fn()
    log_fn(f"\n   ── {tag} ──")
    progress_fn(pct_start, f"{tag} – loading tokeniser …")

    # S2 fix: assert alignment between text list and label array
    assert len(snippets_tr) == len(y_tr), (
        f"run_bert_round: snippets_tr ({len(snippets_tr)}) and "
        f"y_tr ({len(y_tr)}) must have the same length.")
    assert len(snippets_te) == len(y_te), (
        f"run_bert_round: snippets_te ({len(snippets_te)}) and "
        f"y_te ({len(y_te)}) must have the same length.")

    # Convert to NumPy object arrays so idx_ft / idx_val integer indexing
    # is unambiguous regardless of whether lists or arrays are passed.
    snippets_tr = np.array(snippets_tr, dtype=object)
    snippets_te = np.array(snippets_te, dtype=object)

    # S3 fix: seed all random sources for reproducibility
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    np.random.seed(SEED)

    # Bug 2 fix: wrap tokeniser + model loading with a clear, actionable
    # error message for environments without internet access or a local
    # HuggingFace model cache.  HuggingFace raises OSError when it cannot
    # reach the hub AND no locally-cached copy exists.
    def _hf_load_error(model_name_str, exc):
        local_name = model_name_str.replace("/", "_")
        return RuntimeError(
            f"Cannot load '{model_name_str}' from HuggingFace Hub.\n\n"
            f"Root cause: {exc}\n\n"
            "How to fix this on an offline / restricted machine:\n\n"
            "  Option A — Pre-download on an internet-connected machine:\n"
            f"    python -c \"\n"
            f"      from transformers import AutoTokenizer, AutoModel\n"
            f"      AutoTokenizer.from_pretrained('{model_name_str}').save_pretrained('./models/{local_name}')\n"
            f"      AutoModel.from_pretrained('{model_name_str}').save_pretrained('./models/{local_name}')\n"
            f"    \"\n"
            f"  Then set model_name = './models/{local_name}' in the GUI\n"
            "  (or copy the folder to the target machine and point to it).\n\n"
            "  Option B — Set HF_HOME to a folder containing the cache:\n"
            f"    set HF_HOME=D:\\hf_cache   (Windows)\n"
            f"    export HF_HOME=/data/hf_cache   (Linux/Mac)\n\n"
            "  Option C — Enable TRANSFORMERS_OFFLINE mode if you already\n"
            "  have a local cache but HuggingFace is trying to refresh it:\n"
            "    set TRANSFORMERS_OFFLINE=1   (Windows)\n"
            "    export TRANSFORMERS_OFFLINE=1   (Linux/Mac)"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (OSError, EnvironmentError) as exc:
        err = _hf_load_error(model_name, exc)
        log_fn(f"❌  {err}", "error")
        raise err from exc

    # C1 fix: tokenise per batch, not per full corpus
    # Each call to collate_fn tokenises only the current mini-batch,
    # using padding="longest" so padding is to the longest sequence
    # in that batch (not the global max_len).
    def _collate(batch):
        texts  = [item["text"]   for item in batch]
        labels = torch.tensor([item["label"] for item in batch],
                               dtype=torch.long)
        enc = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt")
        enc["labels"] = labels
        return enc

    # S1 fix: carve a 10% validation fold from the training partition
    # BEFORE any training. This fold is used to select the best epoch
    # checkpoint; it never touches the shared test set.
    try:
        idx_ft, idx_val = _tts(
            np.arange(len(snippets_tr)),
            test_size=0.10,
            random_state=SEED,
            stratify=y_tr)
    except ValueError:
        idx_ft, idx_val = _tts(
            np.arange(len(snippets_tr)),
            test_size=0.10,
            random_state=SEED)

    snips_ft  = snippets_tr[idx_ft].tolist()
    y_ft      = y_tr[idx_ft]
    snips_val = snippets_tr[idx_val].tolist()
    y_val     = y_tr[idx_val]

    log_fn(f"     Fine-tune fold   : {len(snips_ft):,} samples  "
           f"(val fold: {len(snips_val):,})")

    # C1 fix: datasets store raw text dicts, not pre-tokenised tensors
    class _TextDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.texts  = texts
            self.labels = labels.astype(np.int64)
        def __len__(self):  return len(self.labels)
        def __getitem__(self, i):
            return {"text": self.texts[i], "label": int(self.labels[i])}

    ds_ft  = _TextDataset(snips_ft,              y_ft)
    ds_val = _TextDataset(snips_val,             y_val)
    ds_te  = _TextDataset(snippets_te.tolist(),  y_te)

    # ── CPU performance guard ──────────────────────────────────
    # Running max_len=512 with batch_size=32 through CodeBERT/GCB on CPU
    # takes 30–90 seconds PER BATCH (attention is O(n²) in sequence length).
    # With 4,000+ training samples this appears as a total hang and can
    # trigger OOM-induced silent process death on machines with <16 GB RAM.
    # Automatically apply conservative CPU defaults to keep the training
    # loop responsive; GPU runs are unchanged.
    _is_cpu          = (device.type == "cpu")
    _user_max_len    = max_len      # store the user-configured originals
    _user_batch_size = batch_size   # before any CPU reduction

    if _is_cpu and max_len > 128:
        max_len = 128
        log_fn(f"     ⚠  CPU detected — max_len reduced "
               f"{_user_max_len}→{max_len} to prevent hang  "
               f"(GPU recommended for max_len={_user_max_len})")

    if _is_cpu and batch_size > 8:
        batch_size = 8
        log_fn(f"     ⚠  CPU detected — batch_size reduced "
               f"{_user_batch_size}→{batch_size} to prevent OOM  "
               f"(GPU recommended for batch_size={_user_batch_size})")

    # Gradient-accumulation steps: accumulate this many micro-batches
    # before calling optimiser.step() so the effective batch size matches
    # the user-configured value even when the physical batch was reduced
    # for CPU.  On GPU (no reduction applied) this is always 1 (no-op).
    _accum_steps = max(1, _user_batch_size // batch_size)

    # S3 fix: seeded generator for reproducible shuffle
    _gen = torch.Generator()
    _gen.manual_seed(SEED)

    dl_ft  = DataLoader(ds_ft,  batch_size=batch_size,   shuffle=True,
                        num_workers=0, collate_fn=_collate,
                        pin_memory=(device.type == "cuda"),
                        generator=_gen)
    dl_val = DataLoader(ds_val, batch_size=batch_size*2, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    dl_te  = DataLoader(ds_te,  batch_size=batch_size*2, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    progress_fn(pct_start + 2, f"{tag} – building model …")

    # C2 fix: compute class weights from the fine-tuning fold distribution
    # w_i = N / (n_classes * count_i)  — standard inverse-frequency weighting
    label_counts = np.bincount(y_ft.astype(int), minlength=n_classes)
    label_counts = np.maximum(label_counts, 1)   # avoid divide-by-zero
    class_weights = len(y_ft) / (n_classes * label_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    log_fn(f"     Class weights    : min={class_weights.min():.3f}  "
           f"max={class_weights.max():.3f}  "
           f"(C2 fix: weighted loss for imbalance)")

    try:
        model = BERTClassifier(model_name, n_classes, device)
    except (OSError, EnvironmentError) as exc:
        err = _hf_load_error(model_name, exc)
        log_fn(f"❌  {err}", "error")
        raise err from exc
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps  = (len(dl_ft) // _accum_steps) * epochs
    warmup_steps = max(1, total_steps // 10)
    if _HAS_LR_SCHEDULER:
        scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
    else:
        scheduler = None

    n_batches = len(dl_ft)
    log_fn(f"     {tag}  |  epochs={epochs}  "
           f"batch={batch_size}  lr={lr:.0e}  "
           f"n_classes={n_classes}  max_len={max_len}")
    log_fn(f"     Model            : {model_name}")
    log_fn(f"     Val fold         : {len(snips_val):,}  "
           f"(S1 fix: best-checkpoint selection)")
    log_fn(f"     Batches/epoch    : {n_batches}  "
           f"(grad accum steps={_accum_steps})")
    if _is_cpu:
        log_fn(f"     ⚠  Training on CPU — each batch may take 5–30s. "
               f"Estimated epoch time: {n_batches * 15 // 60}–"
               f"{n_batches * 30 // 60} min. GPU strongly recommended.")

    # S1 fix: save best checkpoint to a temp file
    _ckpt_dir  = tempfile.mkdtemp(prefix="uavul_bert_")
    _ckpt_path = _os.path.join(_ckpt_dir, "best_model.pt")
    best_val_loss = float("inf")
    best_epoch    = 0

    # ── Training loop ──────────────────────────────────────────
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        check_stop_fn()

        # --- train ---
        model.train()
        total_loss  = 0.0
        _step_count = 0
        optimiser.zero_grad()

        for batch_idx, batch in enumerate(dl_ft):
            check_stop_fn()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch.get("token_type_ids")
            if ttype is not None:
                ttype = ttype.to(device)
            lbls  = batch["labels"].to(device)

            # C2 fix: use weighted criterion instead of model's own loss
            # Divide by accum_steps so gradients are averaged, not summed.
            _, logits = model(ids, mask, ttype)
            loss = criterion(logits, lbls) / _accum_steps
            loss.backward()
            total_loss += loss.item() * _accum_steps

            # Step the optimiser every _accum_steps micro-batches
            if (batch_idx + 1) % _accum_steps == 0 or \
               (batch_idx + 1) == n_batches:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                if scheduler:
                    scheduler.step()
                optimiser.zero_grad()
                _step_count += 1

            # Per-batch progress — keeps the GUI alive and shows the
            # user that training is running (critical on CPU where each
            # batch can take tens of seconds with no other feedback).
            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or \
               batch_idx == 0:
                elapsed = time.time() - t0
                _frac   = ((epoch - 1) * n_batches + batch_idx + 1) / \
                          max(epochs * n_batches, 1)
                eta_s   = int(elapsed / max(_frac, 1e-6) * (1 - _frac))
                pct_cur = pct_start + int(
                    (pct_end - pct_start - 4) * _frac)
                progress_fn(
                    pct_cur,
                    f"{tag} ep{epoch}/{epochs} "
                    f"batch {batch_idx+1}/{n_batches}  "
                    f"loss={total_loss / max(batch_idx+1,1):.4f}  "
                    f"ETA {eta_s//60}m{eta_s%60:02d}s")
                log_fn(f"     Ep{epoch} batch {batch_idx+1:>4}/{n_batches}  "
                       f"loss={total_loss / max(batch_idx+1,1):.4f}  "
                       f"elapsed={elapsed:.0f}s  ETA {eta_s//60}m{eta_s%60:02d}s")

        avg_train_loss = total_loss / max(n_batches, 1)

        # S1 fix: evaluate on val fold each epoch
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in dl_val:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None:
                    ttype = ttype.to(device)
                lbls  = batch["labels"].to(device)
                _, logits = model(ids, mask, ttype)
                val_loss_sum += criterion(logits, lbls).item()
        avg_val_loss = val_loss_sum / max(len(dl_val), 1)

        # Save checkpoint if val loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), _ckpt_path)

        pct_done = pct_start + int(
            (pct_end - pct_start - 4) * epoch / epochs)
        progress_fn(pct_done,
                    f"{tag} – epoch {epoch}/{epochs}  "
                    f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")
        log_fn(f"     Epoch {epoch}/{epochs}  "
               f"train-loss={avg_train_loss:.4f}  "
               f"val-loss={avg_val_loss:.4f}"
               + ("  ← best" if epoch == best_epoch else ""))

    train_time = time.time() - t0
    log_fn(f"     Best epoch       : {best_epoch}  "
           f"(val-loss={best_val_loss:.4f})")

    # S1 fix: reload best checkpoint before evaluation
    if _os.path.exists(_ckpt_path):
        model.load_state_dict(torch.load(_ckpt_path,
                                          map_location=device))
        log_fn("     Best checkpoint  : reloaded for evaluation")
    try:
        _os.remove(_ckpt_path)
        _os.rmdir(_ckpt_dir)
    except Exception:
        pass

    # ── Evaluation on shared test set ──────────────────────────
    progress_fn(pct_end - 2, f"{tag} – evaluating …")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dl_te:
            check_stop_fn()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch.get("token_type_ids")
            if ttype is not None:
                ttype = ttype.to(device)
            lbls  = batch["labels"].to(device)
            _, logits = model(ids, mask, ttype)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(lbls.cpu().numpy())

    yp  = np.concatenate(all_preds)
    yte = np.concatenate(all_labels)

    acc  = accuracy_score(yte, yp)
    mcc  = matthews_corrcoef(yte, yp)
    f1M  = f1_score(yte, yp, average="macro",    zero_division=0)
    f1w  = f1_score(yte, yp, average="weighted", zero_division=0)

    label_names = list(le.classes_)
    rep    = classification_report(yte, yp, target_names=label_names,
                                    output_dict=True, zero_division=0)
    cm_mat = confusion_matrix(yte, yp, labels=list(range(n_classes)))

    log_fn(f"     Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    log_fn(f"     MCC       : {mcc:.4f}")
    log_fn(f"     F1-macro  : {f1M:.4f}")
    log_fn(f"     F1-wt     : {f1w:.4f}")
    log_fn(f"     Time      : {train_time:.1f}s  "
           f"(best epoch {best_epoch}/{epochs})")

    progress_fn(pct_end, f"{tag} – done")

    # S2 fix: Benign/Vuln counts — look up Benign index from LabelEncoder
    # so the count is correct for both binary (n=2) and multi-class (n=69)
    try:
        benign_idx = int(le.transform(["Benign"])[0])
    except Exception:
        benign_idx = 0   # fallback for binary encoders where 0 = Benign
    n_tr_benign = int((y_tr == benign_idx).sum())
    n_tr_vuln   = int((y_tr != benign_idx).sum())
    n_te_benign = int((y_te == benign_idx).sum())
    n_te_vuln   = int((y_te != benign_idx).sum())

    # Free GPU memory
    del model
    _gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _used  = torch.cuda.memory_allocated() / 1e9
        _total = torch.cuda.get_device_properties(device).total_memory / 1e9
        log_fn(f"     VRAM after cleanup: {_used:.2f} GB / {_total:.2f} GB")

    return {
        "accuracy":         round(acc, 4),
        "f1_micro":         round(mcc, 4),
        "mcc":              round(mcc, 4),
        "f1_macro":         round(f1M, 4),
        "f1_weighted":      round(f1w, 4),
        "s1_accuracy":      round(acc, 4),
        "s1_f1_macro":      round(f1M, 4),
        "train_time":       round(train_time, 1),
        "best_epoch":       best_epoch,
        "best_val_loss":    round(best_val_loss, 4),
        "n_train":          len(y_tr),
        "n_test":           len(y_te),
        "n_train_benign":   n_tr_benign,
        "n_train_vuln":     n_tr_vuln,
        "n_test_benign":    n_te_benign,
        "n_test_vuln":      n_te_vuln,
        "train_dist":       {},
        "test_dist":        {},
        "train_dist_smote": None,
        "n_classes":        n_classes,
        "smote":            False,
        "report":           rep,
        "confusion_matrix": cm_mat,
        "cm_labels":        label_names,
        "tag":              tag,
        "smote_tag":        "N/A (fine-tuning)",
        "model_name":       model_name,
    }


# ═══════════════════════════════════════════════════════════════════
#  SMOTE  (custom, adapted from original — no changes to logic)
# ═══════════════════════════════════════════════════════════════════
def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k: int, cap: int, tgt: str) -> tuple:
    """SMOTE minority oversampling on training partition only."""
    rng = np.random.RandomState(SEED)
    classes, counts = np.unique(y, return_counts=True)

    if tgt == "mean":
        target_c = int(np.mean(counts))
    elif tgt == "max":
        target_c = int(np.max(counts))
    else:
        target_c = int(np.median(counts))

    Xs, ys = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt >= target_c:
            continue
        needed = min(target_c - cnt, cap)
        Xc = X[y == cls]
        kk = min(k, len(Xc) - 1)
        if kk < 1:
            idx = rng.choice(len(Xc), needed)
            Xs.append(Xc[idx])
            ys.append(np.full(needed, cls))
            continue
        nn_model = NearestNeighbors(n_neighbors=kk + 1, n_jobs=-1).fit(Xc)
        nbrs = nn_model.kneighbors(Xc, return_distance=False)[:, 1:]
        Xnew = []
        for _ in range(needed):
            i = rng.randint(0, len(Xc))
            j = rng.choice(nbrs[i])
            lam = rng.uniform(0, 1)
            Xnew.append(Xc[i] + lam * (Xc[j] - Xc[i]))
        Xnew = normalize(np.array(Xnew, dtype=np.float32), norm="l2")
        Xs.append(Xnew)
        ys.append(np.full(needed, cls))

    Xo = np.vstack(Xs)
    yo = np.concatenate(ys)
    perm = rng.permutation(len(yo))
    return Xo[perm].astype(np.float32), yo[perm]


# ═══════════════════════════════════════════════════════════════════
#  TWO-STAGE XGBOOST CLASSIFIER
#
#  Stage 1: Binary XGBoost (Benign vs. Vulnerable)
#    - scale_pos_weight = neg/pos (handles class imbalance directly)
#    - early_stopping_rounds on 10% internal validation split
#
#  Stage 2 — two interchangeable variants, selected by which fit_*
#  method the caller invokes:
#
#    fit_stage2()       HIERARCHICAL (Round 1): trained ONLY on rows
#                        the caller has pre-filtered to Vulnerable.
#                        Benign is isolated from the multi-class head;
#                        predict_hierarchical() re-injects "Benign"
#                        whenever Stage-1 gates a row out.
#
#    fit_stage2_flat()  FLAT (Round 3): trained on ALL rows passed in
#                        by the caller (Benign included), against the
#                        full Mtags label space. Benign becomes an
#                        ordinary class the multi-class head can
#                        predict directly. predict_flat_gated() still
#                        consults Stage-1 as a gate (Benign -> forced
#                        output), but for Vulnerable-routed rows it
#                        defers entirely to Stage-2's own prediction —
#                        which may legitimately come back as Benign.
#
#    Both variants apply SMOTE to the Stage-2 training partition only
#    (when enabled); in the flat variant Benign is eligible for SMOTE
#    oversampling on the same terms as every CWE class.
#    Both use early_stopping_rounds on a 10% internal validation split.
# ═══════════════════════════════════════════════════════════════════
class TwoStageXGBoost:
    """Sklearn-style two-stage XGBoost classifier.

    Supports two Stage-2 training/inference regimes:
      - hierarchical (fit_stage2 / predict_hierarchical) — Benign
        excluded from Stage-2 training, re-injected from Stage-1.
      - flat (fit_stage2_flat / predict_flat_gated) — Benign included
        in Stage-2 training as an ordinary class; Stage-1 still gates
        inference but Stage-2 may predict Benign on its own.
    """

    def __init__(self, n_multi_classes: int,
                 n_estimators: int, max_depth: int, lr: float,
                 subsample: float, colsample: float,
                 early_stop: int, device: str,
                 log_fn=None, check_stop_fn=None):
        self.n_multi = n_multi_classes
        self.log = log_fn or (lambda *a, **k: None)
        self.check_stop = check_stop_fn or (lambda: None)

        # Determine XGBoost tree_method
        tree_method = "hist"  # works on both CPU and GPU

        _base = dict(
            n_estimators          = n_estimators,
            max_depth             = max_depth,
            learning_rate         = lr,
            subsample             = subsample,
            colsample_bytree      = colsample,
            early_stopping_rounds = early_stop,
            tree_method           = tree_method,
            random_state          = SEED,
            n_jobs                = -1,
        )
        self.s1_params = {**_base,
                          "objective":   "binary:logistic",
                          "eval_metric": "logloss"}
        self.s2_params = {**_base,
                          "objective":   "multi:softprob",
                          "eval_metric": "mlogloss",
                          "num_class":   n_multi_classes}

        self.stage1: xgb.XGBClassifier | None = None
        self.stage2: xgb.XGBClassifier | None = None

        # Keep label encoders for hierarchical predict
        self._le_bin  = None   # set by caller
        self._le_multi = None  # set by caller
        self._benign_enc = None

    def fit_stage1(self, Xtr, ytr_bin, smote_params=None):
        """Train binary gate. Uses 10% internal val for early stopping.

        Fix C2: accepts optional smote_params — when supplied, SMOTE is
        applied to the binary training fold (90%) only; the 10% val fold
        is always kept real (pre-SMOTE) so the early-stopping signal is
        not contaminated by synthetic samples (also fixes C4 for Stage-1).

        Fix S4: scale_pos_weight is computed from the data actually passed
        to XGBoost's .fit() (after any SMOTE), not from upstream raw counts.
        """
        self.check_stop()

        # C4 FIX (Stage-1): split BEFORE SMOTE so val fold is real samples
        try:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_bin, test_size=0.1,
                random_state=SEED, stratify=ytr_bin)
        except ValueError:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_bin, test_size=0.1, random_state=SEED)

        # C2 FIX: apply SMOTE to 90% train fold when requested
        if smote_params:
            self.log("       Stage-1 SMOTE    : applying to 90% train fold "
                     "only (binary, val fold kept real) …")
            Xf, yf = smote_oversample(Xf_pre, yf_pre, **smote_params)
            self.log(f"       After SMOTE      : {len(Xf):,} samples")
        else:
            Xf, yf = Xf_pre, yf_pre

        # S4 FIX: compute spw from the data being fitted, not raw upstream data
        counts = np.bincount(yf.astype(int))
        neg    = int(counts[0]) if len(counts) > 0 else 1
        pos    = int(counts[1]) if len(counts) > 1 else 1
        spw    = neg / max(pos, 1)

        m = xgb.XGBClassifier(
            **{**self.s1_params, "scale_pos_weight": spw})
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        self.stage1 = m
        self.log(f"       Stage-1 best iter : {m.best_iteration}  "
                 f"(binary, spw={spw:.2f})")
        return self

    def fit_stage2(self, Xtr, ytr_multi, smote_params=None):
        """Train multi-class CWE head on Vulnerable rows only
        (hierarchical / Round-1 variant — Benign excluded by construction,
        since the caller pre-filters Xtr/ytr_multi to vuln_mask rows).

        Fix C4: the 10% internal validation split is carved from the
        ORIGINAL (pre-SMOTE) data so that XGBoost's early-stopping
        criterion is never evaluated against synthetic samples.
        """
        self.check_stop()

        # C4 FIX: split BEFORE SMOTE so the val fold is real samples only
        # Re-encode to 0-indexed for XGBoost multi-class
        le2 = LabelEncoder()
        # Fit encoder on pre-SMOTE labels to guarantee stable class mapping
        ytr_enc_pre = le2.fit_transform(ytr_multi)
        self._le2_internal = le2

        n_cls = len(np.unique(ytr_enc_pre))
        params = {**self.s2_params, "num_class": n_cls}

        try:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_enc_pre, test_size=0.1,
                random_state=SEED, stratify=ytr_enc_pre)
        except ValueError:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_enc_pre, test_size=0.1, random_state=SEED)

        # Apply SMOTE to the 90% training fold only (val fold untouched)
        if smote_params:
            self.log("       Stage-2 SMOTE    : applying to 90% train fold "
                     "only (val fold kept real — no synthetic leakage) …")
            Xf, yf = smote_oversample(Xf_pre, yf_pre, **smote_params)
            self.log(f"       After SMOTE      : {len(Xf):,} samples, "
                     f"min class = {np.bincount(yf.astype(int)).min()}")
        else:
            Xf, yf = Xf_pre, yf_pre

        m = xgb.XGBClassifier(**params)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        self.stage2 = m
        self.log(f"       Stage-2 best iter : {m.best_iteration}  "
                 f"({n_cls} CWE classes)")
        return self

    def fit_stage2_flat(self, Xtr, ytr_multi, smote_params=None):
        """Train multi-class head FLAT — on ALL rows, Benign included,
        directly against the full Mtags label space (Round-3 variant).

        Unlike fit_stage2(), the caller does NOT pre-filter to
        Vulnerable-only rows: Benign is one ordinary class among all
        Mtags here, and is eligible for SMOTE oversampling on the same
        terms as every CWE class when smote_params is supplied.

        Fix C4: the 10% internal validation split is carved from the
        ORIGINAL (pre-SMOTE) data so that XGBoost's early-stopping
        criterion is never evaluated against synthetic samples.
        """
        self.check_stop()

        # C4 FIX: split BEFORE SMOTE so the val fold is real samples only
        le2 = LabelEncoder()
        ytr_enc_pre = le2.fit_transform(ytr_multi)
        self._le2_internal = le2

        n_cls = len(np.unique(ytr_enc_pre))
        params = {**self.s2_params, "num_class": n_cls}

        try:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_enc_pre, test_size=0.1,
                random_state=SEED, stratify=ytr_enc_pre)
        except ValueError:
            Xf_pre, Xv, yf_pre, yv = train_test_split(
                Xtr, ytr_enc_pre, test_size=0.1, random_state=SEED)

        # Apply SMOTE to the 90% training fold only (val fold untouched)
        if smote_params:
            self.log("       Stage-2(flat) SMOTE : applying to 90% train fold "
                     "only (val fold kept real — Benign included) …")
            Xf, yf = smote_oversample(Xf_pre, yf_pre, **smote_params)
            self.log(f"       After SMOTE         : {len(Xf):,} samples, "
                     f"min class = {np.bincount(yf.astype(int)).min()}")
        else:
            Xf, yf = Xf_pre, yf_pre

        m = xgb.XGBClassifier(**params)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        self.stage2 = m
        self.log(f"       Stage-2(flat) best iter : {m.best_iteration}  "
                 f"({n_cls} classes, incl. Benign)")
        return self

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        assert self.stage1 is not None
        return self.stage1.predict(X)

    def predict_hierarchical(self, X: np.ndarray,
                              le_bin: LabelEncoder,
                              le_multi: LabelEncoder) -> np.ndarray:
        """Chain Stage-1 → Stage-2, return Mtags label array.

        Round-1 (hierarchical) variant: Stage-2 was trained on
        Vulnerable-only rows, so Benign is never in its label space —
        Benign is injected here whenever Stage-1 gates a row out.

        Fix S1: _le2_internal was fit on the pre-SMOTE vulnerable training
        labels (a subset of le_multi's classes), so inverse_transform
        always returns integers that are valid indices in le_multi's space.
        """
        bin_pred  = self.stage1.predict(X)   # 0=Benign, 1=Vulnerable
        vuln_mask = bin_pred == 1

        # Default: Benign label (as encoded integer in le_multi space)
        benign_enc = le_multi.transform(["Benign"])[0]
        out_enc    = np.full(len(X), benign_enc, dtype=int)

        if vuln_mask.sum() > 0:
            X_vuln   = X[vuln_mask]
            raw_pred = self.stage2.predict(X_vuln)         # internal 0-indexed
            # _le2_internal classes are a subset of le_multi classes,
            # so inverse_transform gives le_multi-compatible integers (S1 fix)
            orig_enc = self._le2_internal.inverse_transform(raw_pred)
            out_enc[vuln_mask] = orig_enc

        return out_enc   # integers in le_multi space

    def predict_flat_gated(self, X: np.ndarray,
                            le_bin: LabelEncoder,
                            le_multi: LabelEncoder) -> np.ndarray:
        """Chain Stage-1 → Stage-2(flat), return Mtags label array.

        Round-3 (flat) variant: Stage-2 was trained on ALL rows
        (Benign included) against the full Mtags space, so it CAN
        predict Benign on its own. Stage-1 still acts as a gate:
          - Stage-1 = Benign      -> output forced to "Benign"
          - Stage-1 = Vulnerable  -> output = Stage-2(flat)'s own
            prediction for that row (which may itself be any CWE
            class OR Benign, since Benign was never excluded from
            Stage-2's training distribution).
        """
        bin_pred  = self.stage1.predict(X)   # 0=Benign, 1=Vulnerable
        vuln_mask = bin_pred == 1

        benign_enc = le_multi.transform(["Benign"])[0]
        out_enc    = np.full(len(X), benign_enc, dtype=int)

        if vuln_mask.sum() > 0:
            X_vuln   = X[vuln_mask]
            raw_pred = self.stage2.predict(X_vuln)         # internal 0-indexed
            orig_enc = self._le2_internal.inverse_transform(raw_pred)
            out_enc[vuln_mask] = orig_enc

        return out_enc   # integers in le_multi space


# ═══════════════════════════════════════════════════════════════════
#  ML ENGINE  (background thread — queue-based communication)
# ═══════════════════════════════════════════════════════════════════
class MLEngine:
    def __init__(self, params: dict, log_queue: queue.Queue,
                 stop_event: threading.Event):
        self.p       = params
        self.q       = log_queue
        self.stop    = stop_event
        self.results = {}
        self.figures = []
        self.df_raw  = None
        self.df_dedup = None
        self.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = GraphCodeBERTEmbedder(self.device)

    def log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        self.q.put({"type": "log", "level": level,
                    "msg": f"[{ts}] {msg}"})

    def progress(self, pct, label=""):
        self.q.put({"type": "progress", "pct": pct, "label": label})

    def check_stop(self):
        if self.stop.is_set():
            raise InterruptedError("Stopped by user.")

    # ── feature-dataset export (Excel) ────────────────────────────
    def _export_cb_feature_dataset(self, snippets, le_m, le_b,
                                    y_m, y_b, csv_path,
                                    max_len, batch_size, device,
                                    pct_start=92, pct_end=95):
        """
        Extract CodeBERT [CLS] 768-dim embeddings for the full corpus and
        export them as '<base>_CodeBERT_Features.xlsx' alongside the source
        dataset.  This file can be reloaded via the 'Use CodeBERT
        precomputed features' browse button to skip CB extraction in
        future runs.

        Columns: cb_emb_0 … cb_emb_767  |  Mtags  |  Btags

        S2 fix: pct_start/pct_end are caller-supplied so this step's
        progress range never overlaps with the PDF-generation step that
        follows it, eliminating the backward progress-bar jump seen in V13.
        """
        try:
            import gc, openpyxl
            from transformers import AutoModel as _AM_CB
            from sklearn.preprocessing import normalize as _norm2

            self.log("━" * 58)
            self.log("💾  Exporting CodeBERT precomputed feature dataset …")
            self.progress(pct_start, "Exporting CB feature dataset …")
            self.check_stop()

            out_dir  = os.path.dirname(csv_path) or "."
            base     = os.path.splitext(os.path.basename(csv_path))[0]
            out_path = os.path.join(out_dir,
                                    f"{base}_CodeBERT_Features.xlsx")

            try:
                _tok = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
                _mod = _AM_CB.from_pretrained(CODEBERT_MODEL_NAME)
            except (OSError, EnvironmentError) as exc:
                local_name = CODEBERT_MODEL_NAME.replace("/", "_")
                msg = (
                    f"Cannot load '{CODEBERT_MODEL_NAME}' for CB feature "
                    f"export.\nCause: {exc}\n"
                    "Pre-download on an internet-connected machine, then "
                    f"copy './models/{local_name}' here and use that path. "
                    "Or set TRANSFORMERS_OFFLINE=1 if a local cache exists."
                )
                self.log(f"❌  CB export skipped — {msg}", "warn")
                return   # export is optional; do not abort the whole pipeline
            _mod.eval().to(device)
            self.log(f"   CB model loaded   : {CODEBERT_MODEL_NAME}")

            n   = len(snippets)
            emb = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
            _span = max(pct_end - pct_start, 1)
            for s in range(0, n, batch_size):
                self.check_stop()
                batch = list(snippets[s:s + batch_size])
                enc   = _tok(batch, padding="longest", truncation=True,
                             max_length=max_len, return_tensors="pt")
                enc   = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    h = _mod(**enc,
                              output_hidden_states=False,
                              output_attentions=False
                              ).last_hidden_state[:, 0, :]
                emb[s:s + len(batch)] = h.cpu().numpy()
                pct = pct_start + int(_span * min(s + batch_size, n) / max(n, 1))
                self.progress(pct,
                              f"CB export embedding "
                              f"{min(s+batch_size,n):,}/{n:,}")

            emb = _norm2(emb, norm="l2").astype(np.float32)
            del _mod
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Decode labels back to string for the export file
            mtags_str = le_m.inverse_transform(y_m).tolist()
            btags_str = le_b.inverse_transform(y_b).tolist()

            cb_cols = [f"cb_emb_{i}" for i in range(EMBEDDING_DIM)]
            wb = openpyxl.Workbook(write_only=True)
            ws = wb.create_sheet("CB_Features")
            ws.append(cb_cols + ["Mtags", "Btags"])
            chunk = 2000
            for i in range(n):
                ws.append(emb[i].tolist() + [mtags_str[i], btags_str[i]])
                if (i + 1) % chunk == 0:
                    self.check_stop()

            wb.save(out_path)
            self.results["cb_feature_dataset_path"] = out_path
            self.log(f"   CB feature dataset: {out_path}  "
                     f"({EMBEDDING_DIM} dims × {n:,} rows)")

        except InterruptedError:
            raise
        except Exception as exc:
            import traceback
            self.log(f"⚠️  CB feature export failed (pipeline continues): "
                     f"{exc}\n{traceback.format_exc()}", "warn")

    def _export_feature_dataset(self, df, X_all, csv_path):
        """
        Save the concatenated (768-dim GraphCodeBERT + 12-dim
        handcrafted) feature matrix together with Mtags/Btags labels
        and a SHA-256 snippet_hash to an .xlsx file.

        S2 fix: the snippet_hash column (SHA-256 of the stripped SNIPPET
        text) is written alongside the features so that future loads with
        the precomputed-dataset bypass can deduplicate on the original
        code text hash rather than on inexact float-vector comparison.
        """
        import hashlib
        try:
            self.progress(32, "Exporting feature dataset (.xlsx) …")
            self.check_stop()
            t0 = time.time()

            out_dir  = os.path.dirname(csv_path) or "."
            base     = os.path.splitext(os.path.basename(csv_path))[0]
            out_path = os.path.join(
                out_dir, f"{base}_GraphCodeBERT_Features.xlsx")

            mtags    = df["Mtags"].astype(str).tolist()
            btags    = df["Btags"].astype(str).tolist()
            snippets = df["SNIPPET"].astype(str).tolist() \
                       if "SNIPPET" in df.columns else [""] * len(df)
            n = X_all.shape[0]

            # Pre-compute SHA-256 hashes for S2 fix
            hashes = [
                hashlib.sha256(s.strip().encode("utf-8", errors="replace")
                               ).hexdigest()
                for s in snippets
            ]

            wb = openpyxl.Workbook(write_only=True)
            ws = wb.create_sheet("Features")
            ws.append(["snippet_hash"] + EMB_COLS + HC_FEATURE_NAMES
                      + ["Mtags", "Btags"])

            chunk = 2000
            for i in range(n):
                row = [hashes[i]] + X_all[i].tolist()
                row.append(mtags[i])
                row.append(btags[i])
                ws.append(row)
                if (i + 1) % chunk == 0:
                    self.check_stop()
                    pct = 32 + int(3 * (i + 1) / max(n, 1))
                    self.progress(
                        pct,
                        f"Exporting feature dataset … {i+1:,}/{n:,}")

            wb.save(out_path)
            self.results["feature_dataset_path"] = out_path
            self.log(f"   Feature dataset   : {out_path}  "
                     f"({X_all.shape[1]} dims × {n:,} rows, "
                     f"snippet_hash column included — S2 fix, "
                     f"{time.time()-t0:.1f}s)")
        except InterruptedError:
            raise
        except Exception as exc:
            self.log(f"⚠️  Feature-dataset export failed (continuing "
                     f"pipeline without it): {exc}", "warn")

    # ── figure helpers (identical to original) ──────────────────
    def _bar_chart(self, title, labels, values, colours,
                   xlabel="", ylabel="Value"):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1E2233")
        ax.set_facecolor("#252840")
        bars = ax.bar(labels, values, color=colours, edgecolor="#4A4F7A",
                      linewidth=0.6, width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:,.0f}" if val > 1 else f"{val:.4f}",
                    ha="center", va="bottom",
                    color="#E8EAF6", fontsize=8, fontweight="bold")
        ax.set_title(title, color="#E8EAF6", fontsize=11,
                     fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color="#8892B0", fontsize=9)
        ax.set_ylabel(ylabel, color="#8892B0", fontsize=9)
        ax.tick_params(colors="#8892B0", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#3A3F5C")
        ax.yaxis.grid(True, color="#3A3F5C", linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        return fig

    def _metric_bar(self, title, configs, metrics_dict):
        metric_names = ["Accuracy", "MCC", "mF1 (macro)", "wF1 (weighted)"]
        x     = np.arange(len(metric_names))
        width = 0.8 / max(len(configs), 1)
        # M1 fix: extended to 12 distinct colors (was 6) — the "All
        # configurations overview" chart can plot up to 10 simultaneous
        # result keys (R1A/B, R2A/B, R3 x2, R4A x2, R4B x2); a 6-color
        # palette caused entries 7-10 to silently duplicate entries 1-4.
        palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER,
                   ACCENT_TEAL, ACCENT_RED, "#B98AE6",
                   "#16A085", "#2980B9", "#E67E22", "#8E44AD",
                   "#27AE60", "#C0392B"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#1E2233")
        ax.set_facecolor("#252840")
        for k, (cfg_name, key) in enumerate(configs):
            if key not in metrics_dict:
                continue
            m = metrics_dict[key]
            vals = [m["accuracy"], m["f1_micro"],
                    m["f1_macro"],  m["f1_weighted"]]
            offset = (k - len(configs) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.92,
                          label=cfg_name,
                          color=palette[k % len(palette)],
                          edgecolor="#1A1D2E", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom",
                        color="#E8EAF6", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, color="#8892B0", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, color="#E8EAF6", fontsize=11,
                     fontweight="bold", pad=10)
        ax.tick_params(colors="#8892B0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#3A3F5C")
        ax.yaxis.grid(True, color="#3A3F5C", linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, facecolor="#2E3250",
                  edgecolor="#3A3F5C", labelcolor="#E8EAF6")
        plt.tight_layout()
        return fig

    # ── main entry ──────────────────────────────────────────────
    def run(self):
        try:
            self._run_pipeline()
            self.q.put({"type": "done", "results": self.results,
                        "figures": self.figures})
        except InterruptedError:
            self.log("⛔  Process stopped by user.", "warn")
            self.q.put({"type": "stopped"})
        except Exception as exc:
            import traceback
            self.log(f"❌  Error: {exc}", "error")
            self.log(traceback.format_exc(), "error")
            self.q.put({"type": "error", "msg": str(exc)})

    def _run_pipeline(self):
        p = self.p
        use_pre    = bool(p.get("use_precomputed_features", False))
        use_cb_pre = bool(p.get("use_cb_precomputed_features", False))
        gcb_pre_path = p.get("gcb_pre_path", "").strip()
        cb_pre_path  = p.get("cb_pre_path",  "").strip()
        export_cb_pre  = bool(p.get("export_cb_pre",  False))  # M2 fix: default off
        export_gcb_pre = bool(p.get("export_gcb_pre", True))

        # If the user supplied a separate GCB precomputed path, use it
        # as the primary data source regardless of the csv_path field.
        if use_pre and gcb_pre_path:
            _load_path = gcb_pre_path
            self.log(f"   GCB precomputed   : {_load_path}")
        else:
            _load_path = p["csv_path"]

        # ── 1. LOAD ─────────────────────────────────────────────
        self.log("━" * 58)
        self.check_stop()

        if use_pre:
            self.log("📂  STEP 1 — Loading precomputed feature dataset …")
            self.progress(2, "Loading precomputed feature dataset …")
            fpath = _load_path
            if fpath.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(fpath)
            else:
                df = pd.read_csv(fpath)

            required_cols = EMB_COLS + HC_FEATURE_NAMES + ["Mtags", "Btags"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                preview = ", ".join(missing[:6]) + (
                    " …" if len(missing) > 6 else "")
                raise ValueError(
                    "Precomputed feature file is missing required "
                    f"columns: {preview}. Expected the "
                    "'<dataset>_GraphCodeBERT_Features.xlsx' format "
                    "produced by this framework's Step-3 export "
                    f"({EMBEDDING_DIM} 'gcb_emb_*' + {HC_DIM} 'hc_*' "
                    "columns + Mtags + Btags).")
        else:
            self.log("📂  STEP 1 — Loading dataset …")
            self.progress(2, "Loading CSV …")
            df = pd.read_csv(_load_path)

        self.df_raw = df.copy()
        n_raw = len(df)
        self.log(f"   Raw rows          : {n_raw:,}")
        self.results["n_raw"] = n_raw

        btag_raw    = df["Btags"].value_counts().to_dict()
        mtag_raw_n  = df["Mtags"].nunique()
        self.results["btag_raw"]   = btag_raw
        self.results["mtag_raw_n"] = mtag_raw_n
        self.log(f"   Binary dist (raw) : {btag_raw}")
        self.log(f"   Multi-classes(raw): {mtag_raw_n}")

        # ── 2. DEDUP ────────────────────────────────────────────
        self.log("━" * 58)
        self.log("🔄  STEP 2 — Deduplication …")
        self.progress(6, "Deduplicating …")
        self.check_stop()

        if use_pre:
            # S2 FIX: deduplicate on a hash of the original SNIPPET rather
            # than on the float embedding columns. Two semantically identical
            # snippets that differ by a single whitespace character will have
            # marginally different float embeddings and survive float-vector
            # dedup, allowing near-duplicate train/test contamination.
            # The exported feature dataset always carries a 'snippet_hash'
            # column (SHA-256 of the stripped SNIPPET) written during Step 3
            # export. If it is present, use it; fall back to full-vector
            # dedup with a logged warning if the column is absent (e.g. files
            # exported by V05 or earlier).
            if "snippet_hash" in df.columns:
                df = (df.drop_duplicates(subset=["snippet_hash"])
                        .dropna(subset=["Mtags", "Btags"])
                        .reset_index(drop=True))
                self.log("   Dedup method      : snippet_hash (S2 fix — "
                         "exact-hash dedup on original code text)")
            else:
                self.log("   ⚠  'snippet_hash' column absent in precomputed "
                         "file (V05 or earlier export). Falling back to "
                         "float-vector dedup — re-export dataset with V06 "
                         "to get hash-based dedup (S2 fix).")
                df = (df.drop_duplicates(subset=EMB_COLS + HC_FEATURE_NAMES)
                        .dropna(subset=["Mtags", "Btags"])
                        .reset_index(drop=True))
        else:
            df = (df.drop_duplicates(subset=["SNIPPET"])
                    .dropna(subset=["SNIPPET", "Mtags", "Btags"])
                    .reset_index(drop=True))
        self.df_dedup = df.copy()
        n_dedup = len(df)
        self.results["n_dedup"] = n_dedup
        self.log(f"   After dedup       : {n_dedup:,}  "
                 f"(removed {n_raw - n_dedup:,})")

        btag_dedup  = df["Btags"].value_counts().to_dict()
        mtag_dedup_n = df["Mtags"].nunique()
        self.results["btag_dedup"]   = btag_dedup
        self.results["mtag_dedup_n"] = mtag_dedup_n
        self.log(f"   Binary dist(dedup): {btag_dedup}")
        self.log(f"   Multi-class count : {mtag_dedup_n}")

        # Sample cap
        n_samples = int(p["n_samples"])
        shuffle   = bool(p.get("shuffle", True))
        if n_samples < n_dedup:
            if shuffle:
                df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
                self.log(f"   Working sample    : {len(df):,} "
                         f"(shuffled, random_state={SEED})")
            else:
                df = df.head(n_samples).reset_index(drop=True)
                self.log(f"   Working sample    : {len(df):,} "
                         f"(first {n_samples:,} rows, file order)")
        self.results["n_working"] = len(df)

        btag_work  = df["Btags"].value_counts().to_dict()
        mtag_work_n = df["Mtags"].nunique()
        self.results["btag_working"]  = btag_work
        self.results["mtag_working_n"] = mtag_work_n
        self.log(f"   Binary dist(work) : {btag_work}")

        # Class distribution chart
        fig_dist = self._bar_chart(
            "Binary class distribution — working set",
            list(btag_work.keys()),
            list(btag_work.values()),
            [ACCENT_GREEN if k == "Benign" else ACCENT_RED
             for k in btag_work.keys()],
            ylabel="Sample count"
        )
        self.figures.append(("Binary class distribution", fig_dist))

        # ── STEP 3 — SPLIT DATASET (80% train / 20% test) ──────
        self.log("━" * 58)
        self.log("✂️   STEP 3 — Splitting dataset: 80% training / 20% testing …")
        self.progress(10, "Splitting dataset 80/20 …")
        self.check_stop()

        # Label encode on full dataset before any split
        le_m = LabelEncoder()
        le_b = LabelEncoder()
        y_m  = le_m.fit_transform(df["Mtags"])
        y_b  = le_b.fit_transform(df["Btags"])
        self.results["classes_multi"]  = list(le_m.classes_)
        self.results["classes_binary"] = list(le_b.classes_)
        n_multi = len(le_m.classes_)
        n_bin   = len(le_b.classes_)
        self.results["n_multi_classes"] = n_multi
        self.log(f"   Multi-classes     : {n_multi}")
        self.log(f"   Binary classes    : {list(le_b.classes_)}")

        _test_size_global = float(p.get("test_size", 0.2))
        _shuf_global      = bool(p.get("shuffle", True))

        def _make_shared_split(y_b_arr, y_m_arr, n_total):
            """C1 fix: stratify on Mtag; C3 fix: one split for all rounds."""
            idx = np.arange(n_total)
            if _shuf_global:
                try:
                    idx_tr, idx_te = train_test_split(
                        idx, test_size=_test_size_global,
                        random_state=SEED, stratify=y_m_arr)
                    desc = (f"shuffled, stratified across "
                            f"{n_multi} Mtag classes (C1 fix)")
                except ValueError:
                    try:
                        idx_tr, idx_te = train_test_split(
                            idx, test_size=_test_size_global,
                            random_state=SEED, stratify=y_b_arr)
                        desc = ("shuffled, stratified on Btag "
                                "(Mtag fallback — some CWE classes <2 samples)")
                    except ValueError:
                        idx_tr, idx_te = train_test_split(
                            idx, test_size=_test_size_global,
                            random_state=SEED)
                        desc = "shuffled, unstratified (very small dataset)"
            else:
                n_tr   = int(n_total * (1 - _test_size_global))
                idx_tr = idx[:n_tr]
                idx_te = idx[n_tr:]
                desc   = "sequential, file order preserved"
            return idx_tr, idx_te, desc

        idx_tr_shared, idx_te_shared, split_desc_shared = \
            _make_shared_split(y_b, y_m, len(df))

        train_pct_g = (1 - _test_size_global) * 100
        test_pct_g  = _test_size_global * 100
        self.log(f"   Total samples     : {len(df):,}")
        self.log(f"   Train partition   : {len(idx_tr_shared):,} "
                 f"({train_pct_g:.0f}%)")
        self.log(f"   Test  partition   : {len(idx_te_shared):,} "
                 f"({test_pct_g:.0f}%)")
        self.log(f"   Split strategy    : {split_desc_shared}")
        self.results["split_desc"] = split_desc_shared
        self.results["n_train"]    = int(len(idx_tr_shared))
        self.results["n_test"]     = int(len(idx_te_shared))

        # Shared label arrays (used by all four rounds)
        y_b_tr_shared = y_b[idx_tr_shared]
        y_b_te_shared = y_b[idx_te_shared]
        y_m_tr_shared = y_m[idx_tr_shared]
        y_m_te_shared = y_m[idx_te_shared]

        # Raw snippets (used by fine-tuned rounds R1 / R2).
        # S1 fix: train/test slicing is performed exactly once, via NumPy
        # object-array indexing, in the STEP 4 block immediately below
        # (the M2-fix version). The previous list-comprehension slicing
        # that used to live here was dead code — silently overwritten by
        # the NumPy version on every run — and has been removed.
        snippets_all = df["SNIPPET"].astype(str).tolist() \
                       if "SNIPPET" in df.columns else None

        # ── STEP 4 — GraphCodeBERT embedding extraction ──────────
        # NOTE: CodeBERT fine-tuned rounds (R1) load the CB model internally
        # inside run_bert_round() — no separate CB extraction step is needed.
        # Static pre-extracted CB embeddings cannot be used by fine-tuned rounds
        # because backbone weights change each epoch. Step 4 therefore runs the
        # GCB frozen extractor only, which supplies data to Rounds 2, 3, and 4.
        #
        # C1 annotation: the embedder uses pre-trained frozen LayerNorm
        # (no fitted BatchNorm), so extracting over the full corpus before
        # slicing introduces NO cross-sample statistical leakage. However,
        # any future addition of a fitted scaler (e.g. StandardScaler) MUST
        # be fitted on X_all[idx_tr_shared] ONLY and applied to both partitions.
        #
        # S3 guard: Step 4 is skipped entirely when only Round 1 (CB fine-tune)
        # is enabled and R2/R3/R4 are all disabled — no GCB work needed.
        # M2 fix: snippet slices use NumPy object-array indexing for speed.
        # M3 fix: guard against snippets_all=None in live-extraction branch.

        max_len     = int(p.get("max_len", FT_MAX_LEN_DEFAULT))
        embed_batch = int(p.get("embed_batch", FT_BATCH_DEFAULT))
        pooling     = p.get("pooling", "cls")

        # ── Round flags — must be defined before _need_gcb check below ──
        # (Bug 1 fix: these were previously assigned ~130 lines later, after
        # the point where _need_gcb first references them — causing
        # UnboundLocalError at runtime on first access.)
        do_r1_ft   = p.get("do_r1_ft",  True)   # Round 1: CodeBERT fine-tuned
        do_r2_ft   = p.get("do_r2_ft",  True)   # Round 2: GCB fine-tuned
        do_r3_flat = p.get("do_flat",   False)   # Round 3: GCB+XGBoost flat
        do_r4_hier = p.get("do_binary", True)    # Round 4: two-stage hierarchical
        do_smote   = p.get("do_smote",  True)
        do_no_sm   = p.get("do_no_smote", True)

        # M2 fix: convert to NumPy object array for vectorised index slicing
        # (S1 fix: this is now the ONLY place snips_tr/snips_te are computed)
        if snippets_all:
            snips_arr = np.array(snippets_all, dtype=object)
            snips_tr  = snips_arr[idx_tr_shared].tolist()
            snips_te  = snips_arr[idx_te_shared].tolist()
            del snips_arr   # release duplicate copy
        else:
            snips_tr = snips_te = None

        _need_gcb = do_r2_ft or do_r3_flat or do_r4_hier

        if _need_gcb:
            self.log("━" * 58)
            self.log("🧠  STEP 4 — GraphCodeBERT embedding extraction …")
            self.log(f"   Model    : {GRAPHCODEBERT_MODEL_NAME}")
            self.log("   Scope    : full corpus → sliced by shared split indices")
            self.log("   Rounds   : R2 (fine-tune), R3 (flat 768-dim), "
                     "R4 (hierarchical 780-dim)")
            self.progress(14, f"Loading {GRAPHCODEBERT_MODEL_NAME} …")
            self.check_stop()

            if use_pre:
                self.log("   ⏭️  GCB embeddings loaded from precomputed file.")
                t0    = time.time()
                X_emb = df[EMB_COLS].to_numpy(dtype=np.float32)
                X_hc  = df[HC_FEATURE_NAMES].to_numpy(dtype=np.float32)
                X_all = np.hstack([X_emb, X_hc])
                X_all = normalize(X_all, norm="l2").astype(np.float32)
                self.log(f"   GCB feature matrix: {X_all.shape}  "
                         f"[loaded, L2-normalised] ({time.time()-t0:.1f}s)")
                self.results["embedding_model"] = "(precomputed)"
                self.results["max_len"]         = p.get("max_len", "N/A")
            else:
                # M3 fix: guard against missing SNIPPET column
                if snippets_all is None:
                    raise ValueError(
                        "STEP 4: GraphCodeBERT embedding requires a 'SNIPPET' "
                        "column in the dataset. Either provide a CSV/XLSX with "
                        "a SNIPPET column, or enable 'Use precomputed features' "
                        "and point to a pre-extracted feature file.")

                file_types = df["FILE TYPE"].astype(str).tolist() \
                             if "FILE TYPE" in df.columns \
                             else ["unknown"] * len(df)
                self.embedder.load(log_fn=self.log)
                self.progress(16, "GraphCodeBERT embedding (train + test) …")
                t0    = time.time()
                X_emb = self.embedder.embed(
                    snippets_all, max_len=max_len,
                    batch_size=embed_batch, pooling=pooling,
                    progress_fn=self.progress, check_stop_fn=self.check_stop,
                    pct_start=16, pct_end=26)
                self.log(f"   GCB embedding done: {X_emb.shape}  "
                         f"(max_len={max_len}, pooling={pooling}, "
                         f"{time.time()-t0:.1f}s)")

                # M1 fix: log VRAM after GCB extraction before HC step
                if self.device.type == "cuda":
                    _used  = torch.cuda.memory_allocated() / 1e9
                    _total = torch.cuda.get_device_properties(
                        self.device).total_memory / 1e9
                    self.log(f"   VRAM after GCB    : {_used:.2f} GB / "
                             f"{_total:.2f} GB  "
                             f"({100*_used/_total:.0f}%)")

                # Handcrafted features (12-dim) — needed for Round 4 only
                self.progress(26, "Extracting handcrafted features (12-dim) …")
                self.check_stop()
                t0    = time.time()
                X_hc  = extract_handcrafted(snippets_all, file_types)
                X_all = np.hstack([X_emb, X_hc])   # (N, 780)
                X_all = normalize(X_all, norm="l2").astype(np.float32)
                self.log(f"   GCB feature matrix: {X_all.shape}  "
                         f"[GCB 768 + HC 12 = 780 dims] ({time.time()-t0:.1f}s)")
                self.results["embedding_model"] = GRAPHCODEBERT_MODEL_NAME
                self.results["max_len"]         = max_len
                # Export GCB feature dataset if requested
                if export_gcb_pre:
                    self._export_feature_dataset(df, X_all, p["csv_path"])
                else:
                    self.log("   ⏭️  GCB feature export skipped "
                             "(unchecked in Dataset card)")

            self.results["embedding_shape"] = list(X_emb.shape)
            self.results["feature_shape"]   = list(X_all.shape)

            # Slice GCB matrices using shared split indices
            Xtr_shared     = X_all[idx_tr_shared]   # 780-dim → Round 4
            Xte_shared     = X_all[idx_te_shared]
            Xtr_emb_shared = X_emb[idx_tr_shared]   # 768-dim → Round 3
            Xte_emb_shared = X_emb[idx_te_shared]

            self.log(f"   GCB 780-dim train : {Xtr_shared.shape}  [→ Round 4]")
            self.log(f"   GCB 780-dim test  : {Xte_shared.shape}  [→ Round 4]")
            self.log(f"   GCB 768-dim train : {Xtr_emb_shared.shape}  [→ Round 3]")
            self.log(f"   GCB 768-dim test  : {Xte_emb_shared.shape}  [→ Round 3]")
        else:
            # Only Round 1 (CB fine-tuned) selected — GCB extraction not needed
            self.log("   ⏭️  STEP 4 skipped — GCB embeddings not required "
                     "(only Round 1 CB fine-tune selected)")
            # Provide empty placeholders so downstream round guards work safely
            Xtr_shared = Xte_shared = None
            Xtr_emb_shared = Xte_emb_shared = None
            X_emb = X_all = None

        # ── STEP 5 — CodeBERT fine-tuning data note ──────────────
        # Round 1 (CodeBERT fine-tuned) loads the CB model internally
        # inside run_bert_round() from raw text (snips_tr / snips_te).
        # No separate CB embedding extraction step is needed or correct:
        # fine-tuning updates backbone weights so any pre-extracted
        # static embeddings would be stale after the first gradient step.
        # snips_tr / snips_te are already prepared in Step 3 above.
        if do_r1_ft or do_r2_ft:
            if snips_tr is None:
                self.log("   ⚠  'SNIPPET' column not found — "
                         "Rounds 1/2 (fine-tuned) require raw code text. "
                         "Skipping R1 and R2.", "warn")
                do_r1_ft = do_r2_ft = False
            else:
                self.log("━" * 58)
                self.log("📝  STEP 5 — Code snippet data prepared for "
                         "fine-tuned Rounds 1 & 2")
                self.log(f"   Train snippets : {len(snips_tr):,}  "
                         f"(raw text → tokenised inside run_bert_round)")
                self.log(f"   Test  snippets : {len(snips_te):,}")
                self.log("   CB model (R1)  : loaded on-demand inside each "
                         "run_bert_round() call")
                self.log("   GCB model (R2) : loaded on-demand inside each "
                         "run_bert_round() call")
                self.progress(32, "Snippet data ready for fine-tuned rounds …")

        # Accumulator for test-set prediction tracking
        _round_preds: dict = {}


        n_est     = int(p.get("n_estimators", 400))
        max_depth = int(p.get("max_depth", 7))
        xgb_lr    = float(p.get("xgb_lr", 0.05))
        subsample = float(p.get("subsample", 0.8))
        colsample = float(p.get("colsample", 0.8))
        early_stop = int(p.get("early_stop", 30))

        # ────────────────────────────────────────────────────────
        #  run_round: uses the pre-computed shared split (C3 fix).
        #  C2 fix: SMOTE applied to BOTH Stage-1 and Stage-2 training
        #          folds when smote=True.
        #  S5 fix: F1-micro removed (= accuracy for single-label tasks);
        #          replaced with Matthews Correlation Coefficient (MCC).
        # ────────────────────────────────────────────────────────
        from sklearn.metrics import matthews_corrcoef

        def _dist_named(y_arr, label_names):
            d = dict(zip(*np.unique(y_arr, return_counts=True)))
            return {label_names[k]: int(v) for k, v in d.items()
                    if k < len(label_names)}

        def run_round(Xtr, Xte, y_b_tr, y_b_te, y_m_tr, y_m_te,
                      tag, label_names_m, label_names_b,
                      smote=False,
                      pct_start=35, pct_end=60):
            """
            Train the two-stage XGBoost on the pre-computed shared split
            and return metrics.

            C2 fix: when smote=True, SMOTE is passed to fit_stage1() AND
                    fit_stage2() so the binary gate is also oversampled.
            C3 fix: the caller passes the shared split; no new split is
                    generated inside this function.
            S5 fix: MCC replaces F1-micro (which equals accuracy for
                    single-label classification — a redundant metric).
            """
            self.check_stop()
            smote_tag = "with SMOTE" if smote else "no SMOTE"
            self.log(f"\n   ── {tag} [{smote_tag}] ──")
            self.progress(pct_start, f"{tag} – using shared split …")

            train_pct = (1 - _test_size_global) * 100
            test_pct  = _test_size_global * 100

            tr_bin_named = _dist_named(y_b_tr, label_names_b)
            te_bin_named = _dist_named(y_b_te, label_names_b)

            self.log(f"     Dataset division : {len(Xtr)+len(Xte):,} total "
                     f"-> {len(Xtr):,} train ({train_pct:.0f}%) / "
                     f"{len(Xte):,} test ({test_pct:.0f}%), "
                     f"{split_desc_shared}")
            self.log(f"     Train dist       : {tr_bin_named}")
            self.log(f"     Test dist        : {te_bin_named}")

            n_tr_benign = tr_bin_named.get("Benign", 0)
            n_tr_vuln   = sum(v for k, v in tr_bin_named.items()
                              if k != "Benign")
            n_te_benign = te_bin_named.get("Benign", 0)
            n_te_vuln   = sum(v for k, v in te_bin_named.items()
                              if k != "Benign")

            # C2 fix: build smote_params used for BOTH Stage-1 and Stage-2
            smote_params = None
            if smote:
                smote_params = {
                    "k":   int(p.get("smote_k", 5)),
                    "cap": int(p.get("smote_cap", 1200)),
                    "tgt": p.get("smote_target", "median"),
                }

            # ── Build two-stage classifier ──────────────────────
            self.progress(pct_start + 4, f"{tag} – Stage-1 training …")
            self.check_stop()

            clf = TwoStageXGBoost(
                n_multi_classes = n_multi,
                n_estimators    = n_est,
                max_depth       = max_depth,
                lr              = xgb_lr,
                subsample       = subsample,
                colsample       = colsample,
                early_stop      = early_stop,
                device          = str(self.device),
                log_fn          = self.log,
                check_stop_fn   = self.check_stop,
            )

            # Stage 1: binary — C2 fix: pass smote_params
            t0 = time.time()
            self.log("     Stage 1         : Binary (Benign / Vulnerable)"
                     + (" + SMOTE" if smote else ""))
            clf.fit_stage1(Xtr, y_b_tr, smote_params=smote_params)
            s1_time = time.time() - t0
            self.log(f"     Stage-1 time    : {s1_time:.1f}s")

            # Stage 2: multi-class (vulnerable rows only)
            self.progress(pct_start + 8, f"{tag} – Stage-2 training …")
            self.check_stop()

            vuln_mask_tr = y_b_tr == 1   # 1 = Vulnerable
            Xtr_vuln    = Xtr[vuln_mask_tr]
            y_m_tr_vuln = y_m_tr[vuln_mask_tr]
            n_vuln_tr   = vuln_mask_tr.sum()
            self.log(f"     Stage 2         : Multi-class CWE  "
                     f"({n_multi} classes, {n_vuln_tr:,} vuln. train rows)")

            t0 = time.time()
            clf.fit_stage2(Xtr_vuln, y_m_tr_vuln,
                           smote_params=smote_params)
            s2_time = time.time() - t0
            train_time = s1_time + s2_time
            self.log(f"     Stage-2 time    : {s2_time:.1f}s  "
                     f"| total {train_time:.1f}s")

            # ── Evaluate ────────────────────────────────────────
            self.progress(pct_end - 2, f"{tag} – evaluating …")
            self.check_stop()

            yp_enc = clf.predict_hierarchical(Xte, le_b, le_m)
            yp_bin = clf.predict_binary(Xte)

            # Binary metrics (Stage-1 only)
            acc_b      = accuracy_score(y_b_te, yp_bin)
            f1_b_macro = f1_score(y_b_te, yp_bin,
                                  average="macro", zero_division=0)

            # Multi-class metrics — S5 fix: MCC replaces F1-micro
            acc  = accuracy_score(y_m_te, yp_enc)
            mcc  = matthews_corrcoef(y_m_te, yp_enc)
            f1M  = f1_score(y_m_te, yp_enc,
                            average="macro", zero_division=0)
            f1w  = f1_score(y_m_te, yp_enc,
                            average="weighted", zero_division=0)

            self.log(f"     Stage-1 bin acc  : {acc_b:.4f}  "
                     f"(mF1={f1_b_macro:.4f})")
            self.log(f"     Hierarchical acc : {acc:.4f}  ({acc*100:.2f}%)")
            self.log(f"     MCC              : {mcc:.4f}  "
                     f"[replaces F1-micro — S5 fix]")
            self.log(f"     F1-macro (mF1)   : {f1M:.4f}")
            self.log(f"     F1-weighted(wF1) : {f1w:.4f}")

            rep = classification_report(
                y_m_te, yp_enc,
                target_names=label_names_m,
                output_dict=True, zero_division=0)

            cm_mat = confusion_matrix(
                y_m_te, yp_enc, labels=list(range(n_multi)))

            tr_m_dist = _dist_named(y_m_tr, label_names_m)
            te_m_dist = _dist_named(y_m_te, label_names_m)

            self.progress(pct_end, f"{tag} – done")
            return {
                "accuracy":        round(acc, 4),
                "f1_micro":        round(mcc, 4),   # slot kept; now holds MCC
                "mcc":             round(mcc, 4),
                "f1_macro":        round(f1M, 4),
                "f1_weighted":     round(f1w, 4),
                "s1_accuracy":     round(acc_b, 4),
                "s1_f1_macro":     round(f1_b_macro, 4),
                "train_time":      round(train_time, 1),
                "n_train":         len(Xtr),
                "n_test":          len(Xte),
                "n_train_benign":  int(n_tr_benign),
                "n_train_vuln":    int(n_tr_vuln),
                "n_test_benign":   int(n_te_benign),
                "n_test_vuln":     int(n_te_vuln),
                "train_dist":      tr_m_dist,
                "test_dist":       te_m_dist,
                "train_dist_smote": None,
                "n_classes":       n_multi,
                "smote":           smote,
                "report":          rep,
                "confusion_matrix": cm_mat,
                "cm_labels":       label_names_m,
                "tag":             tag,
                "smote_tag":       smote_tag,
            }

        def run_round_flat(Xtr, Xte, y_b_tr, y_b_te, y_m_tr, y_m_te,
                            tag, label_names_m, label_names_b,
                            smote=False,
                            pct_start=35, pct_end=60):
            """
            FLAT variant (Round 3): both Stage-1 and Stage-2 operate on
            the 768-dim GraphCodeBERT embedding (handcrafted features
            excluded). Uses the pre-computed shared split (C3 fix) sliced
            to X_emb columns, guaranteeing the same held-out rows as R1/R2.

            C2 fix: SMOTE passed to fit_stage1() AND fit_stage2_flat().
            S5 fix: MCC replaces F1-micro.
            """
            self.check_stop()
            smote_tag = "with SMOTE" if smote else "no SMOTE"
            self.log(f"\n   ── {tag} [{smote_tag}] ──")
            self.progress(pct_start, f"{tag} – using shared split …")

            train_pct = (1 - _test_size_global) * 100
            test_pct  = _test_size_global * 100

            tr_bin_named = _dist_named(y_b_tr, label_names_b)
            te_bin_named = _dist_named(y_b_te, label_names_b)

            self.log(f"     Dataset division : {len(Xtr)+len(Xte):,} total "
                     f"-> {len(Xtr):,} train ({train_pct:.0f}%) / "
                     f"{len(Xte):,} test ({test_pct:.0f}%), "
                     f"{split_desc_shared}")
            self.log(f"     Train dist       : {tr_bin_named}")
            self.log(f"     Test dist        : {te_bin_named}")

            n_tr_benign = tr_bin_named.get("Benign", 0)
            n_tr_vuln   = sum(v for k, v in tr_bin_named.items()
                              if k != "Benign")
            n_te_benign = te_bin_named.get("Benign", 0)
            n_te_vuln   = sum(v for k, v in te_bin_named.items()
                              if k != "Benign")

            # C2 fix: smote_params for both stages
            smote_params = None
            if smote:
                smote_params = {
                    "k":   int(p.get("smote_k", 5)),
                    "cap": int(p.get("smote_cap", 1200)),
                    "tgt": p.get("smote_target", "median"),
                }

            # ── Build two-stage classifier (flat Stage-2) ───────
            self.progress(pct_start + 4, f"{tag} – Stage-1 training …")
            self.check_stop()

            clf = TwoStageXGBoost(
                n_multi_classes = n_multi,
                n_estimators    = n_est,
                max_depth       = max_depth,
                lr              = xgb_lr,
                subsample       = subsample,
                colsample       = colsample,
                early_stop      = early_stop,
                device          = str(self.device),
                log_fn          = self.log,
                check_stop_fn   = self.check_stop,
            )

            # Stage 1: binary gate — C2 fix: pass smote_params
            t0 = time.time()
            self.log("     Stage 1         : Binary (Benign / Vulnerable)"
                     + (" + SMOTE" if smote else ""))
            clf.fit_stage1(Xtr, y_b_tr, smote_params=smote_params)
            s1_time = time.time() - t0
            self.log(f"     Stage-1 time    : {s1_time:.1f}s")

            # Stage 2 (FLAT): multi-class on ALL rows, Benign included
            self.progress(pct_start + 8, f"{tag} – Stage-2(flat) training …")
            self.check_stop()

            self.log(f"     Stage 2 (flat)  : Multi-class, ALL rows "
                     f"({n_multi} classes incl. Benign, "
                     f"{len(Xtr):,} train rows)")

            t0 = time.time()
            clf.fit_stage2_flat(Xtr, y_m_tr,
                                 smote_params=smote_params)
            s2_time = time.time() - t0
            train_time = s1_time + s2_time
            self.log(f"     Stage-2 time    : {s2_time:.1f}s  "
                     f"| total {train_time:.1f}s")

            # ── Evaluate ────────────────────────────────────────
            self.progress(pct_end - 2, f"{tag} – evaluating …")
            self.check_stop()

            yp_enc = clf.predict_flat_gated(Xte, le_b, le_m)
            yp_bin = clf.predict_binary(Xte)

            acc_b      = accuracy_score(y_b_te, yp_bin)
            f1_b_macro = f1_score(y_b_te, yp_bin,
                                  average="macro", zero_division=0)

            # S5 fix: MCC replaces F1-micro
            acc  = accuracy_score(y_m_te, yp_enc)
            mcc  = matthews_corrcoef(y_m_te, yp_enc)
            f1M  = f1_score(y_m_te, yp_enc,
                            average="macro", zero_division=0)
            f1w  = f1_score(y_m_te, yp_enc,
                            average="weighted", zero_division=0)

            self.log(f"     Stage-1 bin acc  : {acc_b:.4f}  "
                     f"(mF1={f1_b_macro:.4f})")
            self.log(f"     Flat acc         : {acc:.4f}  ({acc*100:.2f}%)")
            self.log(f"     MCC              : {mcc:.4f}  "
                     f"[replaces F1-micro — S5 fix]")
            self.log(f"     F1-macro (mF1)   : {f1M:.4f}")
            self.log(f"     F1-weighted(wF1) : {f1w:.4f}")

            rep = classification_report(
                y_m_te, yp_enc,
                target_names=label_names_m,
                output_dict=True, zero_division=0)

            cm_mat = confusion_matrix(
                y_m_te, yp_enc, labels=list(range(n_multi)))

            tr_m_dist = _dist_named(y_m_tr, label_names_m)
            te_m_dist = _dist_named(y_m_te, label_names_m)

            self.progress(pct_end, f"{tag} – done")
            return {
                "accuracy":        round(acc, 4),
                "f1_micro":        round(mcc, 4),   # slot kept; now holds MCC
                "mcc":             round(mcc, 4),
                "f1_macro":        round(f1M, 4),
                "f1_weighted":     round(f1w, 4),
                "s1_accuracy":     round(acc_b, 4),
                "s1_f1_macro":     round(f1_b_macro, 4),
                "train_time":      round(train_time, 1),
                "n_train":         len(Xtr),
                "n_test":          len(Xte),
                "n_train_benign":  int(n_tr_benign),
                "n_train_vuln":    int(n_tr_vuln),
                "n_test_benign":   int(n_te_benign),
                "n_test_vuln":     int(n_te_vuln),
                "train_dist":      tr_m_dist,
                "test_dist":       te_m_dist,
                "train_dist_smote": None,
                "n_classes":       n_multi,
                "smote":           smote,
                "report":          rep,
                "confusion_matrix": cm_mat,
                "cm_labels":       label_names_m,
                "tag":             tag,
                "smote_tag":       smote_tag,
            }

        # ── binary-only sub-round helper (Round 3A / 4A standalone) ─
        def _binary_only_round(Xtr, Xte, ytr_pre, yte,
                                le_b, smote, tag,
                                pct_start, pct_end,
                                n_est, max_depth, lr,
                                subsample, colsample, early_stop,
                                smote_k=5, smote_cap=1200,
                                smote_tgt="median"):
            """Binary XGBoost standalone (Round 3A / Round 4A).
            C3: shared split. C4: val split pre-SMOTE.
            S4: spw from fitted data. S5: MCC replaces F1-micro.
            """
            self.check_stop()
            smote_tag = "with SMOTE" if smote else "no SMOTE"
            self.log(f"\n   ── {tag} [{smote_tag}] ──")
            self.progress(pct_start, f"{tag} – using shared split …")

            train_pct = (1 - _test_size_global) * 100
            test_pct  = _test_size_global * 100
            bin_names = list(le_b.classes_)
            tr_dist   = _dist_named(ytr_pre, bin_names)
            te_dist   = _dist_named(yte, bin_names)
            self.log(f"     Train {len(Xtr):,}  Test {len(Xte):,}  "
                     f"({train_pct:.0f}%/{test_pct:.0f}%), {split_desc_shared}")
            self.log(f"     Train dist       : {tr_dist}")

            try:
                Xf_pre, Xv, yf_pre, yv = train_test_split(
                    Xtr, ytr_pre, test_size=0.1,
                    random_state=SEED, stratify=ytr_pre)
            except ValueError:
                Xf_pre, Xv, yf_pre, yv = train_test_split(
                    Xtr, ytr_pre, test_size=0.1, random_state=SEED)

            tr_named_smote = None
            if smote:
                self.progress(pct_start + 3,
                              f"{tag} – SMOTE (90% train fold only) …")
                self.check_stop()
                Xf, yf = smote_oversample(
                    Xf_pre, yf_pre, k=smote_k, cap=smote_cap, tgt=smote_tgt)
                self.log(f"     After SMOTE      : {len(Xf):,}")
                tr_named_smote = _dist_named(yf, bin_names)
            else:
                Xf, yf = Xf_pre, yf_pre

            self.progress(pct_start + 6, f"{tag} – training …")
            self.check_stop()

            counts = np.bincount(yf.astype(int))
            neg = int(counts[0]) if len(counts) > 0 else 1
            pos = int(counts[1]) if len(counts) > 1 else 1
            spw = neg / max(pos, 1)

            m_clf = xgb.XGBClassifier(
                objective             = "binary:logistic",
                eval_metric           = "logloss",
                n_estimators          = n_est,
                max_depth             = max_depth,
                learning_rate         = lr,
                subsample             = subsample,
                colsample_bytree      = colsample,
                early_stopping_rounds = early_stop,
                scale_pos_weight      = spw,
                tree_method           = "hist",
                random_state          = SEED,
                n_jobs                = -1,
            )
            t0 = time.time()
            m_clf.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
            train_time = time.time() - t0
            self.log(f"     Best iter        : {m_clf.best_iteration}  "
                     f"({train_time:.1f}s, spw={spw:.2f})")

            self.progress(pct_end - 2, f"{tag} – evaluating …")
            yp  = m_clf.predict(Xte)
            acc = accuracy_score(yte, yp)
            mcc = matthews_corrcoef(yte, yp)
            f1M = f1_score(yte, yp, average="macro",    zero_division=0)
            f1w = f1_score(yte, yp, average="weighted", zero_division=0)

            self.log(f"     Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
            self.log(f"     MCC              : {mcc:.4f}")
            self.log(f"     F1-macro         : {f1M:.4f}")
            self.log(f"     F1-weighted      : {f1w:.4f}")

            rep = classification_report(
                yte, yp, target_names=bin_names,
                output_dict=True, zero_division=0)

            n_tr_b = tr_dist.get("Benign", 0)
            n_tr_v = sum(v for k, v in tr_dist.items() if k != "Benign")
            n_te_b = te_dist.get("Benign", 0)
            n_te_v = sum(v for k, v in te_dist.items() if k != "Benign")

            self.progress(pct_end, f"{tag} – done")
            return {
                "accuracy":         round(acc, 4),
                "f1_micro":         round(mcc, 4),
                "mcc":              round(mcc, 4),
                "f1_macro":         round(f1M, 4),
                "f1_weighted":      round(f1w, 4),
                "s1_accuracy":      round(acc, 4),
                "s1_f1_macro":      round(f1M, 4),
                "train_time":       round(train_time, 1),
                "n_train":          len(Xtr),
                "n_test":           len(Xte),
                "n_train_benign":   int(n_tr_b),
                "n_train_vuln":     int(n_tr_v),
                "n_test_benign":    int(n_te_b),
                "n_test_vuln":      int(n_te_v),
                "train_dist":       tr_dist,
                "test_dist":        te_dist,
                "train_dist_smote": tr_named_smote,
                "n_classes":        2,
                "smote":            smote,
                "report":           rep,
                "tag":              tag,
                "smote_tag":        smote_tag,
            }

        pct = 32   # M4 fix: budget starts at 32% after Steps 1-5

        # Pre-slice 768-dim embedding matrix for XGBoost rounds (R3/R4)
        Xtr_emb_shared = X_emb[idx_tr_shared]
        Xte_emb_shared = X_emb[idx_te_shared]

        # ── Retrieve raw snippets for fine-tuned rounds (R1/R2) ──────
        # snips_tr / snips_te were already prepared in STEP 3 above.
        # If SNIPPET column was absent, disable fine-tuned rounds.
        ft_epochs  = int(p.get("ft_epochs",  FT_EPOCHS_DEFAULT))
        ft_lr      = float(p.get("ft_lr",    FT_LR_DEFAULT))
        ft_batch   = int(p.get("ft_batch",   FT_BATCH_DEFAULT))
        ft_max_len = int(p.get("ft_max_len", FT_MAX_LEN_DEFAULT))

        if (do_r1_ft or do_r2_ft) and snips_tr is None:
            self.log("   ⚠  'SNIPPET' column not found — "
                     "Rounds 1/2 (fine-tuned) require raw code text. "
                     "Skipping.", "warn")
            do_r1_ft = do_r2_ft = False

        # ════════════════════════════════════════════════════════════
        #  STEP 6 — ROUND 1: CodeBERT Fine-tuned Classifier
        #  Data source : STEP 4 (CodeBERT embeddings — train + test)
        #  Arch        : Code snippets → CB → [CLS] 768-dim →
        #                Linear(768→n_classes) → Softmax
        # ════════════════════════════════════════════════════════════
        if do_r1_ft:
            self.log("━" * 58)
            self.log("📊  STEP 6 — ROUND 1: CodeBERT Fine-tuned Classifier")
            self.log(f"   Data source : STEP 4  (CodeBERT embeddings, "
                     f"train {len(snips_tr):,} / test {len(snips_te):,})")
            self.log(f"   Model : {CODEBERT_MODEL_NAME}")
            self.log(f"   Arch  : [CLS] 768 → Linear({n_multi}) → Softmax")
            self.log(f"   Hyper : epochs={ft_epochs}  lr={ft_lr:.0e}  "
                     f"batch={ft_batch}  max_len={ft_max_len}")

            # 1A — Binary
            self.log("━" * 40)
            self.log("   Round 1A — Binary (Benign / Vulnerable)")
            self.check_stop()
            m1a = run_bert_round(
                snips_tr, snips_te,
                y_b_tr_shared, y_b_te_shared,
                model_name    = CODEBERT_MODEL_NAME,
                n_classes     = 2,
                le            = le_b,
                tag           = "R1A — CB Binary",
                device        = self.device,
                epochs        = ft_epochs,
                lr            = ft_lr,
                batch_size    = ft_batch,
                max_len       = ft_max_len,
                log_fn        = self.log,
                progress_fn   = self.progress,
                check_stop_fn = self.check_stop,
                pct_start     = pct,
                pct_end       = pct + 9)
            self.results["R1_binary"] = m1a
            pct += 9

            # 1B — Multi-class (69 classes)
            self.log("━" * 40)
            self.log(f"   Round 1B — Multi-class ({n_multi} classes)")
            self.check_stop()
            m1b = run_bert_round(
                snips_tr, snips_te,
                y_m_tr_shared, y_m_te_shared,
                model_name    = CODEBERT_MODEL_NAME,
                n_classes     = n_multi,
                le            = le_m,
                tag           = "R1B — CB Multi-class",
                device        = self.device,
                epochs        = ft_epochs,
                lr            = ft_lr,
                batch_size    = ft_batch,
                max_len       = ft_max_len,
                log_fn        = self.log,
                progress_fn   = self.progress,
                check_stop_fn = self.check_stop,
                pct_start     = pct,
                pct_end       = pct + 11)
            self.results["R1_multi"] = m1b
            pct += 11

        # ════════════════════════════════════════════════════════════
        #  STEP 7 — ROUND 2: GraphCodeBERT Fine-tuned Classifier
        #  Data source : STEP 5 (GraphCodeBERT embeddings — train + test)
        #  Arch        : Code snippets → GCB → [CLS] 768-dim →
        #                Linear(768→n_classes) → Softmax
        # ════════════════════════════════════════════════════════════
        if do_r2_ft:
            self.log("━" * 58)
            self.log("📊  STEP 7 — ROUND 2: GraphCodeBERT Fine-tuned Classifier")
            self.log(f"   Data source : STEP 5  (GraphCodeBERT embeddings, "
                     f"train {len(snips_tr):,} / test {len(snips_te):,})")
            self.log(f"   Model : {GRAPHCODEBERT_MODEL_NAME}")
            self.log(f"   Arch  : [CLS] 768 → Linear({n_multi}) → Softmax")
            self.log(f"   Hyper : epochs={ft_epochs}  lr={ft_lr:.0e}  "
                     f"batch={ft_batch}  max_len={ft_max_len}")

            # 2A — Binary
            self.log("━" * 40)
            self.log("   Round 2A — Binary (Benign / Vulnerable)")
            self.check_stop()
            m2a = run_bert_round(
                snips_tr, snips_te,
                y_b_tr_shared, y_b_te_shared,
                model_name    = GRAPHCODEBERT_MODEL_NAME,
                n_classes     = 2,
                le            = le_b,
                tag           = "R2A — GCB Binary",
                device        = self.device,
                epochs        = ft_epochs,
                lr            = ft_lr,
                batch_size    = ft_batch,
                max_len       = ft_max_len,
                log_fn        = self.log,
                progress_fn   = self.progress,
                check_stop_fn = self.check_stop,
                pct_start     = pct,
                pct_end       = pct + 9)
            self.results["R2_binary"] = m2a
            pct += 9

            # 2B — Multi-class (69 classes)
            self.log("━" * 40)
            self.log(f"   Round 2B — Multi-class ({n_multi} classes)")
            self.check_stop()
            m2b = run_bert_round(
                snips_tr, snips_te,
                y_m_tr_shared, y_m_te_shared,
                model_name    = GRAPHCODEBERT_MODEL_NAME,
                n_classes     = n_multi,
                le            = le_m,
                tag           = "R2B — GCB Multi-class",
                device        = self.device,
                epochs        = ft_epochs,
                lr            = ft_lr,
                batch_size    = ft_batch,
                max_len       = ft_max_len,
                log_fn        = self.log,
                progress_fn   = self.progress,
                check_stop_fn = self.check_stop,
                pct_start     = pct,
                pct_end       = pct + 11)
            self.results["R2_multi"] = m2b
            pct += 11

        # ════════════════════════════════════════════════════════════
        #  STEP 8 — ROUND 3: GraphCodeBERT + XGBoost  (768-dim, no HC)
        #  Data source : STEP 5 (GCB embeddings, 768-dim, train + test)
        #  Round 3A    : Binary standalone  (Benign / Vulnerable, 768-dim)
        #  Round 3B    : Multi-class Flat  (ALL 69 Mtag classes incl.
        #                Benign; Stage-1 gate → flat Stage-2)
        #  C-NEW fix   : Round 3B uses run_round_flat() so Benign is in
        #                Stage-2 training distribution (not excluded as
        #                in Round 4B).
        # ════════════════════════════════════════════════════════════
        if do_r3_flat:
            self.log("━" * 58)
            self.log("📊  STEP 8 — ROUND 3: GraphCodeBERT + XGBoost "
                     "(768-dim, no handcrafted features)")
            self.log(f"   Data source : STEP 5  (GCB 768-dim embeddings, "
                     f"train {Xtr_emb_shared.shape[0]:,} / "
                     f"test {Xte_emb_shared.shape[0]:,})")
            self.log(f"   Model    : {GRAPHCODEBERT_MODEL_NAME}  (frozen extractor)")
            self.log(f"   Features : GCB [CLS] 768-dim only — no handcrafted features")
            self.results["round3_feature_shape"] = list(X_emb.shape)

            # ── 3A — Binary standalone (Benign / Vulnerable, 768-dim) ──
            self.log("━" * 40)
            self.log("   Round 3A — Binary standalone  (768-dim)")
            if do_no_sm:
                m = _binary_only_round(
                    Xtr_emb_shared, Xte_emb_shared,
                    y_b_tr_shared, y_b_te_shared,
                    le_b, smote=False,
                    tag="Round 3A — Binary (768-dim)",
                    pct_start=pct, pct_end=pct + 10,
                    n_est=n_est, max_depth=max_depth,
                    lr=xgb_lr, subsample=subsample,
                    colsample=colsample, early_stop=early_stop)
                self.results["R3_noSMOTE"] = m
                pct += 10
            self.check_stop()
            if do_smote:
                m = _binary_only_round(
                    Xtr_emb_shared, Xte_emb_shared,
                    y_b_tr_shared, y_b_te_shared,
                    le_b, smote=True,
                    tag="Round 3A — Binary (768-dim)",
                    pct_start=pct, pct_end=pct + 12,
                    n_est=n_est, max_depth=max_depth,
                    lr=xgb_lr, subsample=subsample,
                    colsample=colsample, early_stop=early_stop,
                    smote_k=int(p.get("smote_k", 5)),
                    smote_cap=int(p.get("smote_cap", 1200)),
                    smote_tgt=p.get("smote_target", "median"))
                self.results["R3_SMOTE"] = m
                pct += 12

            # ── 3B — Multi-class Flat (ALL rows, 69 Mtag classes) ──────
            self.log("━" * 40)
            self.log(f"   Round 3B — Multi-class Flat  "
                     f"({n_multi} Mtag classes incl. Benign, 768-dim)")
            self.log(f"   Inference: Stage-1 gate → flat Stage-2 (Benign eligible)")
            if do_no_sm:
                m = run_round_flat(Xtr_emb_shared, Xte_emb_shared,
                                    y_b_tr_shared, y_b_te_shared,
                                    y_m_tr_shared, y_m_te_shared,
                                    "Round 3B — Multi-class Flat (768-dim)",
                                    list(le_m.classes_),
                                    list(le_b.classes_),
                                    smote=False,
                                    pct_start=pct, pct_end=pct + 12)
                self.results["R3B_noSMOTE"] = m
                pct += 12
            self.check_stop()
            if do_smote:
                m = run_round_flat(Xtr_emb_shared, Xte_emb_shared,
                                    y_b_tr_shared, y_b_te_shared,
                                    y_m_tr_shared, y_m_te_shared,
                                    "Round 3B — Multi-class Flat (768-dim)",
                                    list(le_m.classes_),
                                    list(le_b.classes_),
                                    smote=True,
                                    pct_start=pct, pct_end=pct + 14)
                self.results["R3B_SMOTE"] = m
                pct += 14

        # ════════════════════════════════════════════════════════════
        #  STEP 9 — ROUND 4: Two-Stage Hierarchical XGBoost (780-dim + HC)
        #  Data source : STEP 5 (GCB 780-dim + 12-dim HC = 780-dim)
        #  Stage-1     : binary gate  (Benign / Vulnerable)
        #  Stage-2     : CWE multi-class  (Vulnerable rows ONLY)
        # ════════════════════════════════════════════════════════════

        if do_r4_hier:
            self.log("━" * 58)
            self.log(f"📊  STEP 9 — ROUND 4: Two-Stage Hierarchical XGBoost  "
                     f"({n_multi} Mtag classes, GCB 780-dim + HC)")
            self.log(f"   Data source : STEP 5  (GCB 780-dim + 12-dim HC, "
                     f"train {Xtr_shared.shape[0]:,} / "
                     f"test {Xte_shared.shape[0]:,})")
            self.log("   Stage-1  : Binary gate  (Benign / Vulnerable, 780-dim)")
            self.log("   Stage-2  : Multi-class XGBoost — Vulnerable rows ONLY")

            # 4A — Binary standalone (Stage-1 metrics)
            self.log("━" * 40)
            self.log("   Round 4A — Binary standalone  (Stage-1, 780-dim)")
            if do_no_sm:
                m = _binary_only_round(
                    Xtr_shared, Xte_shared,
                    y_b_tr_shared, y_b_te_shared,
                    le_b, smote=False,
                    tag="Round 4A — Binary (780-dim)",
                    pct_start=pct, pct_end=pct + 10,
                    n_est=n_est, max_depth=max_depth,
                    lr=xgb_lr, subsample=subsample,
                    colsample=colsample, early_stop=early_stop)
                self.results["R4_noSMOTE"] = m
                pct += 10
            self.check_stop()
            if do_smote:
                m = _binary_only_round(
                    Xtr_shared, Xte_shared,
                    y_b_tr_shared, y_b_te_shared,
                    le_b, smote=True,
                    tag="Round 4A — Binary (780-dim)",
                    pct_start=pct, pct_end=pct + 12,
                    n_est=n_est, max_depth=max_depth,
                    lr=xgb_lr, subsample=subsample,
                    colsample=colsample, early_stop=early_stop,
                    smote_k=int(p.get("smote_k", 5)),
                    smote_cap=int(p.get("smote_cap", 1200)),
                    smote_tgt=p.get("smote_target", "median"))
                self.results["R4_SMOTE"] = m
                pct += 12

            # 4B — Hierarchical multi-class (Stage-1 + Stage-2, 780-dim)
            self.log("━" * 40)
            self.log(f"   Round 4B — Hierarchical Multi-class  "
                     f"({n_multi} CWE classes, Stage-2 Vuln-only, 780-dim)")
            if do_no_sm:
                m = run_round(Xtr_shared, Xte_shared,
                              y_b_tr_shared, y_b_te_shared,
                              y_m_tr_shared, y_m_te_shared,
                              "Round 4B — Hierarchical (780-dim)",
                              list(le_m.classes_),
                              list(le_b.classes_),
                              smote=False,
                              pct_start=pct, pct_end=pct + 12)
                self.results["R4B_noSMOTE"] = m
                pct += 12
            self.check_stop()
            if do_smote:
                m = run_round(Xtr_shared, Xte_shared,
                              y_b_tr_shared, y_b_te_shared,
                              y_m_tr_shared, y_m_te_shared,
                              "Round 4B — Hierarchical (780-dim)",
                              list(le_m.classes_),
                              list(le_b.classes_),
                              smote=True,
                              pct_start=pct, pct_end=pct + 14)
                self.results["R4B_SMOTE"] = m
                pct += 14

        # ── STEP 10 — Generate PDF report + comparison charts ───
        self.log("━" * 58)
        self.log("📈  STEP 10 — Generating comparison charts and PDF report …")
        self.progress(92, "Generating charts …")

        # C1 fix: 'Use CodeBERT precomputed features' is now functional —
        # if a valid precomputed CB file is supplied, the costly full-corpus
        # CodeBERT forward pass is SKIPPED and the existing file is reused
        # as-is. (Round 1 fine-tuning still always re-tokenises raw text
        # internally inside run_bert_round() — fine-tuning cannot consume
        # a static embedding matrix because backbone weights change every
        # epoch. What CB precomputed mode actually saves is the *separate*
        # frozen-embedding export pass below, not Round 1 itself.)
        cb_cols_expected = [f"cb_emb_{i}" for i in range(EMBEDDING_DIM)]

        def _cb_precomputed_is_valid(path):
            if not path or not os.path.isfile(path):
                return False, None
            try:
                _df_cb = pd.read_excel(path) if path.lower().endswith(
                    (".xlsx", ".xls")) else pd.read_csv(path)
            except Exception as exc:
                self.log(f"   ⚠  CB precomputed file unreadable: {exc}",
                         "warn")
                return False, None
            missing = [c for c in cb_cols_expected if c not in _df_cb.columns]
            if missing:
                self.log(f"   ⚠  CB precomputed file missing "
                         f"{len(missing)} of {EMBEDDING_DIM} cb_emb_* "
                         f"columns — cannot reuse.", "warn")
                return False, None
            return True, _df_cb

        if use_cb_pre and cb_pre_path:
            self.log("━" * 58)
            self.log("📂  CB precomputed file supplied — validating reuse "
                     "instead of re-extraction …")
            _valid, _df_cb_pre = _cb_precomputed_is_valid(cb_pre_path)
            if _valid:
                self.results["cb_feature_dataset_path"] = cb_pre_path
                self.results["cb_embedding_reused"]      = True
                self.log(f"   ✓ Reusing existing CB precomputed file: "
                         f"{cb_pre_path}  ({len(_df_cb_pre):,} rows)")
                self.log("   ⏭️  Skipping CB feature export "
                         "(C1 fix: precomputed file reused).")
                export_cb_pre = False   # don't re-run extraction below
            else:
                self.log("   ⚠  Supplied CB precomputed file is invalid "
                         "or incompatible — falling back to live "
                         "extraction if export is requested.", "warn")

        # Export CodeBERT precomputed features if requested
        # S2 fix: budget kept inside 92-95% so it never collides with the
        # PDF-generation step which starts at 96% — eliminates the
        # progress-bar regression (99% -> 96%) seen in V13.
        if export_cb_pre and snippets_all:
            self.log("━" * 58)
            self.log("💾  Exporting CodeBERT precomputed feature dataset "
                     "(export_cb_pre=True) …")
            self._export_cb_feature_dataset(
                snippets   = np.array(snippets_all, dtype=object),
                le_m       = le_m,
                le_b       = le_b,
                y_m        = y_m,
                y_b        = y_b,
                csv_path   = p["csv_path"],
                max_len    = ft_max_len,
                batch_size = ft_batch,
                device     = self.device,
                pct_start  = 92,
                pct_end    = 95)
        elif export_cb_pre and not snippets_all:
            self.log("   ⚠  CB export skipped — no SNIPPET column in dataset.")

        r1_keys, r2_keys, r3_keys, r4_keys = [], [], [], []

        for rk, label in [("R1_binary","CB Binary"), ("R1_multi","CB Multi")]:
            if rk in self.results:
                r1_keys.append((label, rk))
        if r1_keys:
            self.figures.append(("Round 1 — CodeBERT fine-tuned",
                self._metric_bar("Round 1 — CodeBERT Fine-tuned",
                                  r1_keys, self.results)))

        for rk, label in [("R2_binary","GCB Binary"), ("R2_multi","GCB Multi")]:
            if rk in self.results:
                r2_keys.append((label, rk))
        if r2_keys:
            self.figures.append(("Round 2 — GCB fine-tuned",
                self._metric_bar("Round 2 — GraphCodeBERT Fine-tuned",
                                  r2_keys, self.results)))

        if do_r3_flat:
            if do_no_sm and "R3_noSMOTE" in self.results:
                r3_keys.append(("3A No SMOTE", "R3_noSMOTE"))
            if do_smote and "R3_SMOTE" in self.results:
                r3_keys.append(("3A SMOTE",    "R3_SMOTE"))
            if do_no_sm and "R3B_noSMOTE" in self.results:
                r3_keys.append(("3B No SMOTE", "R3B_noSMOTE"))
            if do_smote and "R3B_SMOTE" in self.results:
                r3_keys.append(("3B SMOTE",    "R3B_SMOTE"))
        if r3_keys:
            self.figures.append(("Round 3 — GraphCodeBERT + XGBoost",
                self._metric_bar("Round 3 — GraphCodeBERT + XGBoost (768-dim)",
                                  r3_keys, self.results)))

        if do_r4_hier:
            if do_no_sm and "R4_noSMOTE" in self.results:
                r4_keys.append(("4A No SMOTE", "R4_noSMOTE"))
            if do_smote and "R4_SMOTE" in self.results:
                r4_keys.append(("4A SMOTE",    "R4_SMOTE"))
            if do_no_sm and "R4B_noSMOTE" in self.results:
                r4_keys.append(("4B No SMOTE", "R4B_noSMOTE"))
            if do_smote and "R4B_SMOTE" in self.results:
                r4_keys.append(("4B SMOTE",    "R4B_SMOTE"))
        if r4_keys:
            self.figures.append(("Round 4 — Hierarchical XGBoost",
                self._metric_bar("Round 4 — Two-Stage Hierarchical XGBoost (780-dim)",
                                  r4_keys, self.results)))

        all_keys_proper = []
        for tag_pfx, key_pfx in [
                ("R1-CB-Bin",   "R1_binary"),   ("R1-CB-Multi",  "R1_multi"),
                ("R2-GCB-Bin",  "R2_binary"),   ("R2-GCB-Multi", "R2_multi"),
                ("R3A-NoSMOTE", "R3_noSMOTE"),  ("R3A-SMOTE",    "R3_SMOTE"),
                ("R3B-NoSMOTE", "R3B_noSMOTE"), ("R3B-SMOTE",    "R3B_SMOTE"),
                ("R4A-NoSMOTE", "R4_noSMOTE"),  ("R4A-SMOTE",    "R4_SMOTE"),
                ("R4B-NoSMOTE", "R4B_noSMOTE"), ("R4B-SMOTE",    "R4B_SMOTE")]:
            if key_pfx in self.results:
                all_keys_proper.append((tag_pfx, key_pfx))
        if len(all_keys_proper) >= 2:
            self.figures.append(("All configurations overview",
                self._metric_bar("All configurations — performance overview",
                                  all_keys_proper, self.results)))

        self.progress(96, "Building PDF report …")

        # ── 4. PDF ───────────────────────────────────────────────
        if p["save_pdf"]:
            self.log("━" * 58)
            self.log("📄  STEP 10 — Generating PDF report …")
            self.check_stop()
            pdf_path = PDFReporter(
                self.results, self.figures, p).build()
            self.results["pdf_path"] = pdf_path
            self.log(f"   PDF saved         : {pdf_path}")

        self.progress(100, "Complete ✓")
        self.log("━" * 58)
        self.log("✅  Pipeline complete.")


# ═══════════════════════════════════════════════════════════════════
#  PDF REPORTER  (updated to reflect new pipeline)
# ═══════════════════════════════════════════════════════════════════
class PDFReporter:
    PAGE_W, PAGE_H = A4

    def __init__(self, results, figures, params):
        self.r    = results
        self.figs = figures
        self.p    = params
        self.styles = getSampleStyleSheet()
        self._define_styles()

    def _define_styles(self):
        s = self.styles
        base = {"fontName": "Helvetica", "fontSize": 10,
                "textColor": colors.HexColor("#1A1D2E"), "leading": 14}
        s.add(ParagraphStyle("ReportTitle",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=colors.HexColor("#1A3A6B"),
            alignment=TA_CENTER, spaceAfter=6))
        s.add(ParagraphStyle("SubTitle",
            fontName="Helvetica-Oblique", fontSize=11,
            textColor=colors.HexColor("#4F6FA0"),
            alignment=TA_CENTER, spaceAfter=18))
        s.add(ParagraphStyle("SectionHead",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=colors.HexColor("#1A3A6B"),
            spaceBefore=16, spaceAfter=6))
        s.add(ParagraphStyle("SubHead",
            fontName="Helvetica-Bold", fontSize=10.5,
            textColor=colors.HexColor("#2C5F8A"),
            spaceBefore=10, spaceAfter=4))
        s.add(ParagraphStyle("Body",
            **{**base, "spaceAfter": 4}))
        s.add(ParagraphStyle("Caption",
            fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=colors.HexColor("#5C6B7A"),
            alignment=TA_CENTER, spaceAfter=10))
        s.add(ParagraphStyle("TableHdr",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=colors.white, alignment=TA_CENTER))
        s.add(ParagraphStyle("TableCell",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#1A1D2E"),
            alignment=TA_CENTER))
        s.add(ParagraphStyle("TableCellL",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#1A1D2E"),
            alignment=TA_LEFT))
        s.add(ParagraphStyle("Metric",
            fontName="Helvetica-Bold", fontSize=16,
            textColor=colors.HexColor("#1A3A6B"),
            alignment=TA_CENTER))

    def _table(self, data, col_widths, hdr_color=None):
        if hdr_color is None:
            hdr_color = colors.HexColor("#1A3A6B")
        t = Table(data, colWidths=col_widths)
        style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), hdr_color),
            ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,  0), 9),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#F5F8FC"), colors.white]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.4,
             colors.HexColor("#BCC8D8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ])
        t.setStyle(style)
        return t

    def _metrics_table(self, key, label):
        m = self.r.get(key)
        if not m:
            return None
        data = [
            ["Metric", "Value"],
            ["Accuracy (hierarchical)", f"{m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)"],
            ["Stage-1 Accuracy (binary)", f"{m.get('s1_accuracy', m['accuracy']):.4f}"],
            ["Stage-1 F1-macro",          f"{m.get('s1_f1_macro', 0):.4f}"],
            ["MCC (Matthews Corr. Coef.)", f"{m.get('mcc', m.get('f1_micro', 0)):.4f}"],
            ["F1-Macro (mF1)",            f"{m['f1_macro']:.4f}"],
            ["F1-Weighted (wF1)",         f"{m['f1_weighted']:.4f}"],
            ["Training samples",          f"{m['n_train']:,}"],
            ["Test samples",              f"{m['n_test']:,}"],
            ["Train — Benign",            f"{m['n_train_benign']:,}"],
            ["Train — Vulnerable",        f"{m['n_train_vuln']:,}"],
            ["Test — Benign",             f"{m['n_test_benign']:,}"],
            ["Test — Vulnerable",         f"{m['n_test_vuln']:,}"],
            ["Training time",             f"{m['train_time']:.1f} s"],
            ["SMOTE applied",             "Yes" if m["smote"] else "No"],
        ]
        return self._table(data, [9*cm, 7*cm])

    def _fig_to_image(self, fig, dpi=110):
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return RLImage(buf, width=16*cm, height=8*cm)

    def _confusion_matrix_image(self, conf_mat, labels, title):
        """Render an N×N confusion-matrix heatmap as an RLImage.

        Cell counts are annotated only when the class count is small
        enough to stay legible; tick labels are thinned for large
        class counts (e.g. the 60+ CWE categories typical of
        UAVulDB01) while the colour scale still conveys the diagonal-
        dominance pattern expected of a well-trained classifier.
        """
        n = len(labels)
        side_in = max(5.0, min(0.16 * n + 2.5, 11.0))
        fig, ax = plt.subplots(figsize=(side_in, side_in))
        fig.patch.set_facecolor("white")
        im = ax.imshow(conf_mat, cmap="Blues", aspect="equal")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted class", fontsize=8)
        ax.set_ylabel("True class", fontsize=8)

        if n <= 30:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90,
                               fontsize=5.5 if n > 16 else 7)
            ax.set_yticklabels(labels, fontsize=5.5 if n > 16 else 7)
        else:
            step = max(1, n // 20)
            ticks = list(range(0, n, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=6)
            ax.set_yticklabels(ticks, fontsize=6)

        if n <= 16:
            vmax = conf_mat.max() if conf_mat.max() > 0 else 1
            thresh = vmax / 2
            for i in range(n):
                for j in range(n):
                    v = int(conf_mat[i, j])
                    if v > 0:
                        ax.text(j, i, str(v), ha="center", va="center",
                                fontsize=6,
                                color="white" if v > thresh else "#1A1D2E")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return RLImage(buf, width=14*cm, height=14*cm)

    def _confusion_matrix_table(self, conf_mat, labels):
        """Per-class one-vs-rest confusion breakdown (TP/FP/FN/TN)."""
        n     = len(labels)
        total = int(conf_mat.sum())
        data = [["CWE / Class", "TP", "FP", "FN", "TN",
                 "Precision", "Recall", "F1", "Support"]]
        for i, lbl in enumerate(labels):
            tp = int(conf_mat[i, i])
            fp = int(conf_mat[:, i].sum() - tp)
            fn = int(conf_mat[i, :].sum() - tp)
            tn = total - tp - fp - fn
            support = tp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0.0)
            data.append([lbl, str(tp), str(fp), str(fn), str(tn),
                        f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}",
                        str(support)])
        return self._table(
            data,
            [3.6*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm,
             1.9*cm, 1.7*cm, 1.5*cm, 1.7*cm])

    def build(self):
        out_dir  = os.path.dirname(self.p["csv_path"])
        pdf_path = os.path.join(out_dir, "VulnDetection_Report.pdf")
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.5*cm, bottomMargin=2*cm,
            title="Vulnerability Detection Report",
            author="GraphCodeBERT + Two-Stage XGBoost Framework")
        story = []
        S = self.styles

        # ── Cover ────────────────────────────────────────────────
        story += [
            Spacer(1, 1.5*cm),
            Paragraph("UAVulDetect — Multi-Round Vulnerability Detection", S["ReportTitle"]),
            Paragraph("CodeBERT · GraphCodeBERT · XGBoost  —  Experiment Report  (V16)", S["SubTitle"]),
            HRFlowable(width="100%", thickness=2,
                       color=colors.HexColor("#1A3A6B"), spaceAfter=8),
            Paragraph(f"Generated: {time.strftime('%Y-%m-%d  %H:%M:%S')}",
                      S["Caption"]),
            Paragraph(f"Dataset: {os.path.basename(self.p['csv_path'])}",
                      S["Caption"]),
            Spacer(1, 0.5*cm),
        ]

        # Experiment parameters table
        story.append(Paragraph("Experiment Parameters", S["SectionHead"]))
        test_pct = self.p.get("test_size", 0.2) * 100
        pdefs = [
            ["Parameter", "Value"],
            ["Primary dataset",        self.p["csv_path"]],
            ["GCB precomputed file",   self.p.get("gcb_pre_path", "—") or "—"],
            ["GCB precomputed used",   "Yes" if self.p.get(
                                           "use_precomputed_features",
                                           False) else "No"],
            ["CB precomputed file",    self.p.get("cb_pre_path", "—") or "—"],
            ["CB precomputed used",    "Yes" if self.p.get(
                                           "use_cb_precomputed_features",
                                           False) else "No"],
            ["CB precomputed reused",  "Yes (export skipped)" if self.r.get(
                                           "cb_embedding_reused", False)
                                           else "No"],
            ["Export CB features",     "Yes" if self.p.get(
                                           "export_cb_pre", False) else "No"],
            ["Export GCB features",    "Yes" if self.p.get(
                                           "export_gcb_pre", True) else "No"],
            ["Total samples",          f"{self.p['n_samples']:,}"],
            ["STEP 3 — Train split",   f"{100-int(float(self.p.get('test_size',0.2))*100)}%"],
            ["STEP 3 — Test split",    f"{int(float(self.p.get('test_size',0.2))*100)}%"],
            ["STEP 3 — Split strategy",self.r.get("split_desc", "—")],
            ["STEP 4 — GCB model",     GRAPHCODEBERT_MODEL_NAME],
            ["STEP 5 — CB fine-tune",  "Loaded on-demand inside run_bert_round()"],
            ["STEP 6 — Round 1 (CB FT)",
             "Yes" if self.p.get("do_r1_ft", True) else "No"],
            ["STEP 7 — Round 2 (GCB FT)",
             "Yes" if self.p.get("do_r2_ft", True) else "No"],
            ["STEP 8 — Round 3 (3A Binary / 3B Multi-class Flat)",
             "Yes" if self.p.get("do_flat", False) else "No"],
            ["STEP 9 — Round 4 (Two-Stage Hierarchical XGB)",
             "Yes" if self.p.get("do_binary", False) else "No"],
            ["SMOTE (Rounds 3 & 4)",   f"No SMOTE: {self.p['do_no_smote']}  |  "
                                        f"With SMOTE: {self.p['do_smote']}"],
        ]
        if self.r.get("feature_dataset_path"):
            pdefs.append(["Exported GCB feature dataset",
                          self.r["feature_dataset_path"]])
        if self.r.get("cb_feature_dataset_path"):
            pdefs.append(["Exported CB feature dataset",
                          self.r["cb_feature_dataset_path"]])
        story.append(self._table(pdefs, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Model configuration
        story.append(Paragraph("Model Configuration", S["SectionHead"]))
        story.append(Paragraph(
            "The V12 pipeline follows ten sequential steps. "
            "Step 1 loads the raw dataset (UAVulDB01, SARD + NVD/CVE). "
            "Step 2 deduplicates by SHA-256 snippet hash. "
            "Step 3 applies a single Mtag-stratified 80/20 split "
            "(SEED=42) shared by all four rounds, and prepares raw code "
            "snippet lists for fine-tuned rounds. "
            "Step 4 runs GraphCodeBERT (microsoft/graphcodebert-base) as a "
            "frozen feature extractor over the full corpus; the resulting "
            "768-dim [CLS] embeddings are concatenated with 12-dim "
            "handcrafted structural features to produce a 780-dim matrix, "
            "then sliced into train/test partitions. "
            "Step 4 is skipped automatically when only Round 1 (CB "
            "fine-tuned) is selected. "
            "Step 5 is a data-preparation note: CodeBERT and "
            "GraphCodeBERT for fine-tuned rounds (R1/R2) are loaded "
            "on-demand inside run_bert_round() from raw text — static "
            "pre-extraction is architecturally incompatible with fine-tuning "
            "because backbone weights change each epoch. "
            "Step 6 (Round 1) fine-tunes CodeBERT end-to-end: "
            "[CLS] 768 → Linear(768, 69) → Softmax, with weighted "
            "CrossEntropyLoss, 10% val-fold checkpoint selection, and "
            "seeded DataLoader. "
            "Step 7 (Round 2) applies the identical fine-tuning setup "
            "with GraphCodeBERT as the backbone. "
            "Step 8 (Round 3) trains two XGBoost classifiers on GCB "
            "768-dim embeddings: Round 3A is a standalone binary "
            "classifier (Benign / Vulnerable); Round 3B is a flat "
            "multi-class classifier over all 69 Mtag classes including "
            "Benign, gated by the Stage-1 binary prediction. "
            "Step 9 (Round 4) trains a two-stage hierarchical XGBoost "
            "on 780-dim features: Stage-1 binary gate then Stage-2 "
            "multi-class on Vulnerable rows only. "
            "Step 10 generates charts and the PDF report.",
            S["Body"]))

        story.append(Paragraph("GraphCodeBERT Embeddings (Rounds 2–4)", S["SubHead"]))
        cb_tbl = [
            ["Parameter", "Value"],
            ["GCB Model (R2 FT / R3 / R4)",  GRAPHCODEBERT_MODEL_NAME],
            ["CB Model (R1 FT)",              CODEBERT_MODEL_NAME],
            ["Pooling strategy",              self.p.get("pooling", "cls")],
            ["Max token length (FT R1/R2)",   str(self.p.get("ft_max_len",
                                                              FT_MAX_LEN_DEFAULT))],
            ["Max token length (Emb R3/R4)",  str(self.p.get("max_len", 512))],
            ["Embedding dim",                 "768"],
            ["Handcrafted features (R4 only)","12  (length, ops, mem-funcs, lang one-hot …)"],
            ["Total feature dim  R3 (Flat)",  "768  (GCB embedding only)"],
            ["Total feature dim  R4 (Hier.)", "780  (768 embedding + 12 handcrafted)"],
            ["Embed batch size",              str(self.p.get("embed_batch", 32))],
        ]
        story.append(self._table(cb_tbl, [9*cm, 7*cm]))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph("XGBoost Hyperparameters", S["SubHead"]))
        xgb_tbl = [
            ["Parameter", "Value"],
            ["n_estimators",         str(self.p.get("n_estimators", 400))],
            ["max_depth",            str(self.p.get("max_depth", 7))],
            ["learning_rate",        str(self.p.get("xgb_lr", 0.05))],
            ["subsample",            str(self.p.get("subsample", 0.8))],
            ["colsample_bytree",     str(self.p.get("colsample", 0.8))],
            ["early_stopping_rounds",str(self.p.get("early_stop", 30))],
            ["Stage-1 objective",    "binary:logistic"],
            ["Stage-2 objective",    "multi:softprob"],
            ["Imbalance handling",   "scale_pos_weight (Stage 1) + SMOTE (Stage 2)"],
        ]
        story.append(self._table(xgb_tbl, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph("SMOTE", S["SubHead"]))
        sm_tbl = [
            ["Parameter", "Value"],
            ["k-neighbours",        str(self.p.get("smote_k", 5))],
            ["Max synthetic/class", str(self.p.get("smote_cap", 1200))],
            ["Balance target",      self.p.get("smote_target", "median")],
            ["Applied to",          "Stage-2 training partition only"],
        ]
        story.append(self._table(sm_tbl, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph(
            "Fine-tuning Hyperparameters (Rounds 1 & 2)", S["SubHead"]))
        ft_cfg_tbl = [
            ["Parameter", "Value"],
            ["Round 1 backbone (CodeBERT)",      CODEBERT_MODEL_NAME],
            ["Round 2 backbone (GraphCodeBERT)", GRAPHCODEBERT_MODEL_NAME],
            ["Epochs (max)",    str(self.p.get("ft_epochs",  FT_EPOCHS_DEFAULT))],
            ["Learning rate",   str(self.p.get("ft_lr",      FT_LR_DEFAULT))],
            ["Batch size",      str(self.p.get("ft_batch",   FT_BATCH_DEFAULT))],
            ["Max token length",str(self.p.get("ft_max_len", FT_MAX_LEN_DEFAULT))],
            ["Optimiser",       "AdamW  (weight_decay=0.01)"],
            ["LR schedule",     "Linear warmup (10% of steps) + linear decay"],
            ["Loss function",   "CrossEntropyLoss  (class-weighted — C2 fix)"],
            ["Architecture",    "[CLS] 768 → Linear(768, 69) → Softmax"],
            ["Tokenisation",    "Dynamic per-batch (padding=longest — C1 fix)"],
            ["Model selection", "Best-val-loss checkpoint saved per epoch (S1 fix)"],
            ["Val fold",        "10% of train split, carved pre-training (S1 fix)"],
            ["Reproducibility", "Seeded DataLoader + torch.manual_seed (S3 fix)"],
            ["Benign count",    "LabelEncoder-based index lookup (S2 fix)"],
            ["Sub-rounds",      "A: Binary (2 classes)  |  B: Multi-class (69 classes)"],
        ]
        story.append(self._table(ft_cfg_tbl, [9*cm, 7*cm]))
        story.append(Spacer(1, 0.4*cm))

        # Dataset statistics
        story.append(Paragraph("Dataset Statistics", S["SectionHead"]))
        dstat = [
            ["Statistic", "Value"],
            ["Raw samples",               f"{self.r.get('n_raw', 'N/A'):,}"],
            ["After deduplication",       f"{self.r.get('n_dedup', 'N/A'):,}"],
            ["Working set size",          f"{self.r.get('n_working', 'N/A'):,}"],
            ["Multi-class categories",    str(self.r.get('mtag_working_n', 'N/A'))],
            ["Feature matrix shape (R1 / R2)", str(self.r.get("feature_shape", "N/A"))],
        ]
        if self.r.get("round3_feature_shape"):
            dstat.append(["Feature matrix shape (R3 — Flat)",
                          str(self.r["round3_feature_shape"])])
        bw = self.r.get("btag_working", {})
        for lbl, cnt in bw.items():
            pct = 100 * cnt / max(sum(bw.values()), 1)
            dstat.append([f"Class: {lbl}", f"{cnt:,}  ({pct:.1f}%)"])
        story.append(self._table(dstat, [8*cm, 8*cm]))
        for title, fig in self.figs:
            if "distribution" in title.lower():
                story.append(self._fig_to_image(fig))
                story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # ── Round 1: CodeBERT Fine-tuned ─────────────────────────────
        for sub_key, sub_label, sub_desc in [
                ("R1_binary",
                 "Round 1A — CodeBERT Fine-tuned  (Binary)",
                 "Code snippets → CodeBERT → [CLS] 768-dim → "
                 "Linear(768, 2) → Softmax → {Benign, Vulnerable}."),
                ("R1_multi",
                 "Round 1B — CodeBERT Fine-tuned  (Multi-class, 69 classes)",
                 "Code snippets → CodeBERT → [CLS] 768-dim → "
                 "Linear(768, 69) → Softmax → 69 CWE-ID classes.")]:
            if sub_key not in self.r:
                continue
            story.append(PageBreak())
            story.append(Paragraph(sub_label, S["SectionHead"]))
            story.append(Paragraph(sub_desc, S["Body"]))
            m = self.r[sub_key]
            story.append(Paragraph(f"{sub_label} — Metrics", S["SubHead"]))
            ft_tbl = [
                ["Metric", "Value"],
                ["Model",        m.get("model_name", CODEBERT_MODEL_NAME)],
                ["Architecture", "[CLS] 768 → Linear(768, 69) → Softmax"],
                ["Loss function","CrossEntropyLoss (class-weighted, C2 fix)"],
                ["Best epoch",   f"{m.get('best_epoch', '—')}  "
                                 f"(val-loss={m.get('best_val_loss','—')})"],
                ["Accuracy",     f"{m['accuracy']*100:.2f}%"],
                ["MCC",          f"{m['mcc']:.4f}"],
                ["F1-macro",     f"{m['f1_macro']:.4f}"],
                ["F1-weighted",  f"{m['f1_weighted']:.4f}"],
                ["Train samples",f"{m['n_train']:,}"],
                ["Test samples", f"{m['n_test']:,}"],
                ["Train time",   f"{m['train_time']:.1f} s"],
            ]
            story.append(self._table(ft_tbl, [8*cm, 8*cm]))
            conf_mat  = m.get("confusion_matrix")
            cm_labels = m.get("cm_labels")
            if conf_mat is not None and cm_labels and len(cm_labels) > 2:
                story.append(PageBreak())
                story.append(Paragraph(
                    f"{sub_label} — Confusion Matrix", S["SubHead"]))
                story.append(self._confusion_matrix_image(
                    conf_mat, cm_labels, f"{sub_label} — Confusion Matrix"))
                story.append(Paragraph(
                    f"Figure: {sub_label} confusion matrix "
                    f"({len(cm_labels)} classes, test set)", S["Caption"]))
                story.append(self._confusion_matrix_table(conf_mat, cm_labels))

        # ── Round 2: GraphCodeBERT Fine-tuned ────────────────────────
        for sub_key, sub_label, sub_desc in [
                ("R2_binary",
                 "Round 2A — GraphCodeBERT Fine-tuned  (Binary)",
                 "Code snippets → GraphCodeBERT → [CLS] 768-dim → "
                 "Linear(768, 2) → Softmax → {Benign, Vulnerable}."),
                ("R2_multi",
                 "Round 2B — GraphCodeBERT Fine-tuned  (Multi-class, 69 classes)",
                 "Code snippets → GraphCodeBERT → [CLS] 768-dim → "
                 "Linear(768, 69) → Softmax → 69 CWE-ID classes.")]:
            if sub_key not in self.r:
                continue
            story.append(PageBreak())
            story.append(Paragraph(sub_label, S["SectionHead"]))
            story.append(Paragraph(sub_desc, S["Body"]))
            m = self.r[sub_key]
            story.append(Paragraph(f"{sub_label} — Metrics", S["SubHead"]))
            ft_tbl = [
                ["Metric", "Value"],
                ["Model",        m.get("model_name", GRAPHCODEBERT_MODEL_NAME)],
                ["Architecture", "[CLS] 768 → Linear(768, 69) → Softmax"],
                ["Loss function","CrossEntropyLoss (class-weighted, C2 fix)"],
                ["Best epoch",   f"{m.get('best_epoch', '—')}  "
                                 f"(val-loss={m.get('best_val_loss','—')})"],
                ["Accuracy",     f"{m['accuracy']*100:.2f}%"],
                ["MCC",          f"{m['mcc']:.4f}"],
                ["F1-macro",     f"{m['f1_macro']:.4f}"],
                ["F1-weighted",  f"{m['f1_weighted']:.4f}"],
                ["Train samples",f"{m['n_train']:,}"],
                ["Test samples", f"{m['n_test']:,}"],
                ["Train time",   f"{m['train_time']:.1f} s"],
            ]
            story.append(self._table(ft_tbl, [8*cm, 8*cm]))
            conf_mat  = m.get("confusion_matrix")
            cm_labels = m.get("cm_labels")
            if conf_mat is not None and cm_labels and len(cm_labels) > 2:
                story.append(PageBreak())
                story.append(Paragraph(
                    f"{sub_label} — Confusion Matrix", S["SubHead"]))
                story.append(self._confusion_matrix_image(
                    conf_mat, cm_labels, f"{sub_label} — Confusion Matrix"))
                story.append(Paragraph(
                    f"Figure: {sub_label} confusion matrix "
                    f"({len(cm_labels)} classes, test set)", S["Caption"]))
                story.append(self._confusion_matrix_table(conf_mat, cm_labels))

        # ── Round 3: GraphCodeBERT + XGBoost (768-dim, no HC) ─────────
        r3a_keys = ["R3_noSMOTE", "R3_SMOTE"]
        r3b_keys = ["R3B_noSMOTE", "R3B_SMOTE"]
        r3a_avail = [k for k in r3a_keys if k in self.r]
        r3b_avail = [k for k in r3b_keys if k in self.r]
        if r3a_avail or r3b_avail:
            story.append(PageBreak())
            story.append(Paragraph(
                "Round 3 — GraphCodeBERT + XGBoost  "
                "(768-dim, no handcrafted features)", S["SectionHead"]))
            story.append(Paragraph(
                "GraphCodeBERT (frozen) → [CLS] 768-dim embeddings only "
                "— no handcrafted structural features. "
                "Round 3A: XGBoost binary classifier standalone "
                "(Benign / Vulnerable). "
                "Round 3B: Stage-1 binary gate → flat Stage-2 XGBoost "
                "multi-class classifier trained on ALL rows "
                "(69 Mtag classes, Benign included).",
                S["Body"]))

            if r3a_avail:
                story.append(Paragraph(
                    "Round 3A — Binary Classification (Standalone, 768-dim)",
                    S["SubHead"]))
                for key in r3a_avail:
                    lbl = "Without SMOTE" if "noSMOTE" in key else "With SMOTE"
                    t = self._metrics_table(key, lbl)
                    if t:
                        story.append(t)
                    story.append(Spacer(1, 0.2*cm))

            if r3b_avail:
                story.append(Paragraph(
                    "Round 3B — Multi-class Classification "
                    "(Flat, ALL rows, 768-dim)",
                    S["SubHead"]))
                for key in r3b_avail:
                    lbl = "Without SMOTE" if "noSMOTE" in key else "With SMOTE"
                    story.append(Paragraph(f"Round 3B — {lbl}", S["SubHead"]))
                    t = self._metrics_table(key, lbl)
                    if t:
                        story.append(t)
                    story.append(Spacer(1, 0.25*cm))
                    m = self.r.get(key, {})
                    conf_mat  = m.get("confusion_matrix")
                    cm_labels = m.get("cm_labels")
                    if conf_mat is not None and cm_labels:
                        story.append(PageBreak())
                        story.append(Paragraph(
                            f"Round 3B — {lbl} — Confusion Matrix",
                            S["SubHead"]))
                        story.append(self._confusion_matrix_image(
                            conf_mat, cm_labels,
                            f"Round 3B ({lbl}) — Confusion Matrix"))
                        story.append(Paragraph(
                            f"Figure: Confusion matrix — Round 3B, {lbl} "
                            f"({len(cm_labels)} classes, test set)",
                            S["Caption"]))
                        story.append(self._confusion_matrix_table(
                            conf_mat, cm_labels))
                        story.append(Spacer(1, 0.3*cm))

            # R3 comparison if both SMOTE variants ran
            all_r3 = r3b_avail if r3b_avail else r3a_avail
            if len(all_r3) >= 2:
                story.append(Paragraph("Round 3 — Comparison", S["SubHead"]))
                comp = [["Configuration", "Accuracy", "MCC",
                          "mF1 (macro)", "wF1 (weighted)"]]
                for k in all_r3:
                    m = self.r[k]
                    comp.append([
                        "No SMOTE" if "noSMOTE" in k else "With SMOTE",
                        f"{m['accuracy']*100:.2f}%",
                        f"{m.get('mcc', 0):.4f}",
                        f"{m['f1_macro']:.4f}",
                        f"{m['f1_weighted']:.4f}",
                    ])
                story.append(self._table(comp, [4.5*cm, 3*cm, 3*cm, 3*cm, 3*cm]))

        # ── Round 4: Two-Stage Hierarchical XGBoost (780-dim + HC) ───
        r4a_keys = ["R4_noSMOTE", "R4_SMOTE"]
        r4b_keys = ["R4B_noSMOTE", "R4B_SMOTE"]
        r4a_avail = [k for k in r4a_keys  if k in self.r]
        r4b_avail = [k for k in r4b_keys  if k in self.r]
        if r4a_avail or r4b_avail:
            story.append(PageBreak())
            story.append(Paragraph(
                "Round 4 — Two-Stage Hierarchical XGBoost  "
                "(GCB 780-dim + Handcrafted Features)", S["SectionHead"]))
            story.append(Paragraph(
                "GraphCodeBERT (frozen) → [CLS] 768-dim + 12-dim "
                "handcrafted structural features = 780-dim total. "
                "Round 4A: Stage-1 XGBoost binary gate standalone "
                "(Benign / Vulnerable). "
                "Round 4B: Stage-1 binary gate → Stage-2 multi-class XGBoost "
                "trained exclusively on Vulnerable rows (69 CWE-ID classes).",
                S["Body"]))

            if r4a_avail:
                story.append(Paragraph(
                    "Round 4A — Binary (Stage-1 Standalone, 780-dim)",
                    S["SubHead"]))
                for key in r4a_avail:
                    lbl = "Without SMOTE" if "noSMOTE" in key else "With SMOTE"
                    t = self._metrics_table(key, lbl)
                    if t:
                        story.append(t)
                    story.append(Spacer(1, 0.2*cm))

            if r4b_avail:
                story.append(Paragraph(
                    "Round 4B — Hierarchical Multi-class (Stage-1 + Stage-2, 780-dim)",
                    S["SubHead"]))
                for key in r4b_avail:
                    lbl = "Without SMOTE" if "noSMOTE" in key else "With SMOTE"
                    story.append(Paragraph(f"Round 4B — {lbl}", S["SubHead"]))
                    t = self._metrics_table(key, lbl)
                    if t:
                        story.append(t)
                    story.append(Spacer(1, 0.25*cm))
                    m = self.r.get(key, {})
                    conf_mat  = m.get("confusion_matrix")
                    cm_labels = m.get("cm_labels")
                    if conf_mat is not None and cm_labels:
                        story.append(PageBreak())
                        story.append(Paragraph(
                            f"Round 4B — {lbl} — Confusion Matrix",
                            S["SubHead"]))
                        story.append(self._confusion_matrix_image(
                            conf_mat, cm_labels,
                            f"Round 4B ({lbl}) — Confusion Matrix"))
                        story.append(Paragraph(
                            f"Figure: Confusion matrix — Round 4B, {lbl} "
                            f"({len(cm_labels)} classes, test set)",
                            S["Caption"]))
                        story.append(self._confusion_matrix_table(
                            conf_mat, cm_labels))
                        story.append(Spacer(1, 0.3*cm))

            # R4 comparison if both SMOTE variants ran
            all_r4 = r4b_avail if r4b_avail else r4a_avail
            if len(all_r4) >= 2:
                story.append(Paragraph("Round 4 — Comparison", S["SubHead"]))
                comp = [["Configuration", "Accuracy", "MCC",
                          "mF1 (macro)", "wF1 (weighted)"]]
                for k in all_r4:
                    m = self.r[k]
                    comp.append([
                        "No SMOTE" if "noSMOTE" in k else "With SMOTE",
                        f"{m['accuracy']*100:.2f}%",
                        f"{m.get('mcc', 0):.4f}",
                        f"{m['f1_macro']:.4f}",
                        f"{m['f1_weighted']:.4f}",
                    ])
                story.append(self._table(comp, [4.5*cm, 3*cm, 3*cm, 3*cm, 3*cm]))

        # Comparison charts
        for title, fig in self.figs:
            if "distribution" not in title.lower():
                story.append(Spacer(1, 0.3*cm))
                story.append(self._fig_to_image(fig))
                story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # Master results
        all_res_keys = [
            ("R1A CB-Binary",    "R1_binary"),
            ("R1B CB-Multi",     "R1_multi"),
            ("R2A GCB-Binary",   "R2_binary"),
            ("R2B GCB-Multi",    "R2_multi"),
            ("R3A Bin-NoSMOTE",  "R3_noSMOTE"),
            ("R3A Bin-SMOTE",    "R3_SMOTE"),
            ("R3B Flat-NoSMOTE", "R3B_noSMOTE"),
            ("R3B Flat-SMOTE",   "R3B_SMOTE"),
            ("R4A Hier-NoSMOTE", "R4_noSMOTE"),
            ("R4A Hier-SMOTE",   "R4_SMOTE"),
            ("R4B Hier-NoSMOTE", "R4B_noSMOTE"),
            ("R4B Hier-SMOTE",   "R4B_SMOTE"),
        ]
        avail = [(n, k) for n, k in all_res_keys if k in self.r]
        if len(avail) >= 2:
            story.append(Spacer(1, 0.4*cm))
            story.append(Paragraph("Master Results Table", S["SectionHead"]))
            mast = [["Configuration", "Accuracy", "MCC",
                      "mF1", "wF1", "Train N", "Test N",
                      "SMOTE", "Time (s)"]]
            for name, key in avail:
                m = self.r[key]
                mast.append([
                    name,
                    f"{m['accuracy']*100:.2f}%",
                    f"{m.get('mcc', m.get('f1_micro', 0)):.4f}",
                    f"{m['f1_macro']:.4f}",
                    f"{m['f1_weighted']:.4f}",
                    f"{m['n_train']:,}",
                    f"{m['n_test']:,}",
                    "Yes" if m["smote"] else "No",
                    f"{m['train_time']:.1f}",
                ])
            story.append(self._table(mast,
                [3*cm,2.2*cm,1.9*cm,1.8*cm,1.8*cm,
                 2.0*cm,1.8*cm,1.7*cm,1.8*cm]))

        story += [
            Spacer(1, 1*cm),
            HRFlowable(width="100%", thickness=1,
                       color=colors.HexColor("#BCC8D8")),
            Paragraph(
                "Generated by UAVulDetect — Multi-Round Vulnerability "
                "Detection Framework  (V16)  |  "
                f"Report date: {time.strftime('%Y-%m-%d')}",
                S["Caption"]),
        ]
        doc.build(story)
        return pdf_path


# ═══════════════════════════════════════════════════════════════════
#  GUI APPLICATION  (layout fully preserved from original)
# ═══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(
            "UAVulDetect  ·  Multi-Round Vulnerability Detection Framework  (V16)")
        self.geometry("1100x860")
        self.minsize(920, 740)
        self.configure(bg=DARK_BG)

        self._job_thread  = None
        self._stop_event  = threading.Event()
        self._log_queue   = queue.Queue()
        self._running     = False

        self._build_ui()
        self._poll_queue()

    # ── UI BUILD ────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self, bg=PANEL_BG, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="⚡", bg=PANEL_BG, fg=ACCENT_AMBER,
                 font=("Segoe UI Emoji", 22)).pack(side="left", padx=(18,4))
        tk.Label(top,
                 text="UAVulDetect — Multi-Round Vulnerability Detection  (V16)",
                 bg=PANEL_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(top, text="CB · GCB · XGBoost", bg=PANEL_BG,
                 fg=ACCENT_TEAL,
                 font=("Segoe UI", 10)).pack(side="left", padx=(8,0))

        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=10, pady=6)

        left = tk.Frame(body, bg=DARK_BG, width=360)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

        bar = tk.Frame(self, bg=PANEL_BG, height=32)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Consolas", 9)).pack(side="left", padx=12)

        self._progress_var = tk.DoubleVar()
        self._progress_lbl = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._progress_lbl,
                 bg=PANEL_BG, fg=ACCENT_TEAL,
                 font=("Consolas", 9)).pack(side="right", padx=6)
        pb = ttk.Progressbar(bar, variable=self._progress_var,
                              maximum=100, length=220, mode="determinate")
        pb.pack(side="right", padx=(0,4), pady=6)
        self._pb = pb

    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=DARK_BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical",
                            command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner  = tk.Frame(canvas, bg=DARK_BG)
        win_id = canvas.create_window((0,0), window=inner, anchor="nw")

        def on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())

        inner.bind("<Configure>", on_configure)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        def _on_wheel(e):
            canvas.yview_scroll(-1 * (e.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)

        self._build_dataset_card(inner)
        self._build_params_card(inner)
        self._build_graphcodebert_card(inner)   # replaces CodeBERT card
        self._build_xgboost_card(inner)          # replaces Bi-LSTM card
        self._build_smote_card(inner)
        self._build_tasks_card(inner)
        self._build_finetuning_card(inner)       # Round 4 & 5 hyperparams
        self._build_output_card(inner)
        self._build_buttons(inner)

    def _card(self, parent, title, icon=""):
        frame = tk.LabelFrame(parent,
                               text=f"  {icon}  {title}" if icon else f"  {title}",
                               bg=CARD_BG, fg=ACCENT_TEAL,
                               font=("Segoe UI", 10, "bold"),
                               bd=1, relief="flat",
                               highlightbackground=BORDER,
                               highlightthickness=1,
                               padx=10, pady=8)
        frame.pack(fill="x", padx=4, pady=4)
        return frame

    def _hparam_entry(self, c, label, var, bounds_txt, note):
        row = tk.Frame(c, bg=CARD_BG)
        row.pack(fill="x", pady=(3, 0))
        tk.Label(row, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), width=20,
                 anchor="w").pack(side="left")
        ent = tk.Entry(row, textvariable=var, width=10,
                       bg="#1E2233", fg=TEXT_PRIMARY,
                       insertbackground=TEXT_PRIMARY,
                       relief="flat", font=("Consolas", 9))
        ent.pack(side="left", ipady=3, padx=(0, 4))
        if bounds_txt:
            tk.Label(row, text=bounds_txt, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(side="left")
        tk.Label(c, text=note, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=300,
                 justify="left").pack(anchor="w", pady=(0, 6))

    def _hparam_combo(self, c, label, var, values, note):
        row = tk.Frame(c, bg=CARD_BG)
        row.pack(fill="x", pady=(3, 0))
        tk.Label(row, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), width=20,
                 anchor="w").pack(side="left")
        combo = ttk.Combobox(row, textvariable=var,
                              values=values, width=10, state="readonly")
        combo.pack(side="left", padx=(0, 4))
        tk.Label(c, text=note, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=300,
                 justify="left").pack(anchor="w", pady=(0, 6))

    # ── Dataset card ────────────────────────────────────────────
    def _build_dataset_card(self, parent):
        c = self._card(parent, "Dataset", "📂")

        # ── Section label ────────────────────────────────────────
        tk.Label(c, text="Primary dataset  (CSV / XLSX with SNIPPET column):",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        # ── Row 1: main dataset path + Browse ───────────────────
        path_row = tk.Frame(c, bg=CARD_BG)
        path_row.pack(fill="x", pady=(2, 0))
        self._csv_var = tk.StringVar()
        ent = tk.Entry(path_row, textvariable=self._csv_var,
                       bg="#1E2233", fg=TEXT_PRIMARY,
                       insertbackground=TEXT_PRIMARY,
                       relief="flat", font=("Consolas", 9))
        ent.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 4))

        self._use_precomputed_var    = tk.BooleanVar(value=False)
        self._use_cb_precomputed_var = tk.BooleanVar(value=False)

        def browse_main():
            p = filedialog.askopenfilename(
                title="Select primary dataset (CSV / XLSX)",
                filetypes=[("CSV / Excel", "*.csv *.xlsx *.xls"),
                           ("All files", "*.*")])
            if p:
                self._csv_var.set(p)

        tk.Button(path_row, text="Browse …", command=browse_main,
                  bg=BTN_BROWSE, fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  padx=8, pady=3).pack(side="left")

        tk.Label(c,
                 text="Select UAVulDB01.csv / .xlsx with SNIPPET, FILE TYPE, "
                      "Mtags, Btags columns.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=310,
                 justify="left").pack(anchor="w", pady=(2, 6))

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=4)

        # ── Section label: precomputed feature files ─────────────
        tk.Label(c,
                 text="Precomputed feature datasets  (skip embedding extraction):",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        # ── Row 2: GraphCodeBERT precomputed path ───────────────
        gcb_row = tk.Frame(c, bg=CARD_BG)
        gcb_row.pack(fill="x", pady=(4, 0))
        self._gcb_pre_var = tk.StringVar()

        gcb_chk = tk.Checkbutton(
            gcb_row,
            text="Use GraphCodeBERT precomputed features:",
            variable=self._use_precomputed_var,
            bg=CARD_BG, fg=ACCENT_TEAL, selectcolor=DARK_BG,
            activebackground=CARD_BG, activeforeground=ACCENT_TEAL,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2")
        gcb_chk.pack(anchor="w")

        gcb_path_row = tk.Frame(c, bg=CARD_BG)
        gcb_path_row.pack(fill="x", pady=(1, 0))
        gcb_ent = tk.Entry(gcb_path_row, textvariable=self._gcb_pre_var,
                           bg="#1E2233", fg=TEXT_PRIMARY,
                           insertbackground=TEXT_PRIMARY,
                           relief="flat", font=("Consolas", 9))
        gcb_ent.pack(side="left", fill="x", expand=True, ipady=3, padx=(14, 4))

        def browse_gcb_pre():
            p = filedialog.askopenfilename(
                title="Select GCB precomputed feature file (.xlsx)",
                filetypes=[("Excel files", "*.xlsx *.xls"),
                           ("All files", "*.*")])
            if p:
                self._gcb_pre_var.set(p)
                self._use_precomputed_var.set(True)

        tk.Button(gcb_path_row, text="Browse …", command=browse_gcb_pre,
                  bg=ACCENT_TEAL, fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  padx=8, pady=2).pack(side="left")

        tk.Label(c,
                 text="  ← GCB 768-dim + 12-dim HC = 780-dim feature file "
                      "(exported by a previous run as "
                      "'<name>_GraphCodeBERT_Features.xlsx'). "
                      "Skips Step 4 GCB extraction.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=310,
                 justify="left").pack(anchor="w", pady=(0, 6))

        # ── Row 3: CodeBERT precomputed path ────────────────────
        cb_row = tk.Frame(c, bg=CARD_BG)
        cb_row.pack(fill="x", pady=(4, 0))
        self._cb_pre_var = tk.StringVar()

        cb_chk = tk.Checkbutton(
            cb_row,
            text="Use CodeBERT precomputed features:",
            variable=self._use_cb_precomputed_var,
            bg=CARD_BG, fg=ACCENT_BLUE, selectcolor=DARK_BG,
            activebackground=CARD_BG, activeforeground=ACCENT_BLUE,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2")
        cb_chk.pack(anchor="w")

        cb_path_row = tk.Frame(c, bg=CARD_BG)
        cb_path_row.pack(fill="x", pady=(1, 0))
        cb_ent = tk.Entry(cb_path_row, textvariable=self._cb_pre_var,
                          bg="#1E2233", fg=TEXT_PRIMARY,
                          insertbackground=TEXT_PRIMARY,
                          relief="flat", font=("Consolas", 9))
        cb_ent.pack(side="left", fill="x", expand=True, ipady=3, padx=(14, 4))

        def browse_cb_pre():
            p = filedialog.askopenfilename(
                title="Select CodeBERT precomputed feature file (.xlsx)",
                filetypes=[("Excel files", "*.xlsx *.xls"),
                           ("All files", "*.*")])
            if p:
                self._cb_pre_var.set(p)
                self._use_cb_precomputed_var.set(True)

        tk.Button(cb_path_row, text="Browse …", command=browse_cb_pre,
                  bg=ACCENT_BLUE, fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  padx=8, pady=2).pack(side="left")

        tk.Label(c,
                 text="  ← CB 768-dim feature file "
                      "(exported by a previous run as "
                      "'<name>_CodeBERT_Features.xlsx'). "
                      "If valid, skips the CB feature-export pass at the "
                      "end of the run. NOTE: Round 1 fine-tuning always "
                      "re-tokenises raw code text internally and cannot "
                      "use static embeddings — this setting only affects "
                      "the separate export step, not Round 1 training.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=310,
                 justify="left").pack(anchor="w", pady=(0, 6))

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=4)

        # ── Export checkboxes ────────────────────────────────────
        tk.Label(c,
                 text="Create precomputed feature datasets after extraction:",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        # M2 fix: CB export now defaults to OFF. It performs a full extra
        # CodeBERT corpus pass independent of which rounds are selected
        # (e.g. it still ran by default even for XGBoost-only R3/R4 runs)
        # — opt-in avoids surprising users with unexpected runtime cost.
        # GCB export stays ON by default since GCB extraction already runs
        # for most round combinations (R2/R3/R4), making its export nearly
        # free as a byproduct rather than an extra pass.
        self._export_cb_pre_var  = tk.BooleanVar(value=False)
        self._export_gcb_pre_var = tk.BooleanVar(value=True)

        for txt, var, clr, note in [
                ("Create precomputed features dataset from CodeBERT embedding",
                 self._export_cb_pre_var, ACCENT_BLUE,
                 "Runs a SEPARATE full-corpus CodeBERT forward pass "
                 "(independent of Round 1/2) and exports "
                 "'<name>_CodeBERT_Features.xlsx'. Off by default — adds "
                 "runtime even if Rounds 1/2 are not selected."),
                ("Create precomputed features dataset from GraphCodeBERT embedding",
                 self._export_gcb_pre_var, ACCENT_TEAL,
                 "Exports '<name>_GraphCodeBERT_Features.xlsx' after Step 4 "
                 "GCB extraction — reuse in future runs to skip GCB "
                 "embedding re-extraction (Rounds 2, 3, 4).")]:
            tk.Checkbutton(c, text=txt, variable=var,
                           bg=CARD_BG, fg=clr, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=clr,
                           font=("Segoe UI", 9, "bold"),
                           wraplength=300, justify="left", anchor="w",
                           cursor="hand2").pack(anchor="w", fill="x",
                                                pady=(3, 0))
            tk.Label(c, text=f"  {note}",
                     bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 8), wraplength=310,
                     justify="left").pack(anchor="w", pady=(0, 4))

    # ── Experiment parameters card ───────────────────────────────
    def _build_params_card(self, parent):
        c = self._card(parent, "Experiment Parameters", "⚙️")

        self._n_samples_var  = tk.StringVar(value="40000")
        self._test_size_var  = tk.StringVar(value="20")

        for label, var, mn, mx in [
                ("Training samples",  self._n_samples_var,  100, 349000),
                ("Test set size (%)", self._test_size_var,  5,   50)]:
            row = tk.Frame(c, bg=CARD_BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), width=20,
                     anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=12,
                           bg="#1E2233", fg=TEXT_PRIMARY,
                           insertbackground=TEXT_PRIMARY,
                           relief="flat", font=("Consolas", 9))
            ent.pack(side="left", ipady=3, padx=(0,4))
            tk.Label(row, text=f"[{mn}–{mx}]",
                     bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(side="left")

        tk.Label(c, text="Train/test split is stratified by binary class "
                 "(Benign/Vulnerable) and performed BEFORE any SMOTE balancing.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=310,
                 justify="left").pack(anchor="w", pady=(2, 6))

        self._shuffle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Shuffle dataset before split",
                       variable=self._shuffle_var,
                       bg=CARD_BG, fg=ACCENT_BLUE, selectcolor=DARK_BG,
                       activebackground=CARD_BG,
                       activeforeground=ACCENT_BLUE,
                       font=("Segoe UI", 9, "bold"),
                       cursor="hand2").pack(anchor="w", pady=(2,0))
        tk.Label(c, text="ON (recommended): stratified random split. "
                 "OFF: sequential split preserving file order.",
                 bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8),
                 wraplength=310, justify="left").pack(anchor="w", pady=(2,0))

    # ── GraphCodeBERT card (replaces CodeBERT card) ──────────────
    def _build_graphcodebert_card(self, parent):
        c = self._card(parent, "GraphCodeBERT Hyperparameters", "🧠")

        self._max_len_var     = tk.StringVar(value="512")
        self._pooling_var     = tk.StringVar(value="cls")
        self._embed_batch_var = tk.StringVar(value="32")

        self._hparam_entry(
            c, "Max token length", self._max_len_var, "[16–512]",
            "Recommended: 512. GraphCodeBERT's full context window. "
            "Analysis of UAVulDB01 shows C snippets average ~126 chars "
            "but Java samples reach 1,357 chars; 512 sub-word tokens "
            "covers ≥95% of snippets without truncation.")

        self._hparam_combo(
            c, "Pooling strategy", self._pooling_var,
            ["cls", "mean", "max"],
            "Recommended: cls. The [CLS] token in GraphCodeBERT is "
            "pre-trained to aggregate whole-sequence code semantics "
            "including data-flow graph structure — ideal for "
            "function-level vulnerability classification. "
            "Mean/max are available as ablation baselines.")

        self._hparam_entry(
            c, "Embed batch size", self._embed_batch_var, "[8–128]",
            "Recommended: 32. Batch size for the GraphCodeBERT forward "
            "pass. Reduce to 8 if GPU VRAM < 6 GB; increase to 64 on "
            "a 16 GB+ GPU for faster extraction.")

    # ── XGBoost card (replaces Bi-LSTM card) ─────────────────────
    def _build_xgboost_card(self, parent):
        c = self._card(parent, "XGBoost Hyperparameters", "🌲")

        self._n_estimators_var = tk.StringVar(value="400")
        self._max_depth_var    = tk.StringVar(value="7")
        self._xgb_lr_var       = tk.StringVar(value="0.05")
        self._subsample_var    = tk.StringVar(value="0.8")
        self._colsample_var    = tk.StringVar(value="0.8")
        self._early_stop_var   = tk.StringVar(value="30")

        self._hparam_entry(
            c, "n_estimators", self._n_estimators_var, "[50–2000]",
            "Recommended: 400. Number of boosting rounds. Combined with "
            "early stopping (below), larger values are safe — training "
            "stops automatically when validation loss plateaus.")

        self._hparam_entry(
            c, "max_depth", self._max_depth_var, "[3–12]",
            "Recommended: 7. Tree depth for Stage-1 (binary) and Stage-2 "
            "(multi-class) heads. Deeper trees capture more complex "
            "feature interactions in the 780-dim embedding space. "
            "Reduce to 5 if overfitting on small samples (< 5k).")

        self._hparam_entry(
            c, "Learning rate", self._xgb_lr_var, "[0.01–0.3]",
            "Recommended: 0.05. Shrinkage per step. Lower values "
            "(0.01–0.05) with more estimators generalise better; "
            "increase to 0.1–0.2 to speed up exploratory runs.")

        self._hparam_entry(
            c, "Subsample", self._subsample_var, "[0.5–1.0]",
            "Recommended: 0.8. Fraction of rows sampled per tree — "
            "acts as stochastic regularisation. 0.8 balances bias "
            "and variance on the 780-dim feature matrix.")

        self._hparam_entry(
            c, "Colsample/tree", self._colsample_var, "[0.5–1.0]",
            "Recommended: 0.8. Fraction of features sampled per tree. "
            "Reduces correlation between trees and improves robustness "
            "on the 68 GraphCodeBERT embedding dimensions that dominate "
            "SHAP importance.")

        self._hparam_entry(
            c, "Early-stop rounds", self._early_stop_var, "[5–100]",
            "Recommended: 30. Training stops if validation logloss/"
            "mlogloss doesn't improve for this many consecutive rounds. "
            "A 10% internal validation split is used — never the held-out "
            "test partition.")

    # ── SMOTE card (unchanged labels/logic) ─────────────────────
    def _build_smote_card(self, parent):
        c = self._card(parent, "SMOTE Hyperparameters", "🎛️")

        self._smote_k_var      = tk.StringVar(value="5")
        self._smote_cap_var    = tk.StringVar(value="1200")
        self._smote_target_var = tk.StringVar(value="median")

        self._hparam_entry(
            c, "k-neighbours", self._smote_k_var, "[1–10]",
            "Recommended: 5. The smallest CWE class in UAVulDB01 has "
            "~3,000 samples — well above k+1=6 required for k=5 "
            "nearest-neighbour interpolation in 780-dim space.")

        self._hparam_entry(
            c, "Max synthetic/class", self._smote_cap_var, "[100–5000]",
            "Recommended: 1200. Caps synthetic samples per minority "
            "CWE class — prevents the rarest categories from "
            "disproportionately inflating the Stage-2 training set.")

        self._hparam_combo(
            c, "Balance target", self._smote_target_var,
            ["median", "mean", "max"],
            "Recommended: median. With 68 CWE classes ranging from "
            "3,000 to 9,371 samples, the median (~4,500) avoids "
            "excessive synthesis while lifting all minority classes.")

    # ── Tasks card ────────────────────────────────────────────────
    def _build_tasks_card(self, parent):
        c = self._card(parent, "Classification Tasks", "🎯")

        # Fine-tuned BERT rounds (R1 CB, R2 GCB)
        self._do_r1_ft_var    = tk.BooleanVar(value=True)
        self._do_r2_ft_var    = tk.BooleanVar(value=True)
        # XGBoost flat ALL-rows round (R3)
        self._do_flat_var     = tk.BooleanVar(value=True)
        # XGBoost hierarchical round (R4)
        self._do_binary_var   = tk.BooleanVar(value=True)
        # SMOTE strategies for XGBoost rounds (R3, R4)
        self._do_smote_var    = tk.BooleanVar(value=True)
        self._do_no_smote_var = tk.BooleanVar(value=True)
        # compat stub — not used in V09
        self._do_multi_var    = tk.BooleanVar(value=False)

        task_frame = tk.Frame(c, bg=CARD_BG)
        task_frame.pack(fill="x", pady=2)

        for txt, var, clr in [
                ("Round 1 — CodeBERT Fine-tuned\n"
                 "  Arch: [CLS] 768 \u2192 Linear(768\u219269) \u2192 Softmax\n"
                 "  1A: Binary (Benign/Vuln)  |  1B: Multi (69 classes)\n"
                 "  Model: microsoft/codebert-base\n"
                 "  \u26a0 Requires raw SNIPPET column  \xb7  GPU recommended",
                 self._do_r1_ft_var,  ACCENT_TEAL),
                ("Round 2 — GraphCodeBERT Fine-tuned\n"
                 "  Arch: [CLS] 768 \u2192 Linear(768\u219269) \u2192 Softmax\n"
                 "  2A: Binary (Benign/Vuln)  |  2B: Multi (69 classes)\n"
                 "  Model: microsoft/graphcodebert-base\n"
                 "  \u26a0 Requires raw SNIPPET column  \xb7  GPU recommended",
                 self._do_r2_ft_var,  ACCENT_BLUE),
                ("Round 3 — GraphCodeBERT + XGBoost  (768-dim)\n"
                 "  GCB frozen \u2192 [CLS] 768-dim, no handcrafted features\n"
                 "  3A: Binary standalone  (768-dim)\n"
                 "  3B: Stage-1 gate + flat Stage-2 multi-class (69 cls)",
                 self._do_flat_var,   ACCENT_AMBER),
                ("Round 4 — Two-Stage Hierarchical XGBoost  (780-dim)\n"
                 "  GCB frozen \u2192 [CLS] 768-dim + 12-dim HC = 780-dim\n"
                 "  4A: Stage-1 binary standalone  (780-dim)\n"
                 "  4B: Stage-1 gate + Stage-2 multi-class (Vuln only)",
                 self._do_binary_var, "#B98AE6")]:
            tk.Checkbutton(task_frame, text=txt, variable=var,
                           bg=CARD_BG, fg=clr, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=clr,
                           font=("Segoe UI", 9, "bold"),
                           wraplength=258, justify="left", anchor="w",
                           cursor="hand2").pack(anchor="w", fill="x",
                                                pady=(2, 6))

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(c, text="Balancing strategies (Rounds 3 & 4, XGBoost only):",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        for txt, var, clr in [
                ("Without SMOTE",  self._do_no_smote_var, ACCENT_AMBER),
                ("With SMOTE",     self._do_smote_var,    ACCENT_TEAL)]:
            tk.Checkbutton(c, text=txt, variable=var,
                           bg=CARD_BG, fg=clr, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=clr,
                           font=("Segoe UI", 9, "bold"),
                           wraplength=300, justify="left", anchor="w",
                           cursor="hand2").pack(anchor="w", fill="x", pady=1)

    # ── Fine-tuning card (Rounds 1 & 2) ──────────────────────────
    def _build_finetuning_card(self, parent):
        c = self._card(parent, "Fine-tuning Hyperparameters (R1 & R2)", "🔬")

        self._ft_epochs_var  = tk.StringVar(value=str(FT_EPOCHS_DEFAULT))
        self._ft_lr_var      = tk.StringVar(value=str(FT_LR_DEFAULT))
        self._ft_batch_var   = tk.StringVar(value=str(FT_BATCH_DEFAULT))
        self._ft_max_len_var = tk.StringVar(value=str(FT_MAX_LEN_DEFAULT))

        self._hparam_entry(
            c, "Epochs", self._ft_epochs_var, "[1–20]",
            "Default: 5. Number of full fine-tuning passes. "
            "3–5 is typical for code classification tasks.")
        self._hparam_entry(
            c, "Learning rate", self._ft_lr_var, "[1e-6–5e-5]",
            "Default: 2e-5. AdamW LR with linear warmup + decay schedule.")
        self._hparam_entry(
            c, "Batch size", self._ft_batch_var, "[4–64]",
            "Default: 32. Reduce to 8–16 if GPU OOM.")
        self._hparam_entry(
            c, "Max token length", self._ft_max_len_var, "[32–512]",
            "Default: 512. Truncates code beyond this token count.")

        tk.Label(c,
                 text="Round 1 backbone : CodeBERT  "
                      "(microsoft/codebert-base)\n"
                      "Round 2 backbone : GraphCodeBERT  "
                      "(microsoft/graphcodebert-base)\n"
                      "Architecture     : [CLS] 768 → Linear(69) → Softmax\n"
                      "\n"
                      "⚠  GPU strongly recommended for Rounds 1 & 2.\n"
                      "   On CPU, max_len is auto-reduced to 128 and\n"
                      "   batch_size to 8 to prevent hangs / OOM.\n"
                      "   Progress is logged every 10% of each epoch.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=310,
                 justify="left").pack(anchor="w", pady=(4, 0))

    # ── Output card (unchanged) ──────────────────────────────────
    def _build_output_card(self, parent):
        c = self._card(parent, "Output Options", "📄")
        self._save_pdf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Generate PDF Report",
                       variable=self._save_pdf_var,
                       bg=CARD_BG, fg=ACCENT_AMBER,
                       selectcolor=DARK_BG, activebackground=CARD_BG,
                       font=("Segoe UI", 9, "bold"),
                       cursor="hand2").pack(anchor="w")
        tk.Label(c, text="PDF saved in the same folder as the dataset.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2,0))

    # ── Execute / Stop buttons (unchanged) ───────────────────────
    def _build_buttons(self, parent):
        bf = tk.Frame(parent, bg=DARK_BG)
        bf.pack(fill="x", padx=4, pady=8)
        self._btn_exec = tk.Button(
            bf, text="▶  Execute",
            command=self._start_run,
            bg=BTN_EXEC, fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=16, pady=8, width=13)
        self._btn_exec.pack(side="left", padx=(0,6))
        self._btn_stop = tk.Button(
            bf, text="⬛  Stop",
            command=self._stop_run,
            bg=BTN_STOP, fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=16, pady=8, width=10,
            state="disabled")
        self._btn_stop.pack(side="left")

    # ── Right panel: log box (unchanged) ────────────────────────
    def _build_right(self, parent):
        hdr = tk.Frame(parent, bg=PANEL_BG, pady=5)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📋  Execution Log",
                 bg=PANEL_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=10)
        tk.Button(hdr, text="Clear", command=self._clear_log,
                  bg=PANEL_BG, fg=TEXT_MUTED,
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9)).pack(side="right", padx=8)

        self._log_box = scrolledtext.ScrolledText(
            parent, bg="#0E1120", fg=TEXT_PRIMARY,
            font=("Consolas", 9), relief="flat",
            wrap="word", state="disabled",
            insertbackground=TEXT_PRIMARY,
            selectbackground=ACCENT_BLUE,
            padx=10, pady=8)
        self._log_box.pack(fill="both", expand=True)

        self._log_box.tag_config("info",  foreground=TEXT_PRIMARY)
        self._log_box.tag_config("warn",  foreground=ACCENT_AMBER)
        self._log_box.tag_config("error", foreground=ACCENT_RED)
        self._log_box.tag_config("ok",    foreground=ACCENT_GREEN)
        self._log_box.tag_config("head",  foreground=ACCENT_TEAL,
                                 font=("Consolas", 9, "bold"))

        self._result_frame = tk.Frame(parent, bg=PANEL_BG)
        self._result_frame.pack(fill="x")

    # ── Log helpers ──────────────────────────────────────────────
    def _log(self, msg, tag="info"):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n", tag)
        self._log_box.configure(state="disabled")
        self._log_box.see("end")

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    # ── Queue polling (unchanged) ────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg  = self._log_queue.get_nowait()
                kind = msg.get("type")

                if kind == "log":
                    lvl  = msg.get("level", "info")
                    text = msg.get("msg", "")
                    if "━" in text or "STEP" in text or "ROUND" in text:
                        tag = "head"
                    elif lvl == "warn":
                        tag = "warn"
                    elif lvl == "error":
                        tag = "error"
                    elif "✅" in text or "complete" in text.lower():
                        tag = "ok"
                    else:
                        tag = "info"
                    self._log(text, tag)

                elif kind == "progress":
                    pct   = msg.get("pct", 0)
                    label = msg.get("label", "")
                    self._progress_var.set(pct)
                    self._progress_lbl.set(label)
                    self._status_var.set(f"{pct:.0f}%  {label}")

                elif kind == "done":
                    self._on_done(msg.get("results", {}))

                elif kind == "stopped":
                    self._on_stopped()

                elif kind == "error":
                    self._on_error(msg.get("msg", "Unknown error"))

        except queue.Empty:
            pass
        finally:
            self.after(120, self._poll_queue)

    # ── Run control ──────────────────────────────────────────────
    def _validate_params(self):
        use_pre    = self._use_precomputed_var.get()
        gcb_pre    = self._gcb_pre_var.get().strip()

        # C2 fix: fall back to the dedicated GCB precomputed path when the
        # primary dataset field is empty and the user is in precomputed
        # mode — using the GCB-specific "Browse…" button alone (without
        # also filling the primary entry) must be a valid, self-sufficient
        # workflow rather than failing validation.
        csv = self._csv_var.get().strip()
        if not csv and use_pre and gcb_pre:
            csv = gcb_pre
            self._csv_var.set(gcb_pre)   # keep the two fields in sync

        if not csv:
            messagebox.showerror(
                "Missing dataset",
                "Please select a precomputed feature dataset (.xlsx) — "
                "either in the primary field or via the GraphCodeBERT "
                "'Browse…' button."
                if use_pre else
                "Please select a CSV dataset file.")
            return None
        if not os.path.isfile(csv):
            messagebox.showerror("File not found",
                                 f"Cannot find:\n{csv}")
            return None
        if not (self._do_r1_ft_var.get() or self._do_r2_ft_var.get()
                or self._do_flat_var.get() or self._do_binary_var.get()):
            messagebox.showerror("No task selected",
                                 "Select at least one classification task.")
            return None
        if not (self._do_smote_var.get() or self._do_no_smote_var.get()):
            messagebox.showerror("No balancing strategy",
                                 "Select at least one balancing strategy.")
            return None

        # S3 fix: GCB precomputed files carry no raw SNIPPET text, so
        # Round 1 / Round 2 (fine-tuned, which require raw code) cannot
        # run when precomputed mode is active. Warn BEFORE the run starts
        # rather than silently disabling the rounds mid-pipeline.
        if use_pre and (self._do_r1_ft_var.get() or self._do_r2_ft_var.get()):
            proceed = messagebox.askyesno(
                "Fine-tuned rounds will be skipped",
                "GraphCodeBERT precomputed feature files do not include "
                "raw code text, so Round 1 (CodeBERT) and/or Round 2 "
                "(GraphCodeBERT) fine-tuning cannot run in this mode and "
                "will be skipped automatically.\n\n"
                "Continue with only the selected XGBoost rounds "
                "(Round 3 / Round 4)?")
            if not proceed:
                return None

        try:
            p = {
                "csv_path":    csv,
                # GCB precomputed path (for Step 4 bypass)
                "gcb_pre_path": self._gcb_pre_var.get().strip(),
                "use_precomputed_features": self._use_precomputed_var.get(),
                # CB precomputed path (for future CB frozen-embedding round)
                "cb_pre_path":  self._cb_pre_var.get().strip(),
                "use_cb_precomputed_features": self._use_cb_precomputed_var.get(),
                # Export flags
                "export_cb_pre":  self._export_cb_pre_var.get(),
                "export_gcb_pre": self._export_gcb_pre_var.get(),
                "n_samples":   int(self._n_samples_var.get()),
                "max_len":     int(self._max_len_var.get()),
                "embed_batch": int(self._embed_batch_var.get()),
                "pooling":     self._pooling_var.get(),
                # XGBoost
                "n_estimators": int(self._n_estimators_var.get()),
                "max_depth":    int(self._max_depth_var.get()),
                "xgb_lr":       float(self._xgb_lr_var.get()),
                "subsample":    float(self._subsample_var.get()),
                "colsample":    float(self._colsample_var.get()),
                "early_stop":   int(self._early_stop_var.get()),
                # Tasks
                "do_r1_ft":     self._do_r1_ft_var.get(),
                "do_r2_ft":     self._do_r2_ft_var.get(),
                "do_flat":      self._do_flat_var.get(),
                "do_multi":     False,   # not used in V13 (removed in V09)
                "do_binary":    self._do_binary_var.get(),
                "do_smote":     self._do_smote_var.get(),
                "do_no_smote":  self._do_no_smote_var.get(),
                "save_pdf":     self._save_pdf_var.get(),
                # Dataset
                "test_size":    float(self._test_size_var.get()) / 100.0,
                "shuffle":      self._shuffle_var.get(),
                # SMOTE
                "smote_k":      int(self._smote_k_var.get()),
                "smote_cap":    int(self._smote_cap_var.get()),
                "smote_target": self._smote_target_var.get(),
                # Fine-tuning (Rounds 1 & 2)
                "ft_epochs":    int(self._ft_epochs_var.get()),
                "ft_lr":        float(self._ft_lr_var.get()),
                "ft_batch":     int(self._ft_batch_var.get()),
                "ft_max_len":   int(self._ft_max_len_var.get()),
            }
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return None

        if not (0.05 <= p["test_size"] <= 0.5):
            messagebox.showerror("Invalid parameter",
                                 "Test set size (%) must be between 5 and 50.")
            return None
        return p

    def _start_run(self):
        if self._running:
            return
        params = self._validate_params()
        if params is None:
            return
        for w in self._result_frame.winfo_children():
            w.destroy()

        self._running = True
        self._stop_event.clear()
        self._progress_var.set(0)
        self._progress_lbl.set("")
        self._btn_exec.configure(state="disabled", bg="#2A4A8A")
        self._btn_stop.configure(state="normal")
        self._status_var.set("Running …")
        self._log("═" * 60, "head")
        self._log("  GraphCodeBERT + Two-Stage XGBoost "
                  "Vulnerability Detection", "head")
        self._log("═" * 60, "head")

        engine = MLEngine(params, self._log_queue, self._stop_event)
        self._job_thread = threading.Thread(
            target=engine.run, daemon=True, name="MLEngine")
        self._job_thread.start()

    def _stop_run(self):
        if not self._running:
            return
        self._stop_event.set()
        self._status_var.set("Stopping …")
        self._btn_stop.configure(state="disabled")

    def _on_done(self, results):
        self._running = False
        self._btn_exec.configure(state="normal", bg=BTN_EXEC)
        self._btn_stop.configure(state="disabled")
        self._status_var.set("✅  Done")
        self._progress_var.set(100)
        self._progress_lbl.set("Complete ✓")
        self._show_results(results)
        if "pdf_path" in results:
            if messagebox.askyesno(
                    "PDF Report",
                    f"PDF saved to:\n{results['pdf_path']}\n\n"
                    "Open the containing folder?"):
                folder = os.path.dirname(results["pdf_path"])
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')

    def _on_stopped(self):
        self._running = False
        self._btn_exec.configure(state="normal", bg=BTN_EXEC)
        self._btn_stop.configure(state="disabled")
        self._status_var.set("⛔  Stopped")

    def _on_error(self, msg):
        self._running = False
        self._btn_exec.configure(state="normal", bg=BTN_EXEC)
        self._btn_stop.configure(state="disabled")
        self._status_var.set("❌  Error")

        # Bug 2 fix: for long error messages (e.g. HuggingFace model-load
        # failures with offline instructions), use a scrollable text dialog
        # instead of a plain messagebox which clips the text.
        if len(msg) > 300:
            dlg = tk.Toplevel(self)
            dlg.title("Pipeline error")
            dlg.configure(bg=DARK_BG)
            dlg.geometry("680x380")
            dlg.resizable(True, True)
            dlg.grab_set()

            tk.Label(dlg, text="❌  Pipeline error",
                     bg=DARK_BG, fg=ACCENT_RED,
                     font=("Segoe UI", 11, "bold")).pack(
                     anchor="w", padx=14, pady=(12, 4))

            frame = tk.Frame(dlg, bg=DARK_BG)
            frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
            sb = tk.Scrollbar(frame)
            sb.pack(side="right", fill="y")
            txt = tk.Text(frame, bg="#1E2233", fg="#E8EAF6",
                          font=("Consolas", 9), wrap="word",
                          yscrollcommand=sb.set, relief="flat",
                          padx=8, pady=8)
            txt.pack(side="left", fill="both", expand=True)
            sb.config(command=txt.yview)
            txt.insert("1.0", msg)
            txt.configure(state="disabled")

            tk.Button(dlg, text="  OK  ", command=dlg.destroy,
                      bg=BTN_EXEC, fg="white",
                      font=("Segoe UI", 9, "bold"),
                      relief="flat", cursor="hand2",
                      padx=12, pady=4).pack(pady=(0, 12))
        else:
            messagebox.showerror("Pipeline error", msg)

    # ── Results panel (structure preserved, labels updated) ──────
    def _show_results(self, r):
        for w in self._result_frame.winfo_children():
            w.destroy()

        hdr = tk.Frame(self._result_frame, bg=PANEL_BG, pady=4)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📊  Results Summary",
                 bg=PANEL_BG, fg=ACCENT_TEAL,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=10)

        grid = tk.Frame(self._result_frame, bg=PANEL_BG)
        grid.pack(fill="x", padx=8, pady=4)

        keys = [
            ("R1A — CB Binary (FT)",        "R1_binary",   ACCENT_TEAL),
            ("R1B — CB Multi (FT)",         "R1_multi",    "#16A085"),
            ("R2A — GCB Binary (FT)",       "R2_binary",   ACCENT_BLUE),
            ("R2B — GCB Multi (FT)",        "R2_multi",    "#2980B9"),
            ("R3A — No SMOTE (Binary)",     "R3_noSMOTE",  ACCENT_AMBER),
            ("R3A — SMOTE (Binary)",        "R3_SMOTE",    "#E67E22"),
            ("R3B — No SMOTE (Flat Multi)", "R3B_noSMOTE", "#D68910"),
            ("R3B — SMOTE (Flat Multi)",    "R3B_SMOTE",   "#B9770E"),
            ("R4A — No SMOTE (Hier.Bin)",   "R4_noSMOTE",  "#B98AE6"),
            ("R4A — SMOTE (Hier.Bin)",      "R4_SMOTE",    "#8957E5"),
            ("R4B — No SMOTE (Hier.Multi)", "R4B_noSMOTE", ACCENT_GREEN),
            ("R4B — SMOTE (Hier.Multi)",    "R4B_SMOTE",   "#27AE60"),
        ]
        col = 0
        row = 0
        max_cols = 3
        for label, key, clr in keys:
            if key not in r:
                continue
            m = r[key]
            card = tk.Frame(grid, bg=CARD_BG, bd=0,
                            highlightbackground=clr,
                            highlightthickness=1)
            card.grid(row=row, column=col, padx=4, pady=2, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            tk.Label(card, text=label, bg=CARD_BG, fg=clr,
                     font=("Segoe UI", 8, "bold"),
                     pady=3).pack()
            tk.Label(card,
                     text=f"{m['accuracy']*100:.2f}%",
                     bg=CARD_BG, fg=TEXT_PRIMARY,
                     font=("Segoe UI", 14, "bold")).pack()
            mcc_val = m.get('mcc', m.get('f1_micro', 0))
            tk.Label(card,
                     text=f"MCC {mcc_val:.4f}  mF1 {m['f1_macro']:.4f}",
                     bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 7.5),
                     pady=2).pack()
            col += 1
            if col >= max_cols:
                col = 0
                row += 1


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()

    style = ttk.Style(app)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TScrollbar",
                    background=PANEL_BG, troughcolor=DARK_BG,
                    bordercolor=DARK_BG, arrowcolor=TEXT_MUTED)
    style.configure("Horizontal.TProgressbar",
                    background=ACCENT_BLUE,
                    troughcolor=DARK_BG, bordercolor=DARK_BG)
    style.configure("TSpinbox",
                    fieldbackground="#1E2233",
                    foreground=TEXT_PRIMARY,
                    background=CARD_BG,
                    bordercolor=BORDER,
                    arrowcolor=TEXT_MUTED)
    app.mainloop()


if __name__ == "__main__":
    main()
