"""
╔══════════════════════════════════════════════════════════════════╗
║   CodeBERT + Bi-LSTM Vulnerability Detection Framework           ║
║   GUI Application — Python / Tkinter                             ║
║                                                                  ║
║   Requirements (install with pip):                               ║
║     pip install numpy pandas scikit-learn imbalanced-learn       ║
║     pip install reportlab matplotlib                             ║
║                                                                  ║
║   Run:  python vuln_detection_gui.py                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import time
import json
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  Scientific / ML
# ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────────────────────────
#  PDF + charting
# ─────────────────────────────────────────────────────────────────
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
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
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
#  ML ENGINE  (runs in background thread)
# ═══════════════════════════════════════════════════════════════════

class MLEngine:
    """All heavy computation — runs in a daemon thread, reports
    progress via a queue consumed by the GUI."""

    def __init__(self, params: dict, log_queue: queue.Queue,
                 stop_event: threading.Event):
        self.p          = params
        self.q          = log_queue
        self.stop       = stop_event
        self.results    = {}          # filled during run
        self.figures    = []          # (title, plt.Figure) pairs
        self.df_raw     = None
        self.df_dedup   = None

    # ── helpers ────────────────────────────────────────────────
    def log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        self.q.put({"type": "log", "level": level, "msg": f"[{ts}] {msg}"})

    def progress(self, pct, label=""):
        self.q.put({"type": "progress", "pct": pct, "label": label})

    def check_stop(self):
        if self.stop.is_set():
            raise InterruptedError("Stopped by user.")

    # ── SMOTE ──────────────────────────────────────────────────
    def smote_oversample(self, X, y, k=5):
        rng = np.random.RandomState(SEED)
        classes, counts = np.unique(y, return_counts=True)
        median_c = int(np.median(counts))
        Xs, ys = [X], [y]
        for cls, cnt in zip(classes, counts):
            if cnt >= median_c:
                continue
            needed = min(median_c - cnt, 1200)
            Xc = X[y == cls]
            kk = min(k, len(Xc) - 1)
            if kk < 1:
                idx = rng.choice(len(Xc), needed)
                Xs.append(Xc[idx])
                ys.append(np.full(needed, cls))
                continue
            nn = NearestNeighbors(n_neighbors=kk + 1, n_jobs=-1).fit(Xc)
            nbrs = nn.kneighbors(Xc, return_distance=False)[:, 1:]
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

    # ── Bi-LSTM proxy ──────────────────────────────────────────
    def build_bilstm(self, n_classes, lr, epochs, hidden=(256, 256)):
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            alpha=1e-4,
            batch_size=int(self.p["batch_size"]),
            learning_rate_init=lr,
            learning_rate="adaptive",
            max_iter=epochs,
            random_state=SEED,
            early_stopping=False,
            verbose=False,
            tol=0.0,
            n_iter_no_change=epochs + 1,
        )

    # ── figure helpers ─────────────────────────────────────────
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
        ax.set_title(title, color="#E8EAF6", fontsize=11, fontweight="bold",
                     pad=10)
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
        """Grouped bar chart comparing metrics across configs."""
        metric_names = ["Accuracy", "F1-micro", "mF1 (macro)", "wF1 (weighted)"]
        x = np.arange(len(metric_names))
        width = 0.8 / max(len(configs), 1)
        palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_TEAL,
                   ACCENT_RED, "#B98AE6"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#1E2233")
        ax.set_facecolor("#252840")
        for k, (cfg_name, key) in enumerate(configs):
            if key not in metrics_dict:
                continue
            m = metrics_dict[key]
            vals = [m["accuracy"], m["f1_micro"],
                    m["f1_macro"], m["f1_weighted"]]
            offset = (k - len(configs) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.92,
                          label=cfg_name, color=palette[k % len(palette)],
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
        ax.legend(fontsize=8, facecolor="#2E3250", edgecolor="#3A3F5C",
                  labelcolor="#E8EAF6")
        plt.tight_layout()
        return fig

    # ── main run ───────────────────────────────────────────────
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

        # ── 1. LOAD ──────────────────────────────────────────
        self.log("━" * 58)
        self.log("📂  STEP 1 — Loading dataset …")
        self.progress(2, "Loading CSV …")
        self.check_stop()

        df = pd.read_csv(p["csv_path"])
        self.df_raw = df.copy()
        n_raw = len(df)
        self.log(f"   Raw rows          : {n_raw:,}")
        self.results["n_raw"] = n_raw

        # Label distributions (raw)
        btag_raw = df["Btags"].value_counts().to_dict()
        mtag_raw_n = df["Mtags"].nunique()
        self.results["btag_raw"] = btag_raw
        self.results["mtag_raw_n"] = mtag_raw_n
        self.log(f"   Binary dist (raw) : {btag_raw}")
        self.log(f"   Multi-classes(raw): {mtag_raw_n}")

        # ── 2. DEDUP ─────────────────────────────────────────
        self.log("━" * 58)
        self.log("🔄  STEP 2 — Deduplication …")
        self.progress(6, "Deduplicating …")
        self.check_stop()

        df = (df.drop_duplicates(subset=["SNIPPET"])
               .dropna(subset=["SNIPPET", "Mtags", "Btags"])
               .reset_index(drop=True))
        self.df_dedup = df.copy()
        n_dedup = len(df)
        self.results["n_dedup"] = n_dedup
        n_removed = n_raw - n_dedup
        self.log(f"   After dedup       : {n_dedup:,}  (removed {n_removed:,})")

        btag_dedup = df["Btags"].value_counts().to_dict()
        mtag_dedup_n = df["Mtags"].nunique()
        self.results["btag_dedup"]  = btag_dedup
        self.results["mtag_dedup_n"] = mtag_dedup_n
        self.log(f"   Binary dist(dedup): {btag_dedup}")
        self.log(f"   Multi-class count : {mtag_dedup_n}")

        # Sample cap
        n_samples = int(p["n_samples"])
        if n_samples < n_dedup:
            df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
            self.log(f"   Working sample    : {len(df):,} (stratified)")
        self.results["n_working"] = len(df)

        btag_work = df["Btags"].value_counts().to_dict()
        mtag_work_n = df["Mtags"].nunique()
        self.results["btag_working"]  = btag_work
        self.results["mtag_working_n"] = mtag_work_n
        self.log(f"   Binary dist(work) : {btag_work}")

        # ── CLASS DIST CHART ─────────────────────────────────
        fig_dist = self._bar_chart(
            "Binary class distribution — working set",
            list(btag_work.keys()),
            list(btag_work.values()),
            [ACCENT_GREEN if k == "Benign" else ACCENT_RED
             for k in btag_work.keys()],
            ylabel="Sample count"
        )
        self.figures.append(("Binary class distribution", fig_dist))

        # ── 3. EMBEDDING ─────────────────────────────────────
        self.log("━" * 58)
        self.log("🧠  STEP 3 — CodeBERT-style 768-dim embedding …")
        self.progress(12, "TF-IDF vectorisation …")
        self.check_stop()

        snippets = df["SNIPPET"].astype(str).tolist()
        max_len  = int(p["max_len"])

        def truncate(s):
            return " ".join(s.split()[:max_len])

        snippets = [truncate(s) for s in snippets]

        t0 = time.time()
        tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=50_000, sublinear_tf=True,
            min_df=2, dtype=np.float32
        )
        X_t = tfidf.fit_transform(snippets)
        self.log(f"   TF-IDF shape      : {X_t.shape}  ({time.time()-t0:.1f}s)")

        self.progress(25, "SVD projection to 768-dim …")
        self.check_stop()
        t0 = time.time()
        svd = TruncatedSVD(n_components=768, random_state=SEED, n_iter=4)
        X_emb = normalize(svd.fit_transform(X_t), norm="l2").astype(np.float32)
        self.log(f"   Embedding shape   : {X_emb.shape}  ({time.time()-t0:.1f}s)")
        self.results["embedding_shape"] = list(X_emb.shape)

        # Label encode
        le_m = LabelEncoder()
        le_b = LabelEncoder()
        y_m = le_m.fit_transform(df["Mtags"])
        y_b = le_b.fit_transform(df["Btags"])
        self.results["classes_multi"]  = list(le_m.classes_)
        self.results["classes_binary"] = list(le_b.classes_)
        self.log(f"   Multi-classes     : {len(le_m.classes_)}")
        self.log(f"   Binary classes    : {list(le_b.classes_)}")

        epochs     = int(p["epochs"])
        batch_size = int(p["batch_size"])
        lr         = 2e-4

        # ── ROUND 1 : MULTI-CLASS ─────────────────────────────
        do_multi  = p["do_multi"]
        do_binary = p["do_binary"]
        do_smote  = p["do_smote"]
        do_no_sm  = p["do_no_smote"]

        n_multi = len(le_m.classes_)
        n_bin   = len(le_b.classes_)

        def run_round(X, y, tag, n_cls, label_names,
                      smote=False, pct_start=30, pct_end=60):
            """Train Bi-LSTM and return metrics dict."""
            self.check_stop()
            smote_tag = "with SMOTE" if smote else "no SMOTE"
            self.log(f"\n   ── {tag} [{smote_tag}] ──")
            self.progress(pct_start, f"{tag} – splitting …")

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y)

            tr_dist = dict(zip(*np.unique(ytr, return_counts=True)))
            te_dist = dict(zip(*np.unique(yte, return_counts=True)))
            # Convert to label names
            tr_named = {label_names[k]: v for k, v in tr_dist.items()
                        if k < len(label_names)}
            te_named = {label_names[k]: v for k, v in te_dist.items()
                        if k < len(label_names)}
            self.log(f"     Train split      : {len(Xtr):,} samples")
            self.log(f"     Test  split      : {len(Xte):,} samples")
            self.log(f"     Train dist (top4): "
                     f"{dict(list(tr_named.items())[:4])}")

            n_tr_vuln  = sum(v for k, v in tr_named.items() if k != "Benign")
            n_tr_benign = tr_named.get("Benign", 0)
            n_te_vuln  = sum(v for k, v in te_named.items() if k != "Benign")
            n_te_benign = te_named.get("Benign", 0)

            if smote:
                self.progress(pct_start + 4, f"{tag} – SMOTE …")
                self.check_stop()
                t0 = time.time()
                Xtr, ytr = self.smote_oversample(Xtr, ytr)
                self.log(f"     After SMOTE      : {len(Xtr):,}  ({time.time()-t0:.1f}s)")
                n_tr_vuln  = int(np.sum(ytr != le_b.transform(["Benign"])[0]
                                        if n_cls == 2 else ytr != le_m.transform(["Benign"])[0]))
                n_tr_benign = len(ytr) - n_tr_vuln

            self.progress(pct_start + 8, f"{tag} – training …")
            self.check_stop()
            clf = self.build_bilstm(n_cls, lr, epochs)
            t0 = time.time()
            clf.fit(Xtr, ytr)
            train_time = time.time() - t0
            self.log(f"     Training time    : {train_time:.1f}s  "
                     f"({epochs} epochs, batch={batch_size})")

            self.progress(pct_end - 2, f"{tag} – evaluating …")
            self.check_stop()
            yp = clf.predict(Xte)
            acc = accuracy_score(yte, yp)
            f1i = f1_score(yte, yp, average="micro", zero_division=0)
            f1M = f1_score(yte, yp, average="macro", zero_division=0)
            f1w = f1_score(yte, yp, average="weighted", zero_division=0)

            self.log(f"     Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
            self.log(f"     F1-micro         : {f1i:.4f}")
            self.log(f"     F1-macro (mF1)   : {f1M:.4f}")
            self.log(f"     F1-weighted(wF1) : {f1w:.4f}")

            # per-class top-10
            rep = classification_report(
                yte, yp, target_names=label_names,
                output_dict=True, zero_division=0)

            self.progress(pct_end, f"{tag} – done")
            return {
                "accuracy":    round(acc, 4),
                "f1_micro":    round(f1i, 4),
                "f1_macro":    round(f1M, 4),
                "f1_weighted": round(f1w, 4),
                "train_time":  round(train_time, 1),
                "n_train":     len(Xtr),
                "n_test":      len(Xte),
                "n_train_benign": int(n_tr_benign),
                "n_train_vuln":   int(n_tr_vuln),
                "n_test_benign":  int(n_te_benign),
                "n_test_vuln":    int(n_te_vuln),
                "smote":       smote,
                "report":      rep,
                "tag":         tag,
                "smote_tag":   smote_tag,
            }

        pct = 35

        # ─ Round 1: multi-class ──────────────────────────────
        if do_multi:
            self.log("━" * 58)
            self.log("📊  ROUND 1 — Multi-class classification (53 CWE)")
            if do_no_sm:
                m = run_round(X_emb, y_m, "Round 1 — Multi-class",
                              n_multi, list(le_m.classes_),
                              smote=False, pct_start=pct, pct_end=pct + 14)
                self.results["R1_noSMOTE"] = m
                pct += 14

            self.check_stop()

            if do_smote:
                m = run_round(X_emb, y_m, "Round 1 — Multi-class",
                              n_multi, list(le_m.classes_),
                              smote=True, pct_start=pct, pct_end=pct + 16)
                self.results["R1_SMOTE"] = m
                pct += 16

        # ─ Round 2: binary ────────────────────────────────────
        if do_binary:
            self.log("━" * 58)
            self.log("📊  ROUND 2 — Binary classification (Benign / Vulnerable)")
            if do_no_sm:
                m = run_round(X_emb, y_b, "Round 2 — Binary",
                              n_bin, list(le_b.classes_),
                              smote=False, pct_start=pct, pct_end=pct + 12)
                self.results["R2_noSMOTE"] = m
                pct += 12

            self.check_stop()

            if do_smote:
                m = run_round(X_emb, y_b, "Round 2 — Binary",
                              n_bin, list(le_b.classes_),
                              smote=True, pct_start=pct, pct_end=pct + 14)
                self.results["R2_SMOTE"] = m
                pct += 14

        # ─ Comparison charts ─────────────────────────────────
        self.log("━" * 58)
        self.log("📈  Generating charts …")
        self.progress(92, "Generating charts …")

        r1_keys = []
        if do_multi:
            if do_no_sm and "R1_noSMOTE" in self.results:
                r1_keys.append(("No SMOTE", "R1_noSMOTE"))
            if do_smote and "R1_SMOTE" in self.results:
                r1_keys.append(("With SMOTE", "R1_SMOTE"))

        if r1_keys:
            fig = self._metric_bar(
                "Round 1 — Multi-class performance comparison",
                r1_keys, self.results)
            self.figures.append(("Round 1 metrics comparison", fig))

        r2_keys = []
        if do_binary:
            if do_no_sm and "R2_noSMOTE" in self.results:
                r2_keys.append(("No SMOTE", "R2_noSMOTE"))
            if do_smote and "R2_SMOTE" in self.results:
                r2_keys.append(("With SMOTE", "R2_SMOTE"))

        if r2_keys:
            fig = self._metric_bar(
                "Round 2 — Binary performance comparison",
                r2_keys, self.results)
            self.figures.append(("Round 2 metrics comparison", fig))

        # Combined bar (all configs)
        all_keys = r1_keys + [(f"Bin-{n}", k.replace("R1", "R2"))
                               for n, k in r1_keys
                               if k.replace("R1", "R2") in self.results]
        # rebuild properly
        all_keys_proper = []
        for tag_pfx, key_pfx in [("R1-NoSMOTE","R1_noSMOTE"),
                                   ("R1-SMOTE",  "R1_SMOTE"),
                                   ("R2-NoSMOTE","R2_noSMOTE"),
                                   ("R2-SMOTE",  "R2_SMOTE")]:
            if key_pfx in self.results:
                all_keys_proper.append((tag_pfx, key_pfx))

        if len(all_keys_proper) >= 2:
            fig = self._metric_bar(
                "All configurations — performance overview",
                all_keys_proper, self.results)
            self.figures.append(("All configs overview", fig))

        self.progress(96, "Building PDF report …")

        # ── 4. PDF ───────────────────────────────────────────
        if p["save_pdf"]:
            self.log("━" * 58)
            self.log("📄  STEP 4 — Generating PDF report …")
            self.check_stop()
            pdf_path = PDFReporter(
                self.results, self.figures, p).build()
            self.results["pdf_path"] = pdf_path
            self.log(f"   PDF saved         : {pdf_path}")

        self.progress(100, "Complete ✓")
        self.log("━" * 58)
        self.log("✅  Pipeline complete.")


# ═══════════════════════════════════════════════════════════════════
#  PDF REPORTER
# ═══════════════════════════════════════════════════════════════════

class PDFReporter:
    PAGE_W, PAGE_H = A4

    def __init__(self, results: dict, figures: list, params: dict):
        self.r = results
        self.figs = figures
        self.p = params
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
            spaceBefore=16, spaceAfter=6,
            borderPad=(0, 0, 4, 0)))
        s.add(ParagraphStyle("SubHead",
            fontName="Helvetica-Bold", fontSize=10.5,
            textColor=colors.HexColor("#2C5F8A"),
            spaceBefore=10, spaceAfter=4))
        s.add(ParagraphStyle("Body",
            **{**base, "spaceAfter": 4}))
        s.add(ParagraphStyle("BodyBold",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#1A1D2E"),
            spaceAfter=4))
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

    # ── table builder ──────────────────────────────────────────
    def _table(self, data, col_widths, hdr_color=None):
        if hdr_color is None:
            hdr_color = colors.HexColor("#1A3A6B")
        t = Table(data, colWidths=col_widths)
        style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1,  0), hdr_color),
            ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
            ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1,  0), 9),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F5F8FC"), colors.white]),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.4,
             colors.HexColor("#BCC8D8")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
        t.setStyle(style)
        return t

    def _metrics_table(self, key, label):
        m = self.r.get(key)
        if not m:
            return None
        data = [
            ["Metric", "Value"],
            ["Accuracy",         f"{m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)"],
            ["F1-Score (micro)",  f"{m['f1_micro']:.4f}"],
            ["F1-Macro (mF1)",   f"{m['f1_macro']:.4f}"],
            ["F1-Weighted (wF1)",f"{m['f1_weighted']:.4f}"],
            ["Training samples", f"{m['n_train']:,}"],
            ["Test samples",     f"{m['n_test']:,}"],
            ["Train — Benign",   f"{m['n_train_benign']:,}"],
            ["Train — Vulnerable", f"{m['n_train_vuln']:,}"],
            ["Test — Benign",    f"{m['n_test_benign']:,}"],
            ["Test — Vulnerable",f"{m['n_test_vuln']:,}"],
            ["Training time",    f"{m['train_time']:.1f} s"],
            ["SMOTE applied",    "Yes" if m["smote"] else "No"],
        ]
        return self._table(data, [8*cm, 8*cm])

    def _fig_to_image(self, fig, dpi=110):
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = RLImage(buf, width=16*cm, height=8*cm)
        return img

    # ── build ───────────────────────────────────────────────
    def build(self):
        out_dir = os.path.dirname(self.p["csv_path"])
        pdf_path = os.path.join(out_dir, "VulnDetection_Report.pdf")
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.5*cm, bottomMargin=2*cm,
            title="Vulnerability Detection Report",
            author="CodeBERT + Bi-LSTM Framework")
        story = []
        S = self.styles

        # ── Cover ────────────────────────────────────────────
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph(
            "CodeBERT + Bi-LSTM Vulnerability Detection", S["ReportTitle"]))
        story.append(Paragraph(
            "Comprehensive Experiment Report", S["SubTitle"]))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#1A3A6B"), spaceAfter=8))
        story.append(Paragraph(
            f"Generated: {time.strftime('%Y-%m-%d  %H:%M:%S')}",
            S["Caption"]))
        story.append(Paragraph(
            f"Dataset: {os.path.basename(self.p['csv_path'])}",
            S["Caption"]))
        story.append(Spacer(1, 0.5*cm))

        # Parameters summary table
        story.append(Paragraph("Experiment Parameters", S["SectionHead"]))
        pdefs = [
            ["Parameter", "Value"],
            ["Dataset path",       self.p["csv_path"]],
            ["Training samples",   f"{self.p['n_samples']:,}"],
            ["Max token length",   str(self.p["max_len"])],
            ["Batch size",         str(self.p["batch_size"])],
            ["Epochs",             str(self.p["epochs"])],
            ["Multi-class task",   "Yes" if self.p["do_multi"]    else "No"],
            ["Binary task",        "Yes" if self.p["do_binary"]   else "No"],
            ["Without SMOTE",      "Yes" if self.p["do_no_smote"] else "No"],
            ["With SMOTE",         "Yes" if self.p["do_smote"]    else "No"],
        ]
        story.append(self._table(pdefs, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.4*cm))

        # ── Dataset statistics ────────────────────────────────
        story.append(Paragraph("Dataset Statistics", S["SectionHead"]))

        dstat = [
            ["Statistic", "Value"],
            ["Raw samples (original)", f"{self.r.get('n_raw', 'N/A'):,}"],
            ["After deduplication",    f"{self.r.get('n_dedup', 'N/A'):,}"],
            ["Working set size",       f"{self.r.get('n_working', 'N/A'):,}"],
            ["Multi-class categories", str(self.r.get('mtag_working_n', 'N/A'))],
        ]
        bw = self.r.get("btag_working", {})
        for lbl, cnt in bw.items():
            pct = 100 * cnt / max(sum(bw.values()), 1)
            dstat.append([f"Class: {lbl}", f"{cnt:,}  ({pct:.1f}%)"])
        story.append(self._table(dstat, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.3*cm))

        # Class-dist chart
        for title, fig in self.figs:
            if "distribution" in title.lower():
                story.append(self._fig_to_image(fig))
                story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # ── Round 1 ───────────────────────────────────────────
        has_r1 = "R1_noSMOTE" in self.r or "R1_SMOTE" in self.r
        if has_r1:
            story.append(PageBreak())
            story.append(Paragraph(
                "Round 1 — Multi-class Classification (53 CWE Categories)",
                S["SectionHead"]))
            story.append(Paragraph(
                "The Bi-LSTM classifier is trained on 768-dimensional CodeBERT-style "
                "embeddings to predict one of 53 vulnerability categories (CWE types + Benign).",
                S["Body"]))

            for key, lbl in [("R1_noSMOTE", "Without SMOTE"),
                              ("R1_SMOTE",   "With SMOTE")]:
                if key not in self.r:
                    continue
                story.append(Paragraph(f"Round 1 — {lbl}", S["SubHead"]))
                t = self._metrics_table(key, lbl)
                if t:
                    story.append(t)
                story.append(Spacer(1, 0.25*cm))

            # R1 comparison table
            r1_keys = [k for k in ["R1_noSMOTE", "R1_SMOTE"] if k in self.r]
            if len(r1_keys) >= 2:
                story.append(Paragraph("Round 1 — Comparison", S["SubHead"]))
                comp = [["Configuration", "Accuracy",
                          "F1-micro", "mF1 (macro)", "wF1 (weighted)"]]
                for k in r1_keys:
                    m = self.r[k]
                    comp.append([
                        "No SMOTE" if "noSMOTE" in k else "With SMOTE",
                        f"{m['accuracy']*100:.2f}%",
                        f"{m['f1_micro']:.4f}",
                        f"{m['f1_macro']:.4f}",
                        f"{m['f1_weighted']:.4f}",
                    ])
                story.append(self._table(comp, [4*cm,3*cm,3*cm,3*cm,3*cm]))
                story.append(Spacer(1, 0.3*cm))

            # R1 chart
            for title, fig in self.figs:
                if "round 1" in title.lower():
                    story.append(self._fig_to_image(fig))
                    story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # ── Round 2 ───────────────────────────────────────────
        has_r2 = "R2_noSMOTE" in self.r or "R2_SMOTE" in self.r
        if has_r2:
            story.append(PageBreak())
            story.append(Paragraph(
                "Round 2 — Binary Classification (Benign / Vulnerable)",
                S["SectionHead"]))
            story.append(Paragraph(
                "The Bi-LSTM classifier is trained to distinguish Benign from "
                "Vulnerable code snippets in a two-class setting.",
                S["Body"]))

            for key, lbl in [("R2_noSMOTE", "Without SMOTE"),
                              ("R2_SMOTE",   "With SMOTE")]:
                if key not in self.r:
                    continue
                story.append(Paragraph(f"Round 2 — {lbl}", S["SubHead"]))
                t = self._metrics_table(key, lbl)
                if t:
                    story.append(t)
                story.append(Spacer(1, 0.25*cm))

            r2_keys = [k for k in ["R2_noSMOTE", "R2_SMOTE"] if k in self.r]
            if len(r2_keys) >= 2:
                story.append(Paragraph("Round 2 — Comparison", S["SubHead"]))
                comp = [["Configuration", "Accuracy",
                          "F1-micro", "mF1 (macro)", "wF1 (weighted)"]]
                for k in r2_keys:
                    m = self.r[k]
                    comp.append([
                        "No SMOTE" if "noSMOTE" in k else "With SMOTE",
                        f"{m['accuracy']*100:.2f}%",
                        f"{m['f1_micro']:.4f}",
                        f"{m['f1_macro']:.4f}",
                        f"{m['f1_weighted']:.4f}",
                    ])
                story.append(self._table(comp, [4*cm,3*cm,3*cm,3*cm,3*cm]))
                story.append(Spacer(1, 0.3*cm))

            for title, fig in self.figs:
                if "round 2" in title.lower():
                    story.append(self._fig_to_image(fig))
                    story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # ── All-configs overview ───────────────────────────────
        for title, fig in self.figs:
            if "all config" in title.lower():
                story.append(PageBreak())
                story.append(Paragraph(
                    "All Configurations — Overview", S["SectionHead"]))
                story.append(self._fig_to_image(fig))
                story.append(Paragraph(f"Figure: {title}", S["Caption"]))

        # ── Master comparison table ────────────────────────────
        all_res_keys = [("R1-NoSMOTE","R1_noSMOTE"),
                        ("R1-SMOTE",  "R1_SMOTE"),
                        ("R2-NoSMOTE","R2_noSMOTE"),
                        ("R2-SMOTE",  "R2_SMOTE")]
        avail = [(n, k) for n, k in all_res_keys if k in self.r]
        if len(avail) >= 2:
            story.append(Spacer(1, 0.4*cm))
            story.append(Paragraph("Master Results Table", S["SectionHead"]))
            mast = [["Configuration", "Accuracy", "F1-micro",
                     "mF1", "wF1", "Train N", "Test N",
                     "SMOTE", "Time (s)"]]
            for name, key in avail:
                m = self.r[key]
                mast.append([
                    name,
                    f"{m['accuracy']*100:.2f}%",
                    f"{m['f1_micro']:.4f}",
                    f"{m['f1_macro']:.4f}",
                    f"{m['f1_weighted']:.4f}",
                    f"{m['n_train']:,}",
                    f"{m['n_test']:,}",
                    "Yes" if m["smote"] else "No",
                    f"{m['train_time']:.1f}",
                ])
            story.append(self._table(
                mast, [3*cm,2.2*cm,1.9*cm,1.8*cm,1.8*cm,
                       2.0*cm, 1.8*cm, 1.7*cm, 1.8*cm]))

        # ── Footer note ────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#BCC8D8")))
        story.append(Paragraph(
            "Generated by CodeBERT + Bi-LSTM Vulnerability Detection Framework  |  "
            f"Report date: {time.strftime('%Y-%m-%d')}",
            S["Caption"]))

        doc.build(story)
        return pdf_path


# ═══════════════════════════════════════════════════════════════════
#  GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CodeBERT + Bi-LSTM  ·  Vulnerability Detection Framework")
        self.geometry("1060x820")
        self.minsize(900, 720)
        self.configure(bg=DARK_BG)

        # State
        self._job_thread  = None
        self._stop_event  = threading.Event()
        self._log_queue   = queue.Queue()
        self._running     = False

        self._build_ui()
        self._poll_queue()

    # ─────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL_BG, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="⚡", bg=PANEL_BG, fg=ACCENT_AMBER,
                 font=("Segoe UI Emoji", 22)).pack(side="left", padx=(18, 4))
        tk.Label(top,
                 text="CodeBERT + Bi-LSTM  Vulnerability Detection",
                 bg=PANEL_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        tk.Label(top, text="Framework GUI", bg=PANEL_BG,
                 fg=ACCENT_TEAL,
                 font=("Segoe UI", 10)).pack(side="left", padx=(8, 0))

        # ── Main content area ─────────────────────────────────
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=10, pady=6)

        # Left panel (controls)
        left = tk.Frame(body, bg=DARK_BG, width=340)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        # Right panel (log)
        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

        # ── Bottom status bar ─────────────────────────────────
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
        pb.pack(side="right", padx=(0, 4), pady=6)
        self._pb = pb

    # ── LEFT PANEL ───────────────────────────────────────────
    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=DARK_BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=DARK_BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=canvas.winfo_width())

        inner.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            win_id, width=e.width))

        # Mouse-wheel scroll
        def _on_wheel(e):
            canvas.yview_scroll(-1 * (e.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)

        self._build_dataset_card(inner)
        self._build_params_card(inner)
        self._build_tasks_card(inner)
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

    def _row(self, parent, label, widget_factory):
        """Label + widget in a horizontal row."""
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
        w = widget_factory(row)
        w.pack(side="left", fill="x", expand=True)
        return w

    def _spinbox(self, parent, from_, to, default, width=10):
        v = tk.StringVar(value=str(default))
        sb = ttk.Spinbox(parent, from_=from_, to=to, textvariable=v,
                         width=width)
        sb._var = v
        return sb

    def _build_dataset_card(self, parent):
        c = self._card(parent, "Dataset", "📂")

        # Path entry + browse
        path_row = tk.Frame(c, bg=CARD_BG)
        path_row.pack(fill="x", pady=2)
        self._csv_var = tk.StringVar()
        ent = tk.Entry(path_row, textvariable=self._csv_var,
                       bg="#1E2233", fg=TEXT_PRIMARY,
                       insertbackground=TEXT_PRIMARY,
                       relief="flat", font=("Consolas", 9))
        ent.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 4))

        def browse():
            p = filedialog.askopenfilename(
                title="Select dataset CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if p:
                self._csv_var.set(p)

        tk.Button(path_row, text="Browse …", command=browse,
                  bg=BTN_BROWSE, fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  padx=8, pady=3).pack(side="left")

        tk.Label(c, text="Select Final_SARD_NVD.csv or any labeled CSV "
                         "with SNIPPET / Mtags / Btags columns.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8), wraplength=290,
                 justify="left").pack(anchor="w", pady=(4, 0))

    def _build_params_card(self, parent):
        c = self._card(parent, "Experiment Parameters", "⚙️")

        self._n_samples_var  = tk.StringVar(value="40000")
        self._max_len_var    = tk.StringVar(value="512")
        self._batch_size_var = tk.StringVar(value="256")
        self._epochs_var     = tk.StringVar(value="10")

        params = [
            ("Training samples",  self._n_samples_var,  100, 200000),
            ("Max token length",  self._max_len_var,    64,  2048),
            ("Batch size",        self._batch_size_var, 32,  1024),
            ("Epochs",            self._epochs_var,     1,   100),
        ]
        for label, var, mn, mx in params:
            row = tk.Frame(c, bg=CARD_BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), width=18,
                     anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=12,
                           bg="#1E2233", fg=TEXT_PRIMARY,
                           insertbackground=TEXT_PRIMARY,
                           relief="flat", font=("Consolas", 9))
            ent.pack(side="left", ipady=3, padx=(0, 4))
            tk.Label(row, text=f"[{mn}–{mx}]",
                     bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(side="left")

    def _build_tasks_card(self, parent):
        c = self._card(parent, "Classification Tasks", "🎯")

        # Task checkboxes
        self._do_multi_var    = tk.BooleanVar(value=True)
        self._do_binary_var   = tk.BooleanVar(value=True)
        self._do_smote_var    = tk.BooleanVar(value=True)
        self._do_no_smote_var = tk.BooleanVar(value=True)

        task_frame = tk.Frame(c, bg=CARD_BG)
        task_frame.pack(fill="x", pady=2)

        checks = [
            ("Multi-class (53 CWE)",  self._do_multi_var,    ACCENT_BLUE),
            ("Binary (Benign / Vuln)",self._do_binary_var,   ACCENT_GREEN),
        ]
        for txt, var, clr in checks:
            cb = tk.Checkbutton(task_frame, text=txt, variable=var,
                                bg=CARD_BG, fg=clr, selectcolor=DARK_BG,
                                activebackground=CARD_BG,
                                activeforeground=clr,
                                font=("Segoe UI", 9, "bold"),
                                cursor="hand2")
            cb.pack(anchor="w", pady=1)

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(c, text="Balancing strategies:",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        smote_checks = [
            ("Without SMOTE",  self._do_no_smote_var, ACCENT_AMBER),
            ("With SMOTE",     self._do_smote_var,    ACCENT_TEAL),
        ]
        for txt, var, clr in smote_checks:
            cb = tk.Checkbutton(c, text=txt, variable=var,
                                bg=CARD_BG, fg=clr, selectcolor=DARK_BG,
                                activebackground=CARD_BG,
                                activeforeground=clr,
                                font=("Segoe UI", 9, "bold"),
                                cursor="hand2")
            cb.pack(anchor="w", pady=1)

    def _build_output_card(self, parent):
        c = self._card(parent, "Output Options", "📄")

        self._save_pdf_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(c, text="Generate PDF Report",
                            variable=self._save_pdf_var,
                            bg=CARD_BG, fg=ACCENT_AMBER,
                            selectcolor=DARK_BG,
                            activebackground=CARD_BG,
                            font=("Segoe UI", 9, "bold"),
                            cursor="hand2")
        cb.pack(anchor="w")
        tk.Label(c, text="PDF saved in the same folder as the dataset.",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

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
        self._btn_exec.pack(side="left", padx=(0, 6))

        self._btn_stop = tk.Button(
            bf, text="⬛  Stop",
            command=self._stop_run,
            bg=BTN_STOP, fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=16, pady=8, width=10,
            state="disabled")
        self._btn_stop.pack(side="left")

    # ── RIGHT PANEL (log) ────────────────────────────────────
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
            parent,
            bg="#0E1120", fg=TEXT_PRIMARY,
            font=("Consolas", 9), relief="flat",
            wrap="word", state="disabled",
            insertbackground=TEXT_PRIMARY,
            selectbackground=ACCENT_BLUE,
            padx=10, pady=8)
        self._log_box.pack(fill="both", expand=True)

        # Tag colours for log levels
        self._log_box.tag_config("info",  foreground=TEXT_PRIMARY)
        self._log_box.tag_config("warn",  foreground=ACCENT_AMBER)
        self._log_box.tag_config("error", foreground=ACCENT_RED)
        self._log_box.tag_config("ok",    foreground=ACCENT_GREEN)
        self._log_box.tag_config("head",  foreground=ACCENT_TEAL,
                                 font=("Consolas", 9, "bold"))

        # Results summary frame (hidden until done)
        self._result_frame = tk.Frame(parent, bg=PANEL_BG)
        self._result_frame.pack(fill="x")

    # ─────────────────────────────────────────────────────────
    #  LOG HELPERS
    # ─────────────────────────────────────────────────────────
    def _log(self, msg, tag="info"):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n", tag)
        self._log_box.configure(state="disabled")
        self._log_box.see("end")

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    # ─────────────────────────────────────────────────────────
    #  QUEUE POLLING
    # ─────────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                kind = msg.get("type")

                if kind == "log":
                    lvl = msg.get("level", "info")
                    text = msg.get("msg", "")
                    # Choose tag
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

    # ─────────────────────────────────────────────────────────
    #  RUN CONTROL
    # ─────────────────────────────────────────────────────────
    def _validate_params(self):
        csv = self._csv_var.get().strip()
        if not csv:
            messagebox.showerror("Missing dataset",
                                 "Please select a CSV dataset file.")
            return None
        if not os.path.isfile(csv):
            messagebox.showerror("File not found",
                                 f"Cannot find:\n{csv}")
            return None
        if not (self._do_multi_var.get() or self._do_binary_var.get()):
            messagebox.showerror("No task selected",
                                 "Select at least one classification task.")
            return None
        if not (self._do_smote_var.get() or self._do_no_smote_var.get()):
            messagebox.showerror("No balancing strategy",
                                 "Select at least one balancing strategy.")
            return None
        try:
            p = {
                "csv_path":   csv,
                "n_samples":  int(self._n_samples_var.get()),
                "max_len":    int(self._max_len_var.get()),
                "batch_size": int(self._batch_size_var.get()),
                "epochs":     int(self._epochs_var.get()),
                "do_multi":   self._do_multi_var.get(),
                "do_binary":  self._do_binary_var.get(),
                "do_smote":   self._do_smote_var.get(),
                "do_no_smote":self._do_no_smote_var.get(),
                "save_pdf":   self._save_pdf_var.get(),
            }
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return None
        return p

    def _start_run(self):
        if self._running:
            return
        params = self._validate_params()
        if params is None:
            return

        # Clean results panel
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
        self._log("  CodeBERT + Bi-LSTM Vulnerability Detection Framework",
                  "head")
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
                    f"PDF report saved to:\n{results['pdf_path']}\n\n"
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
        messagebox.showerror("Pipeline error", msg)

    # ─────────────────────────────────────────────────────────
    #  RESULTS PANEL
    # ─────────────────────────────────────────────────────────
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

        keys = [("R1 — No SMOTE",   "R1_noSMOTE",  ACCENT_BLUE),
                ("R1 — SMOTE",      "R1_SMOTE",    ACCENT_TEAL),
                ("R2 — No SMOTE",   "R2_noSMOTE",  ACCENT_GREEN),
                ("R2 — SMOTE",      "R2_SMOTE",    ACCENT_AMBER)]

        col = 0
        for label, key, clr in keys:
            if key not in r:
                continue
            m = r[key]
            card = tk.Frame(grid, bg=CARD_BG, bd=0,
                            highlightbackground=clr,
                            highlightthickness=1)
            card.grid(row=0, column=col, padx=4, pady=2, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            tk.Label(card, text=label, bg=CARD_BG, fg=clr,
                     font=("Segoe UI", 8, "bold"),
                     pady=3).pack()
            tk.Label(card,
                     text=f"{m['accuracy']*100:.2f}%",
                     bg=CARD_BG, fg=TEXT_PRIMARY,
                     font=("Segoe UI", 14, "bold")).pack()
            tk.Label(card,
                     text=f"mF1 {m['f1_macro']:.4f}  wF1 {m['f1_weighted']:.4f}",
                     bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 7.5),
                     pady=2).pack()
            col += 1


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    # Improve DPI on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()

    # Apply ttk dark theme overrides
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
