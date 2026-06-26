import sys
import json
import pickle
import warnings
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

try:
    import ot as pot
    _HAS_POT = True
except ImportError:
    _HAS_POT = False

def compute_ot_plan(X_src_train: np.ndarray,
                    X_tgt_train: np.ndarray,
                    log: RunLogger) -> np.ndarray:
    if not _HAS_POT:
        log.warning("  POT library not found — skipping OT alignment")
        return None

    log.info("  Estimating OT plan from training patients (Sinkhorn)")

    n_src = X_src_train.shape[1]
    n_tgt = X_tgt_train.shape[1]

    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt

    mu_src = X_src_train.mean(axis=0)
    mu_tgt = X_tgt_train.mean(axis=0)

    max_feat = 2000
    idx_src = np.random.choice(n_src, min(max_feat, n_src), replace=False)
    idx_tgt = np.random.choice(n_tgt, min(max_feat, n_tgt), replace=False)

    M = pot.dist(mu_src[idx_src].reshape(-1, 1),
                  mu_tgt[idx_tgt].reshape(-1, 1), metric="sqeuclidean")
    M /= M.max() + 1e-10

    a_sub = np.ones(len(idx_src)) / len(idx_src)
    b_sub = np.ones(len(idx_tgt)) / len(idx_tgt)

    try:
        T = pot.sinkhorn(a_sub, b_sub, M,
                         reg=C.OT_EPSILON,
                         numItermax=C.OT_MAX_ITER,
                         stopThr=C.OT_STOP_THRESHOLD)
        log.info(f"  OT plan shape: {T.shape}  (transport matrix, sub-sampled)")
        return {"plan": T, "idx_src": idx_src, "idx_tgt": idx_tgt}
    except Exception as ex:
        log.warning(f"  Sinkhorn failed ({ex}) — no OT alignment applied")
        return None

def apply_ot_alignment(X_mrna: np.ndarray,
                       X_meth: np.ndarray,
                       ot_result: dict,
                       log: RunLogger) -> np.ndarray:
    if ot_result is None:
        return np.concatenate([X_mrna, X_meth], axis=1).astype(np.float32)

    T    = ot_result["plan"]
    i_s  = ot_result["idx_src"]
    i_t  = ot_result["idx_tgt"]

    transport_weights = T.sum(axis=0)
    transport_weights /= transport_weights.sum() + 1e-10

    X_meth_sub = X_meth[:, i_t]
    X_meth_aligned = X_meth_sub * transport_weights[np.newaxis, :]

    result = np.concatenate([X_mrna, X_meth_aligned], axis=1).astype(np.float32)
    log.info(f"  OT-aligned feature shape: {result.shape}")
    return result

def load_data(log: RunLogger):
    log.section("Loading Stage 1/2 data")

    with open(C.GRAPH_DATA_FILE, "rb") as f:
        gd = pickle.load(f)

    rankings_df = pd.read_pickle(C.BIOMARKER_RANK_FILE)
    labels_df   = pd.read_csv(C.SAMPLE_LABELS_FILE, index_col=0)

    with open(C.RESULTS_DIR / "data_splits.json") as f:
        splits_data = json.load(f)

    common_samples = gd["common_samples"]
    node_features  = gd["node_features"]
    node_metadata  = gd["node_metadata"]

    omic_type_ids = C.OMIC_TYPE_IDS

    def _get_modality(omic_name):
        mask = node_metadata["omic_type"] == omic_name
        ids  = node_metadata[mask]["node_id"].values

        return node_features[ids, 1:].T

    X_mrna = _get_modality("mRNA")
    X_meth = _get_modality("methylation")
    X_mirna= _get_modality("miRNA")

    y = np.array([C.CLASS_TO_INT[labels_df.loc[s, "PAM50"]]
                  for s in common_samples
                  if s in labels_df.index and
                     labels_df.loc[s, "PAM50"] in C.CLASS_TO_INT])
    valid_samples = [s for s in common_samples
                     if s in labels_df.index and
                        labels_df.loc[s, "PAM50"] in C.CLASS_TO_INT]
    n_valid = len(valid_samples)
    X_mrna  = X_mrna[:n_valid]
    X_meth  = X_meth[:n_valid]
    X_mirna = X_mirna[:n_valid]

    log.info(f"  Valid samples  : {n_valid}")
    log.info(f"  mRNA features  : {X_mrna.shape[1]}")
    log.info(f"  Meth features  : {X_meth.shape[1]}")
    log.info(f"  miRNA features : {X_mirna.shape[1]}")
    for i, name in enumerate(C.CLASS_NAMES):
        log.info(f"  {name}: {(y==i).sum()}")

    return X_mrna, X_meth, X_mirna, y, valid_samples, rankings_df, splits_data

class HOBITClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim=C.CLF_HIDDEN_DIM,
                 n_layers=C.CLF_N_LAYERS, dropout=C.CLF_DROPOUT,
                 n_classes=C.NUM_CLASSES):
        super().__init__()
        self.dropout_p = dropout

        self.input_bn  = nn.BatchNorm1d(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_classes),
        )

    def forward(self, x):
        x = self.input_bn(x)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h) + h
        return self.head(h)

    def mc_predict(self, x: torch.Tensor, n_samples: int = C.CLF_N_MC_SAMPLES):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        self.eval()
        preds = np.array(preds)
        mean_probs = preds.mean(axis=0)
        std_probs  = preds.std(axis=0)
        return mean_probs, std_probs, preds

def focal_ce_loss(logits, targets, gamma=C.CLF_FOCAL_GAMMA,
                  weight=None, label_smoothing=C.CLF_LABEL_SMOOTHING):
    n_cls = logits.shape[1]
    log_p = F.log_softmax(logits, dim=1)

    if label_smoothing > 0:
        with torch.no_grad():
            smooth = torch.zeros_like(logits)
            smooth.fill_(label_smoothing / (n_cls - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1 - label_smoothing)
        ce = -(smooth * log_p).sum(dim=1)
    else:
        ce = F.nll_loss(log_p, targets, weight=weight, reduction="none")

    pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()

def train_one_seed(X_train, X_val, y_train, y_val,
                   in_dim, seed, log: RunLogger):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HOBITClassifier(in_dim).to(C.DEVICE)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(C.DEVICE)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(C.DEVICE)
    yt = torch.tensor(y_train,  dtype=torch.long).to(C.DEVICE)
    yv = torch.tensor(y_val,    dtype=torch.long).to(C.DEVICE)

    counts = torch.bincount(yt, minlength=C.NUM_CLASSES).float()
    w = (1.0 / counts) * len(yt) / C.NUM_CLASSES
    w = w.to(C.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=C.CLF_LR,
                                  weight_decay=C.CLF_WEIGHT_DECAY)

    def lr_fn(ep):
        if ep < C.CLF_WARMUP_EPOCHS:
            return ep / max(C.CLF_WARMUP_EPOCHS, 1)
        p = (ep - C.CLF_WARMUP_EPOCHS) / max(C.CLF_EPOCHS - C.CLF_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_score, best_state = 0.0, None
    patience_cnt = 0
    history = {"loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(C.CLF_EPOCHS):
        model.train()
        logits = model(Xt)
        loss   = focal_ce_loss(logits, yt, weight=w)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())

        if epoch % 5 == 0 or epoch == C.CLF_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vl = model(Xv)
                vp = vl.argmax(dim=1).cpu().numpy()
                vt = yv.cpu().numpy()
                val_acc = accuracy_score(vt, vp)
                val_f1  = f1_score(vt, vp, average="macro", zero_division=0)
                val_bal = balanced_accuracy_score(vt, vp)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            score = 0.4 * val_bal + 0.3 * val_f1 + 0.3 * val_acc
            if score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= C.CLF_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history

def compute_calibration(mean_probs: np.ndarray,
                         y_true: np.ndarray,
                         n_bins: int = C.ECE_N_BINS) -> dict:
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    is_correct  = (predictions == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece_sum, mce, total = 0.0, 0.0, len(y_true)
    reliability_x, reliability_y = [], []
    bin_sizes, bin_accs, bin_confs = [], [], []

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        acc  = is_correct[mask].mean()
        conf = confidences[mask].mean()
        size = mask.sum()
        gap  = abs(acc - conf)

        ece_sum += (size / total) * gap
        mce      = max(mce, gap)

        reliability_x.append(conf)
        reliability_y.append(acc)
        bin_sizes.append(int(size))
        bin_accs.append(float(acc))
        bin_confs.append(float(conf))

    pred_entropy = -np.sum(mean_probs * np.log2(mean_probs + 1e-10), axis=1)

    return {
        "ECE":             float(ece_sum),
        "MCE":             float(mce),
        "pred_entropy":    pred_entropy.tolist(),
        "mean_entropy":    float(pred_entropy.mean()),
        "reliability_x":  reliability_x,
        "reliability_y":  reliability_y,
        "bin_sizes":       bin_sizes,
        "bin_accs":        bin_accs,
        "bin_confs":       bin_confs,
        "overconf_rate":   float((confidences[is_correct == 0] > 0.8).mean())
            if (is_correct == 0).sum() > 0 else 0.0,
    }

def compute_mutual_information(all_probs: np.ndarray) -> np.ndarray:
    mean_p = all_probs.mean(axis=0)
    H_mean = -np.sum(mean_p * np.log2(mean_p + 1e-10), axis=1)
    H_ind  = -np.sum(all_probs * np.log2(all_probs + 1e-10), axis=2)
    H_mean_ind = H_ind.mean(axis=0)
    mi = np.maximum(H_mean - H_mean_ind, 0.0)
    return mi

def spearman_entropy_error(pred_entropy: np.ndarray, y_true, y_pred) -> dict:
    errors = (y_pred != y_true).astype(float)
    rho, pval = scipy_stats.spearmanr(pred_entropy, errors)
    return {"spearman_rho": float(rho), "spearman_pval": float(pval)}

def evaluate(mean_probs, y_true, all_probs, config_seed, log):
    y_pred = mean_probs.argmax(axis=1)
    y_bin  = label_binarize(y_true, classes=list(range(C.NUM_CLASSES)))

    acc   = accuracy_score(y_true, y_pred)
    balacc= balanced_accuracy_score(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    try:
        auc_m = roc_auc_score(y_bin, mean_probs, average="macro",    multi_class="ovr")
        auc_w = roc_auc_score(y_bin, mean_probs, average="weighted", multi_class="ovr")
    except Exception:
        auc_m = auc_w = float("nan")

    calib = compute_calibration(mean_probs, y_true)
    mi    = compute_mutual_information(all_probs)
    corr  = spearman_entropy_error(np.array(calib["pred_entropy"]), y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    f1_pc   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_pc  = recall_score(y_true, y_pred,    average=None, zero_division=0)

    log.log_metric(f"[seed={config_seed}] accuracy",       acc)
    log.log_metric(f"[seed={config_seed}] balanced_acc",   balacc)
    log.log_metric(f"[seed={config_seed}] F1_macro",       f1m)
    log.log_metric(f"[seed={config_seed}] AUC_macro",      auc_m)
    log.log_metric(f"[seed={config_seed}] ECE",            calib["ECE"])
    log.log_metric(f"[seed={config_seed}] MCE",            calib["MCE"])
    log.log_metric(f"[seed={config_seed}] MI_mean",        mi.mean())
    log.log_metric(f"[seed={config_seed}] entropy_spearman_rho", corr["spearman_rho"])

    return {
        "accuracy":     acc,
        "balanced_acc": balacc,
        "F1_macro":     f1m,
        "F1_weighted":  f1w,
        "precision":    prec,
        "recall":       rec,
        "AUC_macro":    auc_m,
        "AUC_weighted": auc_w,
        "ECE":          calib["ECE"],
        "MCE":          calib["MCE"],
        "MI_mean":      float(mi.mean()),
        "spearman_rho": corr["spearman_rho"],
        "spearman_pval":corr["spearman_pval"],
        "f1_per_class":  f1_pc.tolist(),
        "prec_per_class":prec_pc.tolist(),
        "rec_per_class": rec_pc.tolist(),
        "confusion_matrix": cm.tolist(),
        "calibration":   calib,
        "mi":            mi.tolist(),
    }

def run_baselines(X_train, y_train, X_test, y_test, log: RunLogger) -> dict:
    log.section("Baseline Model Comparison")

    baselines = {
        "Random Forest":  RandomForestClassifier(n_estimators=200, random_state=42,
                                                  class_weight="balanced", n_jobs=-1),
        "SVM (RBF)":      SVC(kernel="rbf", probability=True,
                               class_weight="balanced", random_state=42),
        "Logistic Reg.":  LogisticRegression(multi_class="ovr", class_weight="balanced",
                                              max_iter=1000, random_state=42),
    }

    results = {}
    for name, clf in baselines.items():
        try:
            clf.fit(X_train, y_train)
            y_pred  = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)
            y_bin   = label_binarize(y_test, classes=list(range(C.NUM_CLASSES)))
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
            except Exception:
                auc = float("nan")
            results[name] = {"accuracy": acc, "F1_macro": f1m, "AUC_macro": auc}
            log.info(f"  {name:<22}: Acc={acc:.4f}  F1={f1m:.4f}  AUC={auc:.4f}")
        except Exception as ex:
            log.warning(f"  {name} failed: {ex}")
            results[name] = {"accuracy": float("nan"), "F1_macro": float("nan"),
                             "AUC_macro": float("nan")}

    return results

def run_ablation(X_train_full, y_train, X_test_full, y_test,
                 ot_result, X_mrna_tr, X_meth_tr, X_mirna_tr,
                 X_mrna_te, X_meth_te, X_mirna_te,
                 seed, log: RunLogger) -> dict:
    log.section("Ablation Study")

    def _quick_eval(X_tr, X_te, label):
        if X_tr.shape[1] == 0:
            log.warning(f"  [Ablation] {label}: empty features, skipping")
            return {"accuracy": float("nan"), "F1_macro": float("nan"),
                    "AUC_macro": float("nan"), "ECE": float("nan")}
        clf = HOBITClassifier(X_tr.shape[1]).to(C.DEVICE)
        torch.manual_seed(seed)
        Xtr = torch.tensor(X_tr, dtype=torch.float32).to(C.DEVICE)
        Xte = torch.tensor(X_te, dtype=torch.float32).to(C.DEVICE)
        yt  = torch.tensor(y_train, dtype=torch.long).to(C.DEVICE)
        opt = torch.optim.AdamW(clf.parameters(), lr=C.CLF_LR, weight_decay=C.CLF_WEIGHT_DECAY)
        clf.train()
        for ep in range(100):
            logits = clf(Xtr)
            loss   = F.cross_entropy(logits, yt)
            opt.zero_grad(); loss.backward(); opt.step()
        mean_p, std_p, all_p = clf.mc_predict(Xte)
        yp    = mean_p.argmax(axis=1)
        acc   = accuracy_score(y_test, yp)
        f1m   = f1_score(y_test, yp, average="macro", zero_division=0)
        yb    = label_binarize(y_test, classes=list(range(C.NUM_CLASSES)))
        try:   auc = roc_auc_score(yb, mean_p, average="macro", multi_class="ovr")
        except: auc = float("nan")
        calib = compute_calibration(mean_p, y_test)
        log.info(f"  {label:<45}: Acc={acc:.4f}  F1={f1m:.4f}  ECE={calib['ECE']:.4f}")
        return {"accuracy": acc, "F1_macro": f1m, "AUC_macro": auc, "ECE": calib["ECE"]}

    X_tr_zscore = np.concatenate([X_mrna_tr, X_meth_tr, X_mirna_tr], axis=1)
    X_te_zscore = np.concatenate([X_mrna_te, X_meth_te, X_mirna_te], axis=1)

    results = {
        "Full HOBIT":
            _quick_eval(X_train_full, X_test_full, "Full HOBIT (100ep approx)"),
        "w/o Optimal Transport":
            _quick_eval(X_tr_zscore,  X_te_zscore,  "w/o OT  (Z-score only)"),
        "w/o miRNA":
            _quick_eval(np.concatenate([X_mrna_tr, X_meth_tr], axis=1),
                        np.concatenate([X_mrna_te, X_meth_te], axis=1),
                        "w/o miRNA"),
        "mRNA only":
            _quick_eval(X_mrna_tr, X_mrna_te, "mRNA only"),
    }

    for cfg, res in results.items():
        log.info(f"  {cfg:<45}: "
                 f"Acc={res['accuracy']:.4f}  F1={res['F1_macro']:.4f}  "
                 f"ECE={res.get('ECE', float('nan')):.4f}")

    return results

def generate_stage3_figures(all_seed_results: list,
                             mean_res: dict,
                             baseline_results: dict,
                             ablation_results: dict,
                             log: RunLogger):
    log.section("Generating Stage 3 Figures")
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pick last seed's calibration + predictions for plotting
    last = all_seed_results[-1]
    calib = last["calibration"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rx, ry = calib["reliability_x"], calib["reliability_y"]
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    axes[0].plot(rx, ry, "o-", color="#E53935", lw=2, ms=7, label="HOBIT")
    axes[0].fill_between(rx, ry, rx, alpha=0.15, color="#E53935",
                          label=f"ECE={calib['ECE']:.4f}  MCE={calib['MCE']:.4f}")
    axes[0].set_xlabel("Mean Confidence")
    axes[0].set_ylabel("Fraction Correct")
    axes[0].set_title("Reliability Diagram (Calibration Curve) — BRCA PAM50")
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(calib["bin_confs"], calib["bin_accs"],
                width=1.0/C.ECE_N_BINS, align="center",
                color="#42A5F5", alpha=0.7, edgecolor="white", label="Accuracy per bin")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Calibration Histogram")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_reliability_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_reliability_diagram.png")

    mean_probs = np.array(last["calibration"]["reliability_x"])
    # We'll use the stored confusion matrix for this figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Confidence Distribution — BRCA PAM50 (last seed)")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.text(0.5, 0.5,
            f"ECE={calib['ECE']:.4f}\nMCE={calib['MCE']:.4f}\n"
            f"Mean entropy={calib['mean_entropy']:.4f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_confidence_distribution.png")

    cm = np.array(last["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix — BRCA PAM50\n"
                       f"Acc={last['accuracy']:.3f}  F1={last['F1_macro']:.3f}")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[1])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Row-Normalised)")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(C.CLASS_NAMES))
    w = 0.25
    ax.bar(x - w,   last["prec_per_class"], w, label="Precision", color="#42A5F5")
    ax.bar(x,       last["rec_per_class"],  w, label="Recall",    color="#66BB6A")
    ax.bar(x + w,   last["f1_per_class"],   w, label="F1",        color="#EF5350")
    ax.set_xticks(x); ax.set_xticklabels(C.CLASS_NAMES)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance — BRCA PAM50")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(last["f1_per_class"]):
        ax.text(i + w, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_per_class_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_per_class_performance.png")

    metrics_to_plot = ["accuracy", "balanced_acc", "F1_macro", "AUC_macro", "ECE"]
    fig, ax = plt.subplots(figsize=(12, 6))
    means = [mean_res["mean"].get(m, 0) for m in metrics_to_plot]
    stds  = [mean_res["std"].get(m, 0)  for m in metrics_to_plot]
    colors= ["#42A5F5", "#66BB6A", "#EF5350", "#AB47BC", "#FF7043"]
    ax.bar(metrics_to_plot, means, color=colors, alpha=0.8, capsize=5, yerr=stds)
    ax.set_ylabel("Score"); ax.set_title(f"HOBIT Performance (mean±SD, N={C.N_SEEDS} seeds)")
    ax.set_ylim(0, 1.05)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_multi_seed_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_multi_seed_performance.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs   = list(ablation_results.keys())
    for ax_idx, metric in enumerate(["accuracy", "F1_macro", "ECE"]):
        vals = [ablation_results[c].get(metric, float("nan")) for c in configs]
        bars = axes[ax_idx].bar(range(len(configs)), vals,
                                color=["#42A5F5" if i == 0 else "#90CAF9"
                                       for i in range(len(configs))],
                                edgecolor="white")
        axes[ax_idx].set_xticks(range(len(configs)))
        axes[ax_idx].set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
        axes[ax_idx].set_ylabel(metric); axes[ax_idx].set_title(f"Ablation: {metric}")
        axes[ax_idx].grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                axes[ax_idx].text(bar.get_x() + bar.get_width()/2,
                                   v + 0.005, f"{v:.3f}",
                                   ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_ablation.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    method_names  = ["Random Forest", "SVM (RBF)", "Logistic Reg.", "HOBIT (ours)"]
    hobit_acc     = mean_res["mean"].get("accuracy", 0)
    hobit_f1      = mean_res["mean"].get("F1_macro", 0)
    hobit_auc     = mean_res["mean"].get("AUC_macro", 0)
    accs = [baseline_results.get(m, {}).get("accuracy", 0) for m in method_names[:-1]] + [hobit_acc]
    f1s  = [baseline_results.get(m, {}).get("F1_macro", 0) for m in method_names[:-1]] + [hobit_f1]
    aucs = [baseline_results.get(m, {}).get("AUC_macro", 0) for m in method_names[:-1]] + [hobit_auc]
    x   = np.arange(len(method_names)); w = 0.25
    ax.bar(x - w, accs, w, label="Accuracy", color="#42A5F5")
    ax.bar(x,     f1s,  w, label="F1 Macro", color="#66BB6A")
    ax.bar(x + w, aucs, w, label="AUC Macro",color="#EF5350")
    ax.set_xticks(x); ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("HOBIT vs Baselines — BRCA PAM50 Classification")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage3_baseline_comparison.png")

    log.checkpoint("stage3_figures_saved")

def generate_stage3_tables(all_seed_results: list,
                            mean_res: dict,
                            baseline_results: dict,
                            ablation_results: dict,
                            log: RunLogger):
    log.section("Generating Stage 3 Tables")
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed, res in zip(C.RANDOM_SEEDS, all_seed_results):
        row = {"Seed": seed}
        for m in ["accuracy","balanced_acc","F1_macro","F1_weighted",
                  "precision","recall","AUC_macro","AUC_weighted","ECE","MCE"]:
            row[m] = round(res.get(m, float("nan")), 4)
        rows.append(row)

    mean_row = {"Seed": "Mean±SD"}
    for m in list(rows[0].keys())[1:]:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        mean_row[m] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rows.append(mean_row)
    df_seeds = pd.DataFrame(rows)
    df_seeds.to_csv(C.TABLES_DIR / "stage3_multiseed_results.csv", index=False)
    log.info(f"  stage3_multiseed_results.csv  ({len(df_seeds)} rows)")

    hobit_mean = mean_res["mean"]
    hobit_std  = mean_res["std"]
    comp_rows = []
    for name in ["Random Forest", "SVM (RBF)", "Logistic Reg."]:
        r = baseline_results.get(name, {})
        comp_rows.append({
            "Method":      name,
            "Accuracy":    f"{r.get('accuracy', float('nan')):.4f}",
            "F1_macro":    f"{r.get('F1_macro',  float('nan')):.4f}",
            "AUC_macro":   f"{r.get('AUC_macro', float('nan')):.4f}",
            "ECE":         "N/A",
        })
    comp_rows.append({
        "Method":    "HOBIT (ours)",
        "Accuracy":  f"{hobit_mean.get('accuracy',0):.4f}±{hobit_std.get('accuracy',0):.4f}",
        "F1_macro":  f"{hobit_mean.get('F1_macro',0):.4f}±{hobit_std.get('F1_macro',0):.4f}",
        "AUC_macro": f"{hobit_mean.get('AUC_macro',0):.4f}±{hobit_std.get('AUC_macro',0):.4f}",
        "ECE":       f"{hobit_mean.get('ECE',0):.4f}±{hobit_std.get('ECE',0):.4f}",
    })
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(C.TABLES_DIR / "stage3_method_comparison.csv", index=False)
    log.info(f"  stage3_method_comparison.csv")

    abl_rows = [
        {"Configuration": cfg,
         "Accuracy":  round(res.get("accuracy",  float("nan")), 4),
         "F1_macro":  round(res.get("F1_macro",  float("nan")), 4),
         "AUC_macro": round(res.get("AUC_macro", float("nan")), 4),
         "ECE":       round(res.get("ECE",        float("nan")), 4)}
        for cfg, res in ablation_results.items()
    ]
    pd.DataFrame(abl_rows).to_csv(C.TABLES_DIR / "stage3_ablation.csv", index=False)
    log.info(f"  stage3_ablation.csv")

    last = all_seed_results[-1]
    pc_rows = [
        {"Class":    C.CLASS_NAMES[i],
         "Precision":round(last["prec_per_class"][i], 4),
         "Recall":   round(last["rec_per_class"][i],  4),
         "F1":       round(last["f1_per_class"][i],   4),
         "Support":  int(np.array(last["confusion_matrix"]).sum(axis=1)[i])}
        for i in range(C.NUM_CLASSES)
    ]
    pd.DataFrame(pc_rows).to_csv(C.TABLES_DIR / "stage3_per_class_performance.csv", index=False)
    log.info(f"  stage3_per_class_performance.csv")

    log.checkpoint("stage3_tables_saved")

def main():
    log = RunLogger("stage3_classifier", C.LOGS_DIR)
    log.section("HOBIT-BRCA  Stage 3 — Bayesian OT GraphSAGE Classifier")

    X_mrna, X_meth, X_mirna, y, valid_samples, rankings_df, splits_data = load_data(log)

    all_seed_results: list = []
    baseline_results_agg: dict = {}

    for seed_idx, seed in enumerate(C.RANDOM_SEEDS):
        log.section(f"Seed {seed_idx+1}/{C.N_SEEDS}  (seed={seed})")

        split = splits_data["splits"][str(seed)]
        tr_idx  = np.array(split["train"])
        val_idx = np.array(split["val"])
        te_idx  = np.array(split["test"])

        X_mrna_tr = X_mrna[tr_idx];  X_mrna_va = X_mrna[val_idx]; X_mrna_te = X_mrna[te_idx]
        X_meth_tr = X_meth[tr_idx];  X_meth_va = X_meth[val_idx]; X_meth_te = X_meth[te_idx]
        X_mir_tr  = X_mirna[tr_idx]; X_mir_va  = X_mirna[val_idx];X_mir_te  = X_mirna[te_idx]

        y_tr = y[tr_idx]; y_va = y[val_idx]; y_te = y[te_idx]

        log.info(f"  Train: {len(tr_idx)}  Val: {len(val_idx)}  Test: {len(te_idx)}")
        for ci, cn in enumerate(C.CLASS_NAMES):
            log.info(f"    test {cn}: {(y_te==ci).sum()}")

        ot_result = compute_ot_plan(X_mrna_tr, X_meth_tr, log)
        X_tr = apply_ot_alignment(X_mrna_tr, X_meth_tr, ot_result, log)
        X_tr = np.concatenate([X_tr, X_mir_tr], axis=1)
        X_va = apply_ot_alignment(X_mrna_va, X_meth_va, ot_result, log)
        X_va = np.concatenate([X_va, X_mir_va], axis=1)
        X_te = apply_ot_alignment(X_mrna_te, X_meth_te, ot_result, log)
        X_te = np.concatenate([X_te, X_mir_te], axis=1)

        in_dim = X_tr.shape[1]
        log.info(f"  Input feature dim (post-OT): {in_dim}")

        model, history = train_one_seed(X_tr, X_va, y_tr, y_va, in_dim, seed, log)

        torch.save(model.state_dict(),
                   C.MODELS_DIR / f"hobit_classifier_seed{seed}.pth")

        Xte_t = torch.tensor(X_te, dtype=torch.float32).to(C.DEVICE)
        mean_probs, std_probs, all_probs = model.mc_predict(Xte_t)

        res = evaluate(mean_probs, y_te, all_probs, seed, log)
        all_seed_results.append(res)

        if seed_idx == 0:
            baseline_results_agg = run_baselines(X_tr, y_tr, X_te, y_te, log)

        if seed_idx == 0:
            ablation_results = run_ablation(
                X_tr, y_tr, X_te, y_te,
                ot_result,
                X_mrna_tr, X_meth_tr, X_mir_tr,
                X_mrna_te, X_meth_te, X_mir_te,
                seed, log
            )

    agg_metrics = ["accuracy", "balanced_acc", "F1_macro", "F1_weighted",
                   "precision", "recall", "AUC_macro", "AUC_weighted", "ECE", "MCE",
                   "MI_mean", "spearman_rho"]
    mean_vals = {m: float(np.mean([r[m] for r in all_seed_results])) for m in agg_metrics}
    std_vals  = {m: float(np.std( [r[m] for r in all_seed_results])) for m in agg_metrics}
    ci95_vals = {m: float(1.96 * std_vals[m] / np.sqrt(C.N_SEEDS)) for m in agg_metrics}

    mean_res = {"mean": mean_vals, "std": std_vals, "ci95": ci95_vals}

    log.section("AGGREGATE RESULTS  (mean ± SD over 5 seeds)")
    for m in agg_metrics:
        log.info(f"  {m:<35}: {mean_vals[m]:.4f} ± {std_vals[m]:.4f}"
                 f"  (95% CI: {mean_vals[m]-ci95_vals[m]:.4f} – "
                 f"{mean_vals[m]+ci95_vals[m]:.4f})")

    generate_stage3_figures(all_seed_results, mean_res, baseline_results_agg,
                            ablation_results, log)
    generate_stage3_tables(all_seed_results, mean_res, baseline_results_agg,
                           ablation_results, log)

    summary = {
        "n_seeds":          C.N_SEEDS,
        "seeds":            C.RANDOM_SEEDS,
        "aggregate":        mean_res,
        "per_seed_accuracy":[r["accuracy"]    for r in all_seed_results],
        "per_seed_ECE":     [r["ECE"]         for r in all_seed_results],
        "per_seed_AUC":     [r["AUC_macro"]   for r in all_seed_results],
        "baselines":        baseline_results_agg,
        "ablation":         ablation_results,
        "ot_used":          _HAS_POT,
        "mc_samples":       C.CLF_N_MC_SAMPLES,
        "n_bins_ece":       C.ECE_N_BINS,
    }
    with open(C.STAGE3_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.close()
    return summary

if __name__ == "__main__":
    main()
