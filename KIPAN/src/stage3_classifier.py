import sys
import json
import pickle
import warnings
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
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

def compute_ot_plan(X_mrna_train: np.ndarray,
                    X_meth_train: np.ndarray,
                    log: RunLogger) -> dict:
    if not _HAS_POT:
        log.warning("  POT not installed — skipping OT; using plain concat")
        return None

    log.info("  Estimating OT plan from training patients (Sinkhorn)")
    n_src, n_tgt = X_mrna_train.shape[1], X_meth_train.shape[1]

    mu_src = X_mrna_train.mean(axis=0)
    mu_tgt = X_meth_train.mean(axis=0)

    max_feat = 2000
    rng = np.random.default_rng(42)
    idx_src = rng.choice(n_src, min(max_feat, n_src), replace=False)
    idx_tgt = rng.choice(n_tgt, min(max_feat, n_tgt), replace=False)

    M = pot.dist(mu_src[idx_src].reshape(-1, 1),
                  mu_tgt[idx_tgt].reshape(-1, 1), metric="sqeuclidean")
    M /= M.max() + 1e-10

    a_s = np.ones(len(idx_src)) / len(idx_src)
    b_s = np.ones(len(idx_tgt)) / len(idx_tgt)

    try:
        T = pot.sinkhorn(a_s, b_s, M,
                         reg=C.OT_EPSILON,
                         numItermax=C.OT_MAX_ITER,
                         stopThr=C.OT_STOP_THRESHOLD)
        log.info(f"  OT plan shape: {T.shape}  (transport matrix, sub-sampled)")
        return {"plan": T, "idx_src": idx_src, "idx_tgt": idx_tgt}
    except Exception as ex:
        log.warning(f"  Sinkhorn failed ({ex}) — plain concat fallback")
        return None

def apply_ot_alignment(X_mrna: np.ndarray,
                        X_meth: np.ndarray,
                        X_mirna: np.ndarray,
                        ot_result: dict,
                        log: RunLogger) -> np.ndarray:
    if ot_result is None:
        result = np.concatenate([X_mrna, X_meth, X_mirna], axis=1).astype(np.float32)
        log.info(f"  No OT — plain concat shape: {result.shape}")
        return result

    T   = ot_result["plan"]
    i_t = ot_result["idx_tgt"]

    tw  = T.sum(axis=0)
    tw /= tw.sum() + 1e-10
    X_meth_aligned = X_meth[:, i_t] * tw[np.newaxis, :]

    result = np.concatenate([X_mrna, X_meth_aligned, X_mirna],
                             axis=1).astype(np.float32)
    log.info(f"  OT-aligned feature shape: {result.shape}")
    return result

def load_data(log: RunLogger):
    log.section("Loading Stage 1/2 data")

    with open(C.GRAPH_DATA_FILE, "rb") as f:
        gd = pickle.load(f)

    rankings_df = pd.read_pickle(C.BIOMARKER_RANK_FILE)
    labels_df   = pd.read_csv(C.SAMPLE_LABELS_FILE, index_col=0)

    with open(C.DATA_SPLITS_FILE) as f:
        splits_data = json.load(f)

    common_samples = gd["common_samples"]
    node_features  = gd["node_features"]
    node_metadata  = gd["node_metadata"]

    def _get_modality(omic_name):
        mask = node_metadata["omic_type"] == omic_name
        ids  = node_metadata[mask]["node_id"].values
        return node_features[ids, 1:].T

    X_mrna  = _get_modality("mRNA")
    X_meth  = _get_modality("methylation")
    X_mirna = _get_modality("miRNA")

    y_list = []
    valid_samples = []
    for s in common_samples:
        if s in labels_df.index:
            subtype = labels_df.loc[s, C.LABEL_COLUMN]
            if subtype in C.CLASS_TO_INT:
                y_list.append(C.CLASS_TO_INT[subtype])
                valid_samples.append(s)

    y = np.array(y_list, dtype=np.int64)
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

    def __init__(self, in_dim: int,
                 hidden_dim: int = C.CLF_HIDDEN_DIM,
                 n_layers: int   = C.CLF_N_LAYERS,
                 dropout: float  = C.CLF_DROPOUT,
                 n_classes: int  = C.NUM_CLASSES):
        super().__init__()
        self.dropout_p = dropout

        self.input_bn   = nn.BatchNorm1d(in_dim)
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

    def mc_predict(self, x: torch.Tensor,
                   n_samples: int = C.CLF_N_MC_SAMPLES):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        self.eval()
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0), preds

def focal_ce_loss(logits, targets,
                  gamma=C.CLF_FOCAL_GAMMA,
                  weight=None,
                  label_smoothing=C.CLF_LABEL_SMOOTHING):
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
                   in_dim, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HOBITClassifier(in_dim).to(C.DEVICE)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(C.DEVICE)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(C.DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long).to(C.DEVICE)
    yv = torch.tensor(y_val,   dtype=torch.long).to(C.DEVICE)

    counts = torch.bincount(yt, minlength=C.NUM_CLASSES).float()
    w = (1.0 / counts.clamp(min=1)) * len(yt) / C.NUM_CLASSES
    w = w.to(C.DEVICE)
    log.info(f"  Class weights: "
             + "  ".join(f"{C.CLASS_NAMES[i]}={w[i].item():.2f}"
                          for i in range(C.NUM_CLASSES)))

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
                log.info(f"  Early stop at epoch {epoch}  best={best_score:.4f}")
                break

    model.load_state_dict(best_state)
    log.info(f"  Training complete  best_val_score={best_score:.4f}")
    return model, history

def compute_calibration(mean_probs: np.ndarray,
                         y_true: np.ndarray,
                         n_bins: int = C.ECE_N_BINS):
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    is_correct  = (predictions == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)

    ece_sum = mce = 0.0
    total = len(y_true)
    rx, ry, sizes, accs, confs = [], [], [], [], []

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc  = is_correct[mask].mean()
        conf = confidences[mask].mean()
        sz   = mask.sum()
        gap  = abs(acc - conf)
        ece_sum += (sz / total) * gap
        mce = max(mce, gap)
        rx.append(float(conf)); ry.append(float(acc))
        sizes.append(int(sz)); accs.append(float(acc)); confs.append(float(conf))

    pred_entropy = -np.sum(mean_probs * np.log2(mean_probs + 1e-10), axis=1)
    return {
        "ECE": float(ece_sum), "MCE": float(mce),
        "pred_entropy": pred_entropy.tolist(),
        "mean_entropy": float(pred_entropy.mean()),
        "reliability_x": rx, "reliability_y": ry,
        "bin_sizes": sizes, "bin_accs": accs, "bin_confs": confs,
    }

def compute_mutual_information(all_probs):
    mean_p = all_probs.mean(axis=0)
    H_mean = -np.sum(mean_p * np.log2(mean_p + 1e-10), axis=1)
    H_ind  = -np.sum(all_probs * np.log2(all_probs + 1e-10), axis=2)
    return np.maximum(H_mean - H_ind.mean(axis=0), 0.0)

def evaluate(mean_probs, y_true, all_probs, seed, log):
    y_pred = mean_probs.argmax(axis=1)
    y_bin  = label_binarize(y_true, classes=list(range(C.NUM_CLASSES)))

    acc    = accuracy_score(y_true, y_pred)
    balacc = balanced_accuracy_score(y_true, y_pred)
    f1m    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    try:
        auc_m = roc_auc_score(y_bin, mean_probs, average="macro",    multi_class="ovr")
        auc_w = roc_auc_score(y_bin, mean_probs, average="weighted", multi_class="ovr")
    except Exception:
        auc_m = auc_w = float("nan")

    calib = compute_calibration(mean_probs, y_true)
    mi    = compute_mutual_information(all_probs)
    pred_entropy = np.array(calib["pred_entropy"])
    errors = (y_pred != y_true).astype(float)
    rho, _ = scipy_stats.spearmanr(pred_entropy, errors)

    cm      = confusion_matrix(y_true, y_pred)
    f1_pc   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_pc  = recall_score(y_true, y_pred,    average=None, zero_division=0)

    log.log_metric(f"[seed={seed}] accuracy",     acc)
    log.log_metric(f"[seed={seed}] balanced_acc", balacc)
    log.log_metric(f"[seed={seed}] F1_macro",     f1m)
    log.log_metric(f"[seed={seed}] AUC_macro",    auc_m)
    log.log_metric(f"[seed={seed}] ECE",          calib["ECE"])
    log.log_metric(f"[seed={seed}] MCE",          calib["MCE"])
    log.log_metric(f"[seed={seed}] MI_mean",      mi.mean())
    log.log_metric(f"[seed={seed}] spearman_rho", float(rho))

    return {
        "accuracy": acc, "balanced_acc": balacc,
        "F1_macro": f1m, "F1_weighted": f1w,
        "precision": prec, "recall": rec,
        "AUC_macro": auc_m, "AUC_weighted": auc_w,
        "ECE": calib["ECE"], "MCE": calib["MCE"],
        "MI_mean": float(mi.mean()),
        "spearman_rho": float(rho),
        "f1_per_class":  f1_pc.tolist(),
        "prec_per_class":prec_pc.tolist(),
        "rec_per_class": rec_pc.tolist(),
        "confusion_matrix": cm.tolist(),
        "calibration": calib,
        "mi": mi.tolist(),
    }

def run_baselines(X_train, y_train, X_test, y_test, log):
    log.section("Baseline Model Comparison")
    baselines = {
        "Random Forest":  RandomForestClassifier(n_estimators=300,
                              max_features="sqrt", random_state=42,
                              class_weight="balanced", n_jobs=-1),
        "SVM (RBF)":      SVC(kernel="rbf", C=10, probability=True,
                               class_weight="balanced", random_state=42),
        "Logistic Reg.":  LogisticRegression(C=1.0, multi_class="ovr",
                              class_weight="balanced",
                              max_iter=2000, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, clf in baselines.items():
        try:
            clf.fit(X_train, y_train)
            yp  = clf.predict(X_test)
            ypr = clf.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=list(range(C.NUM_CLASSES)))
            acc = accuracy_score(y_test, yp)
            f1m = f1_score(y_test, yp, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_bin, ypr, average="macro", multi_class="ovr")
            except Exception:
                auc = float("nan")
            results[name] = {"accuracy": acc, "F1_macro": f1m, "AUC_macro": auc}
            log.info(f"  {name:<22}: Acc={acc:.4f}  F1={f1m:.4f}  AUC={auc:.4f}")
        except Exception as ex:
            log.warning(f"  {name} failed: {ex}")
            results[name] = {"accuracy": float("nan"), "F1_macro": float("nan"),
                             "AUC_macro": float("nan")}
    return results

def run_ablation(X_tr_full, y_tr, X_te_full, y_te,
                 ot_result,
                 X_mrna_tr, X_meth_tr, X_mir_tr,
                 X_mrna_te, X_meth_te, X_mir_te,
                 seed, log):
    log.section("Ablation Study")

    def _quick(X_tr, X_te, label, n_ep=100):
        if X_tr.shape[1] == 0:
            return {"accuracy": float("nan"), "F1_macro": float("nan"),
                    "AUC_macro": float("nan"), "ECE": float("nan")}
        clf = HOBITClassifier(X_tr.shape[1]).to(C.DEVICE)
        torch.manual_seed(seed)
        Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(C.DEVICE)
        Xte_t = torch.tensor(X_te, dtype=torch.float32).to(C.DEVICE)
        yt_t  = torch.tensor(y_tr, dtype=torch.long).to(C.DEVICE)
        opt   = torch.optim.AdamW(clf.parameters(), lr=C.CLF_LR,
                                   weight_decay=C.CLF_WEIGHT_DECAY)
        for ep in range(n_ep):
            clf.train()
            loss = focal_ce_loss(clf(Xtr_t), yt_t)
            opt.zero_grad(); loss.backward(); opt.step()
        mp, _, ap = clf.mc_predict(Xte_t)
        yp    = mp.argmax(axis=1)
        acc   = accuracy_score(y_te, yp)
        f1m   = f1_score(y_te, yp, average="macro", zero_division=0)
        y_bin = label_binarize(y_te, classes=list(range(C.NUM_CLASSES)))
        try:
            auc = roc_auc_score(y_bin, mp, average="macro", multi_class="ovr")
        except Exception:
            auc = float("nan")
        cal = compute_calibration(mp, y_te)
        log.info(f"  {label:<50}: Acc={acc:.4f}  F1={f1m:.4f}  ECE={cal['ECE']:.4f}")
        return {"accuracy": acc, "F1_macro": f1m, "AUC_macro": auc, "ECE": cal["ECE"]}

    X_tr_zs = np.concatenate([X_mrna_tr, X_meth_tr, X_mir_tr], axis=1)
    X_te_zs = np.concatenate([X_mrna_te, X_meth_te, X_mir_te], axis=1)

    results = {
        "Full HOBIT":
            _quick(X_tr_full, X_te_full, "Full HOBIT (OT + Bayesian)"),
        "w/o Optimal Transport":
            _quick(X_tr_zs, X_te_zs, "w/o OT  (Z-score only)"),
        "w/o miRNA":
            _quick(
                apply_ot_alignment(X_mrna_tr, X_meth_tr,
                                   np.zeros((len(X_mrna_tr),0)), ot_result, log),
                apply_ot_alignment(X_mrna_te, X_meth_te,
                                   np.zeros((len(X_mrna_te),0)), ot_result, log),
                "w/o miRNA (OT)"),
        "mRNA only":
            _quick(X_mrna_tr, X_mrna_te, "mRNA only"),
    }
    return results

def generate_stage3_figures(all_seed_results, mean_res, baseline_results,
                             ablation_results, log):
    log.section("Generating Stage 3 Figures")
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    last  = all_seed_results[-1]
    calib = last["calibration"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rx, ry = calib["reliability_x"], calib["reliability_y"]
    if rx:
        axes[0].plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")
        axes[0].plot(rx, ry, "o-", color="#E53935", lw=2, ms=7, label="HOBIT")
        axes[0].fill_between(rx, ry, rx, alpha=0.15, color="#E53935",
                              label=f"ECE={calib['ECE']:.4f}  MCE={calib['MCE']:.4f}")
        axes[0].set_xlabel("Mean Confidence"); axes[0].set_ylabel("Fraction Correct")
        axes[0].set_title("Reliability Diagram (Calibration Curve) — KIPAN RCC")
        axes[0].legend(loc="upper left"); axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)
        axes[0].grid(True, alpha=0.3)
        if calib["bin_confs"]:
            axes[1].bar(calib["bin_confs"], calib["bin_accs"],
                        width=1/C.ECE_N_BINS, color="#42A5F5", alpha=0.7,
                        edgecolor="white", label="Accuracy per bin")
            axes[1].plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")
            axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Calibration Histogram")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_reliability_diagram.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_reliability_diagram.png")

    cm    = np.array(last["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix  Acc={last['accuracy']:.3f}  F1={last['F1_macro']:.3f}")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[1])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Row-Normalised)")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_confusion_matrix.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(C.CLASS_NAMES)); w = 0.25
    ax.bar(x-w, last["prec_per_class"], w, label="Precision", color="#42A5F5")
    ax.bar(x,   last["rec_per_class"],  w, label="Recall",    color="#66BB6A")
    ax.bar(x+w, last["f1_per_class"],   w, label="F1",        color="#EF5350")
    ax.set_xticks(x); ax.set_xticklabels(C.CLASS_NAMES)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance — KIPAN RCC Subtypes")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(last["f1_per_class"]):
        ax.text(i+w, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_per_class_performance.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_per_class_performance.png")

    metrics_p = ["accuracy","balanced_acc","F1_macro","AUC_macro","ECE"]
    fig, ax = plt.subplots(figsize=(12, 6))
    means = [mean_res["mean"].get(m,0) for m in metrics_p]
    stds  = [mean_res["std"].get(m,0)  for m in metrics_p]
    colors = ["#42A5F5","#66BB6A","#EF5350","#AB47BC","#FF7043"]
    ax.bar(metrics_p, means, color=colors, alpha=0.8, capsize=5, yerr=stds)
    ax.set_ylabel("Score")
    ax.set_title(f"HOBIT Performance (mean±SD, N={C.N_SEEDS} seeds) — KIPAN RCC")
    ax.set_ylim(0, 1.05)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m+s+0.01, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_multi_seed_performance.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_multi_seed_performance.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = list(ablation_results.keys())
    for ai, metric in enumerate(["accuracy","F1_macro","ECE"]):
        vals = [ablation_results[c].get(metric, float("nan")) for c in configs]
        bars = axes[ai].bar(range(len(configs)), vals,
                            color=["#42A5F5" if i==0 else "#90CAF9"
                                   for i in range(len(configs))],
                            edgecolor="white")
        axes[ai].set_xticks(range(len(configs)))
        axes[ai].set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
        axes[ai].set_ylabel(metric); axes[ai].set_title(f"Ablation: {metric}")
        axes[ai].grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                axes[ai].text(bar.get_x()+bar.get_width()/2, v+0.005,
                               f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_ablation.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_ablation.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    methods = ["Random Forest","SVM (RBF)","Logistic Reg.","HOBIT (ours)"]
    ho_acc  = mean_res["mean"].get("accuracy", 0)
    ho_f1   = mean_res["mean"].get("F1_macro", 0)
    ho_auc  = mean_res["mean"].get("AUC_macro", 0)
    accs = [baseline_results.get(m,{}).get("accuracy",0) for m in methods[:-1]] + [ho_acc]
    f1s  = [baseline_results.get(m,{}).get("F1_macro", 0) for m in methods[:-1]] + [ho_f1]
    aucs = [baseline_results.get(m,{}).get("AUC_macro",0) for m in methods[:-1]] + [ho_auc]
    x = np.arange(len(methods)); w = 0.25
    ax.bar(x-w, accs, w, label="Accuracy",  color="#42A5F5")
    ax.bar(x,   f1s,  w, label="F1 Macro",  color="#66BB6A")
    ax.bar(x+w, aucs, w, label="AUC Macro", color="#EF5350")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("HOBIT vs Baselines — KIPAN RCC Subtyping")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage3_baseline_comparison.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage3_baseline_comparison.png")

    log.checkpoint("stage3_figures_saved")

def generate_stage3_tables(all_seed_results, mean_res, baseline_results,
                            ablation_results, log):
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
        vals = [r[m] for r in rows if not np.isnan(float(r[m]))]
        mean_row[m] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rows.append(mean_row)
    pd.DataFrame(rows).to_csv(C.TABLES_DIR / "stage3_multiseed_results.csv", index=False)
    log.info("  stage3_multiseed_results.csv")

    hm, hs = mean_res["mean"], mean_res["std"]
    comp = []
    for name in ["Random Forest","SVM (RBF)","Logistic Reg."]:
        r = baseline_results.get(name, {})
        comp.append({"Method": name,
                     "Accuracy":  f"{r.get('accuracy',float('nan')):.4f}",
                     "F1_macro":  f"{r.get('F1_macro',float('nan')):.4f}",
                     "AUC_macro": f"{r.get('AUC_macro',float('nan')):.4f}",
                     "ECE": "N/A"})
    comp.append({"Method": "HOBIT (ours)",
                 "Accuracy":  f"{hm.get('accuracy',0):.4f}±{hs.get('accuracy',0):.4f}",
                 "F1_macro":  f"{hm.get('F1_macro',0):.4f}±{hs.get('F1_macro',0):.4f}",
                 "AUC_macro": f"{hm.get('AUC_macro',0):.4f}±{hs.get('AUC_macro',0):.4f}",
                 "ECE":       f"{hm.get('ECE',0):.4f}±{hs.get('ECE',0):.4f}"})
    pd.DataFrame(comp).to_csv(C.TABLES_DIR / "stage3_method_comparison.csv", index=False)
    log.info("  stage3_method_comparison.csv")

    def _fmt(r, k):
        v  = r.get(k, float("nan"))
        sd = r.get(k+"_std", float("nan"))
        if np.isnan(v): return "nan"
        if np.isnan(sd): return f"{v:.4f}"
        return f"{v:.4f}±{sd:.4f}"

    abl_rows = [{"Configuration": cfg,
                 "Accuracy":  _fmt(r, "accuracy"),
                 "F1_macro":  _fmt(r, "F1_macro"),
                 "AUC_macro": _fmt(r, "AUC_macro"),
                 "ECE":       _fmt(r, "ECE")}
                for cfg, r in ablation_results.items()]
    pd.DataFrame(abl_rows).to_csv(C.TABLES_DIR / "stage3_ablation.csv", index=False)
    log.info("  stage3_ablation.csv")

    last = all_seed_results[-1]
    pc_rows = [{"Class": C.CLASS_NAMES[i],
                "Precision": round(last["prec_per_class"][i], 4),
                "Recall":    round(last["rec_per_class"][i],  4),
                "F1":        round(last["f1_per_class"][i],   4),
                "Support":   int(np.array(last["confusion_matrix"]).sum(axis=1)[i])}
               for i in range(C.NUM_CLASSES)]
    pd.DataFrame(pc_rows).to_csv(C.TABLES_DIR/"stage3_per_class_performance.csv", index=False)
    log.info("  stage3_per_class_performance.csv")

    log.checkpoint("stage3_tables_saved")

def _find_temperature(model: nn.Module,
                       X_val: torch.Tensor,
                       y_val: torch.Tensor,
                       log: RunLogger,
                       T_range: tuple = (0.1, 10.0),
                       n_steps: int   = 200) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X_val)

    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(T_range[0], T_range[1], n_steps):
        scaled = logits / T
        nll = F.cross_entropy(scaled, y_val).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-10
    log_p = np.log(np.clip(probs, eps, 1.0))
    log_p_scaled = log_p / T
    log_p_scaled -= log_p_scaled.max(axis=1, keepdims=True)
    p_scaled = np.exp(log_p_scaled)
    return p_scaled / p_scaled.sum(axis=1, keepdims=True)

def main():
    log = RunLogger("stage3_classifier", C.LOGS_DIR)
    log.section("HOBIT-KIPAN  Stage 3 — Bayesian OT GraphSAGE Classifier (3-class)")

    X_mrna, X_meth, X_mirna, y, valid_samples, rankings_df, splits_data = load_data(log)

    all_seed_results   = []
    baseline_results   = {}
    ablation_results   = {}

    for seed_idx, seed in enumerate(C.RANDOM_SEEDS):
        log.section(f"Seed {seed_idx+1}/{C.N_SEEDS}  (seed={seed})")

        split   = splits_data["splits"][str(seed)]

        sample_to_idx = {s: i for i, s in enumerate(valid_samples)}

        tr_idx  = np.array([sample_to_idx[s] for s in split["train"] if s in sample_to_idx])
        val_idx = np.array([sample_to_idx[s] for s in split["val"]   if s in sample_to_idx])
        te_idx  = np.array([sample_to_idx[s] for s in split["test"]  if s in sample_to_idx])

        X_mrna_tr = X_mrna[tr_idx];  X_mrna_va = X_mrna[val_idx]; X_mrna_te = X_mrna[te_idx]
        X_meth_tr = X_meth[tr_idx];  X_meth_va = X_meth[val_idx]; X_meth_te = X_meth[te_idx]
        X_mir_tr  = X_mirna[tr_idx]; X_mir_va  = X_mirna[val_idx];X_mir_te  = X_mirna[te_idx]

        y_tr = y[tr_idx]; y_va = y[val_idx]; y_te = y[te_idx]

        log.info(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(te_idx)}")
        for ci, cn in enumerate(C.CLASS_NAMES):
            log.info(f"    test {cn}: {(y_te==ci).sum()}")

        ot_result = compute_ot_plan(X_mrna_tr, X_meth_tr, log)

        X_tr_raw = apply_ot_alignment(X_mrna_tr, X_meth_tr, X_mir_tr, ot_result, log)
        X_va_raw = apply_ot_alignment(X_mrna_va, X_meth_va, X_mir_va, ot_result, log)
        X_te_raw = apply_ot_alignment(X_mrna_te, X_meth_te, X_mir_te, ot_result, log)

        in_dim = X_tr_raw.shape[1]
        log.info(f"  Input feature dim (post-OT): {in_dim}")

        model, history = train_one_seed(X_tr_raw, X_va_raw, y_tr, y_va, in_dim, seed, log)

        Xte_t = torch.tensor(X_te_raw, dtype=torch.float32).to(C.DEVICE)
        mean_probs, std_probs, all_probs = model.mc_predict(Xte_t)

        Xva_t = torch.tensor(X_va_raw, dtype=torch.float32).to(C.DEVICE)
        yva_t = torch.tensor(y_va, dtype=torch.long).to(C.DEVICE)
        T_opt = _find_temperature(model, Xva_t, yva_t, log)
        mean_probs_cal = _apply_temperature(mean_probs, T_opt)

        all_probs_cal  = np.stack([_apply_temperature(p, T_opt) for p in all_probs])

        torch.save(model.state_dict(),
                   C.MODELS_DIR / f"hobit_classifier_seed{seed}.pth")

        res = evaluate(mean_probs_cal, y_te, all_probs_cal, seed, log)
        res["temperature"] = float(T_opt)

        calib_raw = compute_calibration(mean_probs, y_te)
        res["ECE_before_calibration"] = calib_raw["ECE"]
        log.info(f"  Temperature T={T_opt:.3f}  ECE: {calib_raw['ECE']:.4f} -> {res['ECE']:.4f}")
        all_seed_results.append(res)

        if seed_idx == 0:
            baseline_results = run_baselines(X_tr_raw, y_tr, X_te_raw, y_te, log)

        seed_ablation = run_ablation(
            X_tr_raw, y_tr, X_te_raw, y_te,
            ot_result,
            X_mrna_tr, X_meth_tr, X_mir_tr,
            X_mrna_te, X_meth_te, X_mir_te,
            seed, log
        )
        for cfg, res in seed_ablation.items():
            if cfg not in ablation_results:
                ablation_results[cfg] = {m: [] for m in res}
            for m, v in res.items():
                ablation_results[cfg][m].append(v)

    ablation_agg = {}
    for cfg, metric_lists in ablation_results.items():
        ablation_agg[cfg] = {}
        for m, vals in metric_lists.items():
            clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            ablation_agg[cfg][m]         = float(np.mean(clean)) if clean else float("nan")
            ablation_agg[cfg][m+"_std"]  = float(np.std(clean))  if clean else float("nan")
    ablation_results = ablation_agg

    agg_metrics = ["accuracy","balanced_acc","F1_macro","F1_weighted",
                   "precision","recall","AUC_macro","AUC_weighted","ECE","MCE",
                   "MI_mean","spearman_rho"]
    mean_vals = {m: float(np.nanmean([r[m] for r in all_seed_results])) for m in agg_metrics}
    std_vals  = {m: float(np.nanstd( [r[m] for r in all_seed_results])) for m in agg_metrics}
    ci95_vals = {m: float(1.96*std_vals[m]/np.sqrt(C.N_SEEDS)) for m in agg_metrics}
    mean_res  = {"mean": mean_vals, "std": std_vals, "ci95": ci95_vals}

    log.section("AGGREGATE RESULTS  (mean ± SD over 5 seeds)")
    for m in agg_metrics:
        log.info(f"  {m:<35}: {mean_vals[m]:.4f} ± {std_vals[m]:.4f}"
                 f"  95%CI [{mean_vals[m]-ci95_vals[m]:.4f},"
                 f" {mean_vals[m]+ci95_vals[m]:.4f}]")

    generate_stage3_figures(all_seed_results, mean_res, baseline_results,
                            ablation_results, log)
    generate_stage3_tables(all_seed_results, mean_res, baseline_results,
                           ablation_results, log)

    summary = {
        "n_seeds": C.N_SEEDS, "seeds": C.RANDOM_SEEDS,
        "aggregate": mean_res,
        "per_seed_accuracy": [r["accuracy"]  for r in all_seed_results],
        "per_seed_ECE":      [r["ECE"]       for r in all_seed_results],
        "per_seed_AUC":      [r["AUC_macro"] for r in all_seed_results],
        "baselines":         baseline_results,
        "ablation":          ablation_results,
        "ot_used":           _HAS_POT,
        "mc_samples":        C.CLF_N_MC_SAMPLES,
        "n_bins_ece":        C.ECE_N_BINS,
        "n_classes":         C.NUM_CLASSES,
        "class_names":       C.CLASS_NAMES,
    }
    with open(C.STAGE3_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.close()
    return summary

if __name__ == "__main__":
    main()
