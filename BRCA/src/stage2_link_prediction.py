import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
import networkx as nx

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

try:
    import dgl
    import dgl.nn as dglnn
    _BACKEND = "dgl"
except ImportError:
    raise ImportError("DGL is required. Install: pip install dgl")

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.dropout  = nn.Dropout(dropout)
        dims = [in_feats] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(dglnn.SAGEConv(dims[i], dims[i+1], "mean"))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))

    def forward(self, g, x):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(g, h)
            h = bn(h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

class EdgePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_src, h_dst):
        return self.net(torch.cat([h_src, h_dst], dim=1)).squeeze(-1)

class LinkPredModel(nn.Module):
    def __init__(self, in_feats, hidden_dim,
                 num_layers=C.LP_NUM_LAYERS, dropout=C.LP_DROPOUT):
        super().__init__()
        self.encoder   = GraphSAGEEncoder(in_feats, hidden_dim, num_layers, dropout)
        self.predictor = EdgePredictor(hidden_dim)

    def encode(self, g, x):
        return self.encoder(g, x)

    def predict(self, h, src, dst):
        return self.predictor(h[src], h[dst])

def load_stage1_outputs(log: RunLogger):
    log.section("Loading Stage 1 outputs")

    with open(C.GRAPH_DATA_FILE, "rb") as f:
        gd = pickle.load(f)

    with open(C.RESULTS_DIR / "data_splits.json") as f:
        splits_data = json.load(f)

    primary_seed = str(C.RANDOM_SEEDS[0])
    train_patient_idx = np.array(splits_data["splits"][primary_seed]["train"])

    log.info(f"  Nodes        : {gd['statistics']['total_nodes']:,}")
    log.info(f"  Edges        : {gd['statistics']['total_edges']:,}")
    log.info(f"  Samples      : {gd['statistics']['num_samples']}")
    log.info(f"  Train patients (primary seed): {len(train_patient_idx)}")

    labels_df = pd.read_csv(C.SAMPLE_LABELS_FILE, index_col=0)

    return gd, train_patient_idx, labels_df, splits_data

def build_training_graph(gd: dict, train_patient_idx: np.ndarray, log: RunLogger):
    log.section("Building training-patient DGL graph")

    feat_full = gd["node_features"]

    train_cols = np.concatenate([[0], train_patient_idx + 1])
    feat_train = feat_full[:, train_cols]

    edge_index = gd["edge_index"]
    src = torch.tensor(edge_index[0], dtype=torch.long)
    dst = torch.tensor(edge_index[1], dtype=torch.long)

    g = dgl.graph((src, dst), num_nodes=feat_full.shape[0])
    g.ndata["feat"]   = torch.tensor(feat_train, dtype=torch.float32)
    g.edata["weight"] = torch.tensor(gd["edge_weights"], dtype=torch.float32)

    log.info(f"  Training-graph  nodes  : {g.num_nodes():,}")
    log.info(f"  Training-graph  edges  : {g.num_edges():,}")
    log.info(f"  Node feature dim       : {g.ndata['feat'].shape[1]}")
    return g

def split_edges(g, log: RunLogger, seed=C.RANDOM_SEEDS[0]):
    log.section("Edge-level split (80/10/10) on training graph")
    rng = np.random.default_rng(seed)

    src_np, dst_np = g.edges()
    src_np, dst_np = src_np.numpy(), dst_np.numpy()

    seen = set()
    uniq_src, uniq_dst = [], []
    for s, d in zip(src_np, dst_np):
        key = (min(s, d), max(s, d))
        if key not in seen:
            seen.add(key)
            uniq_src.append(s)
            uniq_dst.append(d)

    n_pos = len(uniq_src)
    perm  = rng.permutation(n_pos)
    n_tr  = int(n_pos * C.LP_TRAIN_FRAC)
    n_va  = int(n_pos * C.LP_VAL_FRAC)

    tr_perm = perm[:n_tr]
    va_perm = perm[n_tr:n_tr+n_va]
    te_perm = perm[n_tr+n_va:]

    pos = {
        "train": (torch.tensor([uniq_src[i] for i in tr_perm]),
                  torch.tensor([uniq_dst[i] for i in tr_perm])),
        "val":   (torch.tensor([uniq_src[i] for i in va_perm]),
                  torch.tensor([uniq_dst[i] for i in va_perm])),
        "test":  (torch.tensor([uniq_src[i] for i in te_perm]),
                  torch.tensor([uniq_dst[i] for i in te_perm])),
    }

    n_nodes = g.num_nodes()
    existing = set(zip(src_np, dst_np))

    def _neg(n, existing, n_nodes, rng):
        neg_s, neg_d = [], []
        while len(neg_s) < n:
            batch_s = rng.integers(0, n_nodes, n * 3)
            batch_d = rng.integers(0, n_nodes, n * 3)
            for s, d in zip(batch_s, batch_d):
                if s != d and (int(s), int(d)) not in existing:
                    neg_s.append(int(s)); neg_d.append(int(d))
                    if len(neg_s) >= n:
                        break
        return torch.tensor(neg_s[:n]), torch.tensor(neg_d[:n])

    neg = {
        split: _neg(len(pos[split][0]), existing, n_nodes, rng)
        for split in ["train", "val", "test"]
    }

    for split in ["train", "val", "test"]:
        log.info(f"  {split:5s}: +{len(pos[split][0]):,}  -{len(neg[split][0]):,}")

    log.checkpoint("edges_split")
    return pos, neg, existing

def train_link_prediction(g, pos, neg, log: RunLogger):
    log.section("Training Link Prediction Model")

    in_feats  = g.ndata["feat"].shape[1]
    model     = LinkPredModel(in_feats, C.LP_HIDDEN_DIM).to(C.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=C.LP_LR, weight_decay=C.LP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C.LP_EPOCHS, eta_min=1e-5)

    g_dev = g.to(C.DEVICE)
    feats = g_dev.ndata["feat"]

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    history        = {"train_loss": [], "val_auc": [], "val_ap": [], "val_epochs": []}

    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"  Device    : {C.DEVICE}")

    for epoch in range(C.LP_EPOCHS):
        model.train()
        h = model.encode(g_dev, feats)

        ps, pd = pos["train"]
        ns, nd = neg["train"]
        ps, pd, ns, nd = ps.to(C.DEVICE), pd.to(C.DEVICE), ns.to(C.DEVICE), nd.to(C.DEVICE)

        pos_scores = model.predict(h, ps, pd)
        neg_scores = model.predict(h, ns, nd)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(len(ps)), torch.zeros(len(ns))]).to(C.DEVICE)
        loss   = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["train_loss"].append(loss.item())

        if epoch % 10 == 0 or epoch == C.LP_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                h_val = model.encode(g_dev, feats)
                vs, vd = pos["val"]
                vns, vnd = neg["val"]
                vs, vd   = vs.to(C.DEVICE), vd.to(C.DEVICE)
                vns, vnd = vns.to(C.DEVICE), vnd.to(C.DEVICE)

                vp_sc = torch.sigmoid(model.predict(h_val, vs, vd)).cpu().numpy()
                vn_sc = torch.sigmoid(model.predict(h_val, vns, vnd)).cpu().numpy()
                v_scores = np.concatenate([vp_sc, vn_sc])
                v_labels = np.concatenate([np.ones(len(vs)), np.zeros(len(vns))])

                val_auc = roc_auc_score(v_labels, v_scores)
                val_ap  = average_precision_score(v_labels, v_scores)

            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            history["val_epochs"].append(epoch)
            improved = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
                improved = "  ←"
            else:
                patience_count += 1

            if epoch % 50 == 0 or improved:
                log.info(f"  ep {epoch:3d} | loss={loss.item():.4f}"
                         f" | val_AUC={val_auc:.4f} | val_AP={val_ap:.4f}{improved}")

            if patience_count >= C.LP_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    log.checkpoint("link_pred_trained")

    model.eval()
    g_dev2 = g.to(C.DEVICE)
    with torch.no_grad():
        h_te = model.encode(g_dev2, g_dev2.ndata["feat"])
        ts, td = pos["test"]
        tns, tnd = neg["test"]
        tp_sc = torch.sigmoid(model.predict(h_te, ts.to(C.DEVICE), td.to(C.DEVICE))).cpu().numpy()
        tn_sc = torch.sigmoid(model.predict(h_te, tns.to(C.DEVICE), tnd.to(C.DEVICE))).cpu().numpy()
        t_sc  = np.concatenate([tp_sc, tn_sc])
        t_lab = np.concatenate([np.ones(len(ts)), np.zeros(len(tns))])

    test_auc = roc_auc_score(t_lab, t_sc)
    test_ap  = average_precision_score(t_lab, t_sc)
    log.log_metric("link_pred_test_AUC", test_auc)
    log.log_metric("link_pred_test_AP",  test_ap)
    log.log_metric("test_pos_edges",     len(ts),  fmt="d")
    log.log_metric("test_neg_edges",     len(tns), fmt="d")

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.LINK_MODEL_FILE)
    log.info(f"  Model saved → {C.LINK_MODEL_FILE}")

    return model, history, {"test_auc": test_auc, "test_ap": test_ap,
                             "best_val_auc": best_val_auc}

def predict_all_edges(model, g, existing_edges: set, log: RunLogger):
    log.section("Predicting new edges for graph completion")

    model.eval()
    g_dev = g.to(C.DEVICE)
    with torch.no_grad():
        h = model.encode(g_dev, g_dev.ndata["feat"])

    src_np, dst_np = g.edges()
    src_np, dst_np = src_np.numpy(), dst_np.numpy()

    batch_size = 50_000
    existing_scores = []
    for i in range(0, len(src_np), batch_size):
        bs = torch.tensor(src_np[i:i+batch_size]).to(C.DEVICE)
        bd = torch.tensor(dst_np[i:i+batch_size]).to(C.DEVICE)
        with torch.no_grad():
            sc = torch.sigmoid(model.predict(h, bs, bd)).cpu().numpy()
        existing_scores.extend(sc.tolist())

    edge_scores = np.array(existing_scores, dtype=np.float32)
    np.save(C.EDGE_PRED_FILE, edge_scores)

    n_nodes    = g.num_nodes()
    n_existing = len(src_np)
    n_sample   = min(n_existing * 3, 5_000_000)

    rng     = np.random.default_rng(42)
    new_src_list, new_dst_list = [], []
    batch   = 200_000

    sampled = 0
    while sampled < n_sample:
        this_batch = min(batch, n_sample - sampled)
        cs = rng.integers(0, n_nodes, this_batch)
        cd = rng.integers(0, n_nodes, this_batch)

        candidate_mask = np.array(
            [s != d and (int(s), int(d)) not in existing_edges
             for s, d in zip(cs, cd)]
        )
        cs = cs[candidate_mask]
        cd = cd[candidate_mask]

        if len(cs) == 0:
            sampled += this_batch
            continue

        bs = torch.tensor(cs, dtype=torch.long).to(C.DEVICE)
        bd = torch.tensor(cd, dtype=torch.long).to(C.DEVICE)
        with torch.no_grad():
            sc = torch.sigmoid(model.predict(h, bs, bd)).cpu().numpy()

        keep = sc >= C.LP_PRED_THRESHOLD
        new_src_list.extend(cs[keep].tolist())
        new_dst_list.extend(cd[keep].tolist())
        sampled += this_batch

    new_src = np.array(new_src_list, dtype=np.int64)
    new_dst = np.array(new_dst_list, dtype=np.int64)
    n_new   = len(new_src)

    density_increase = n_new / max(n_existing, 1) * 100

    if n_new > 0:
        np.save(C.RESULTS_DIR / "predicted_edges_added.npy",
                np.stack([new_src, new_dst], axis=0))
    else:
        np.save(C.RESULTS_DIR / "predicted_edges_added.npy",
                np.zeros((2, 0), dtype=np.int64))

    log.info(f"  Scored {n_existing:,} existing edges")
    log.info(f"  Sampled {n_sample:,} candidate non-edge pairs")
    log.info(f"  Genuinely NEW edges (score≥{C.LP_PRED_THRESHOLD}): {n_new:,}")
    log.log_metric("predicted_edges_added",         n_new, fmt="d")
    log.log_metric("network_density_increase_pct",  density_increase, fmt=".1f")
    log.log_metric("existing_edges",                n_existing, fmt="d")

    log.checkpoint("edges_predicted")
    return edge_scores, new_src, new_dst

def compute_centralities_on_training_graph(gd: dict,
                                           train_patient_idx: np.ndarray,
                                           log: RunLogger) -> pd.DataFrame:
    log.section("Computing centralities on training graph")

    node_df = gd["node_metadata"]
    n_nodes = len(node_df)

    feat_full = gd["node_features"]
    edge_index = gd["edge_index"]
    src_np, dst_np = edge_index[0], edge_index[1]

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for s, d in zip(src_np, dst_np):
        if not G.has_edge(int(s), int(d)):
            G.add_edge(int(s), int(d))

    log.info(f"  NetworkX graph: {G.number_of_nodes():,} nodes, "
             f"{G.number_of_edges():,} edges")

    degrees = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1
    wdc = np.array([degrees.get(i, 0) / max(max_deg, 1) for i in range(n_nodes)])

    try:
        pr_dict = nx.pagerank(G, max_iter=200)
        pr = np.array([pr_dict.get(i, 0.0) for i in range(n_nodes)])
    except Exception:
        pr = wdc.copy()
        log.warning("  PageRank failed, using degree centrality as fallback")

    try:
        n_samp = min(500, n_nodes)
        bc_dict = nx.betweenness_centrality(G, k=n_samp, normalized=True, seed=42)
        bc = np.array([bc_dict.get(i, 0.0) for i in range(n_nodes)])
    except Exception:
        bc = np.zeros(n_nodes)
        log.warning("  Betweenness centrality failed, using zeros")

    hybrid = (C.GMM_ALPHA_WDC * wdc +
              C.GMM_BETA_PR   * pr  +
              C.GMM_GAMMA_BC  * bc)

    log.info(f"  Hybrid score range: [{hybrid.min():.4f}, {hybrid.max():.4f}]")

    df = node_df.copy()
    df["wdc"]          = wdc
    df["pagerank"]     = pr
    df["betweenness"]  = bc
    df["hybrid_score"] = hybrid

    log.checkpoint("centralities_computed")
    return df

def fit_gmm_importance(centrality_df: pd.DataFrame, log: RunLogger):
    log.section("Fitting GMM on training-graph centrality scores")

    scores = centrality_df["hybrid_score"].values.reshape(-1, 1)

    best_bic, best_k, best_gmm = np.inf, 3, None
    bic_scores = {}
    for k in C.GMM_K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=300)
        gmm.fit(scores)
        bic = gmm.bic(scores)
        bic_scores[k] = bic
        log.info(f"  k={k}: BIC={bic:.2f}")
        if bic < best_bic:
            best_bic, best_k, best_gmm = bic, k, gmm

    log.info(f"  Selected k={best_k}  (BIC={best_bic:.2f})")

    probs  = best_gmm.predict_proba(scores)
    labels = best_gmm.predict(scores)

    means = best_gmm.means_.flatten()
    order = np.argsort(means)
    label_map = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([label_map[l] for l in labels])
    probs_sorted  = probs[:, order]

    centrality_df = centrality_df.copy()
    centrality_df["gmm_label"]      = labels_sorted
    centrality_df["gmm_confidence"] = probs_sorted.max(axis=1)
    centrality_df["prob_high"]      = probs_sorted[:, -1]

    if best_k == 2:
        label_names = {0: "Low", 1: "High"}
    elif best_k == 3:
        label_names = {0: "Low", 1: "Medium", 2: "High"}
    elif best_k == 4:
        label_names = {0: "Low", 1: "Medium-Low", 2: "Medium-High", 3: "High"}
    else:
        # General case: only label the top group "High", rest as Group_i
        label_names = {i: f"Group_{i}" for i in range(best_k)}
        label_names[best_k - 1] = "High"   # top group is always "High"
        label_names[0]          = "Low"     # bottom group is always "Low"
    centrality_df["importance_label"] = centrality_df["gmm_label"].map(label_names)

    for name, cnt in centrality_df["importance_label"].value_counts().items():
        log.info(f"  {name:<10}: {cnt:,}  ({100*cnt/len(centrality_df):.1f}%)")

    mrna_nodes = centrality_df[centrality_df["omic_type"] == "mRNA"]
    high_genes = set(mrna_nodes[mrna_nodes["importance_label"] == "High"]["name"])
    cgc_in_data = C.COSMIC_CGC_GENES & set(mrna_nodes["name"])
    cgc_recovered = C.COSMIC_CGC_GENES & high_genes
    pam50_in_data  = C.PAM50_GENE_SET  & set(mrna_nodes["name"])
    pam50_recovered= C.PAM50_GENE_SET  & high_genes
    recovery_rate  = len(cgc_recovered) / max(len(cgc_in_data), 1)
    pam50_rate     = len(pam50_recovered)/ max(len(pam50_in_data), 1)

    log.log_metric("CGC_genes_in_dataset",     len(cgc_in_data), fmt="d")
    log.log_metric("CGC_genes_recovered",      len(cgc_recovered), fmt="d")
    log.log_metric("CGC_recovery_rate",        recovery_rate)
    log.log_metric("PAM50_genes_in_dataset",   len(pam50_in_data), fmt="d")
    log.log_metric("PAM50_genes_recovered",    len(pam50_recovered), fmt="d")
    log.log_metric("PAM50_recovery_rate",      pam50_rate)

    gmm_data = {
        "gmm_model":   best_gmm,
        "n_components":best_k,
        "bic_scores":  bic_scores,
        "label_names": label_names,
        "means":       means.tolist(),
        "order":       order.tolist(),
        "validation": {
            "cgc_in_data":      len(cgc_in_data),
            "cgc_recovered":    len(cgc_recovered),
            "cgc_recovery_rate":round(recovery_rate, 4),
            "pam50_in_data":    len(pam50_in_data),
            "pam50_recovered":  len(pam50_recovered),
            "pam50_rate":       round(pam50_rate, 4),
        },
    }
    with open(C.GMM_MODEL_FILE, "wb") as f:
        pickle.dump(gmm_data, f)
    log.info(f"  GMM model saved → {C.GMM_MODEL_FILE}")

    log.checkpoint("gmm_fitted")
    return centrality_df, gmm_data

def rank_biomarkers(centrality_df: pd.DataFrame,
                    edge_scores: np.ndarray,
                    gd: dict,
                    log: RunLogger) -> pd.DataFrame:
    log.section("Building Confidence-Penalized Biomarker Ranking")

    df = centrality_df.copy()
    df["hybrid_score_raw"] = df["hybrid_score"].copy()

    for omic in df["omic_type"].unique():
        mask = df["omic_type"] == omic
        vals = df.loc[mask, "hybrid_score"].values
        n    = len(vals)

        from scipy.stats import rankdata
        pct = (rankdata(vals, method="average") - 1) / max(n - 1, 1)
        df.loc[mask, "hybrid_score"] = pct.astype(np.float64)

    log.info("  Within-omic percentile-normalised hybrid scores:")
    for omic in df["omic_type"].unique():
        mask = df["omic_type"] == omic
        log.info(f"    {omic:<16}: "
                 f"max={df.loc[mask,'hybrid_score'].max():.4f}  "
                 f"mean={df.loc[mask,'hybrid_score'].mean():.4f}")

    entropy = -df["prob_high"] * np.log2(df["prob_high"] + 1e-10)
    df["entropy"] = entropy

    df["confidence_penalized_score"] = (
        df["hybrid_score"] * (1 - C.UNCERTAINTY_PENALTY_WEIGHT * entropy)
    )

    df["in_pam50_signature"] = df["name"].isin(C.PAM50_GENE_SET)
    df["in_cosmic_cgc"]      = df["name"].isin(C.COSMIC_CGC_GENES)

    df = df.sort_values("confidence_penalized_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)

    df.to_pickle(C.BIOMARKER_RANK_FILE)
    df.to_csv(C.RESULTS_DIR / "biomarker_importance_rankings.csv", index=False)

    # Top 20 high-importance — use the top label name from GMM (always "High" by our naming)
    high_label = "High"   # our label_names convention always sets highest group to "High"
    high = df[df["importance_label"] == high_label].head(20)
    log.info("  Top 20 High-Importance Biomarkers:")
    log.info(f"  {'Rank':<6}{'Name':<18}{'Type':<16}"
             f"{'Score':<12}{'Conf':<10}{'CGC':<8}{'PAM50'}")
    for _, r in high.iterrows():
        log.info(f"  {int(r['rank']):<6}{r['name']:<18}{r['omic_type']:<16}"
                 f"{r['confidence_penalized_score']:<12.4f}"
                 f"{r['gmm_confidence']:<10.3f}"
                 f"{'✓' if r['in_cosmic_cgc'] else '':<8}"
                 f"{'✓' if r['in_pam50_signature'] else ''}")

    log.checkpoint("biomarkers_ranked")
    return df

def generate_stage2_figures(centrality_df: pd.DataFrame,
                             ranking_df: pd.DataFrame,
                             gmm_data: dict,
                             history: dict,
                             lp_metrics: dict,
                             log: RunLogger):
    log.section("Generating Stage 2 Figures")
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], color="#3498db", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Link Prediction Training Loss")
    axes[0].grid(True, alpha=0.3)

    val_epochs = history.get("val_epochs", [])
    val_auc    = history.get("val_auc", [])
    val_ap     = history.get("val_ap", [])

    if val_epochs and val_auc and len(val_epochs) == len(val_auc):
        axes[1].plot(val_epochs, val_auc, "o-", color="#e74c3c", lw=2, label="AUC-ROC")
        axes[1].plot(val_epochs, val_ap,  "s-", color="#2ecc71", lw=2, label="Avg Precision")
        axes[1].axhline(lp_metrics["test_auc"], ls="--", color="#e74c3c", alpha=0.5,
                        label=f"Test AUC={lp_metrics['test_auc']:.3f}")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
        axes[1].set_title("Link Prediction Validation Metrics")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No validation data recorded",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage2_link_prediction_training.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage2_link_prediction_training.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scores = centrality_df["hybrid_score"].values

    label_names  = gmm_data["label_names"]
    unique_labels = sorted(centrality_df["importance_label"].unique())
    base_colors  = ["#2196F3", "#FF9800", "#FF5722", "#F44336"]
    palette = {}
    for i, lbl in enumerate(sorted(label_names.values())):
        palette[lbl] = base_colors[i % len(base_colors)]
    # Ensure "High" is always red, "Low" always blue
    if "High" in palette: palette["High"] = "#F44336"
    if "Low"  in palette: palette["Low"]  = "#2196F3"

    node_colors = centrality_df["importance_label"].map(
        lambda x: palette.get(x, "#9E9E9E"))

    axes[0].scatter(range(len(scores)), scores, c=node_colors, s=1, alpha=0.5)
    axes[0].set_xlabel("Node Index"); axes[0].set_ylabel("Hybrid Importance Score")
    axes[0].set_title("GMM-Based Biomarker Importance Stratification")

    from matplotlib.patches import Patch
    legend_els = [Patch(color=palette[lbl], label=lbl) for lbl in unique_labels
                  if lbl in palette]
    axes[0].legend(handles=legend_els)

    k_vals = list(gmm_data["bic_scores"].keys())
    bics   = list(gmm_data["bic_scores"].values())
    bar_colors_bic = ["#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"]
    axes[1].bar(k_vals, bics, color=bar_colors_bic[:len(k_vals)])
    axes[1].set_xlabel("Number of GMM Components (k)")
    axes[1].set_ylabel("BIC Score (lower = better)")
    axes[1].set_title("GMM Model Selection via BIC")
    axes[1].set_xticks(k_vals)
    # Label just above the bar top (bars go negative, so "above" = closer to 0)
    y_range = max(bics) - min(bics)
    offset  = y_range * 0.01 if y_range > 0 else abs(min(bics)) * 0.01
    for k, b in zip(k_vals, bics):
        axes[1].text(k, b + offset, f"{b:.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage2_gmm_importance.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage2_gmm_importance.png")

    label_names  = gmm_data["label_names"]
    top_label    = label_names[max(label_names.keys())]   # highest index = "High"
    high_df      = ranking_df[ranking_df["importance_label"] == top_label].head(25)
    high_df      = high_df.sort_values("confidence_penalized_score", ascending=True)

    if len(high_df) == 0:
        log.warning(f"  No nodes labelled '{top_label}' — plotting top-25 by hybrid_score instead")
        high_df = ranking_df.nlargest(25, "hybrid_score")
        high_df = high_df.sort_values("hybrid_score", ascending=True)
        score_col = "hybrid_score"
    else:
        score_col = "confidence_penalized_score"

    color_map  = {"mRNA": "#E53935", "methylation": "#1E88E5", "miRNA": "#43A047"}
    bar_colors = [color_map.get(t, "#757575") for t in high_df["omic_type"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(high_df) * 0.38)))
    ax.barh(high_df["name"], high_df[score_col], color=bar_colors, alpha=0.85)
    ax.set_xlabel("Confidence-Penalized Importance Score")
    ax.set_title(f"Top {len(high_df)} High-Importance BRCA Biomarkers (HOBIT)")
    legend_els = [Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=legend_els, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage2_top_biomarkers.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage2_top_biomarkers.png")

    log.checkpoint("stage2_figures_saved")

def main():
    log = RunLogger("stage2_link_prediction", C.LOGS_DIR)
    log.section("HOBIT-BRCA  Stage 2 — Link Prediction & GMM Importance")

    gd, train_patient_idx, labels_df, splits_data = load_stage1_outputs(log)

    g = build_training_graph(gd, train_patient_idx, log)

    pos, neg, existing_edges = split_edges(g, log)

    model, history, lp_metrics = train_link_prediction(g, pos, neg, log)

    edge_scores, new_src, new_dst = predict_all_edges(model, g, existing_edges, log)

    centrality_df = compute_centralities_on_training_graph(gd, train_patient_idx, log)

    centrality_df, gmm_data = fit_gmm_importance(centrality_df, log)

    ranking_df = rank_biomarkers(centrality_df, edge_scores, gd, log)

    generate_stage2_figures(centrality_df, ranking_df, gmm_data, history, lp_metrics, log)

    label_counts = ranking_df["importance_label"].value_counts().to_dict()
    summary = {
        "link_prediction": lp_metrics,
        "edges_added": int(len(new_src)),
        "network_density_increase_pct": round(len(new_src) / max(gd["statistics"]["total_edges"], 1) * 100, 2),
        "gmm_k": gmm_data["n_components"],
        "gmm_validation": gmm_data["validation"],
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "n_high_importance": int(label_counts.get("High", 0)),
    }
    with open(C.STAGE2_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    log.log_metric("link_pred_test_AUC",          lp_metrics["test_auc"])
    log.log_metric("link_pred_test_AP",            lp_metrics["test_ap"])
    log.log_metric("predicted_edges_added",        len(new_src), fmt="d")
    log.log_metric("GMM_k_selected",               gmm_data["n_components"], fmt="d")
    log.log_metric("high_importance_biomarkers",
                   (ranking_df["importance_label"] == "High").sum(), fmt="d")

    log.close()
    return summary

if __name__ == "__main__":
    main()
