import os
import sys
import json
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

def _zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu  = np.nanmean(X, axis=1, keepdims=True)
    sig = np.nanstd(X, axis=1, keepdims=True)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return mu, sig

def _zscore_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sig
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)

def load_omics_data(log: RunLogger):
    log.section("STEP 1 — Loading BRCA Omics Data")

    mrna_df  = pd.read_csv(C.MRNA_FILE,        index_col=0)
    meth_df  = pd.read_csv(C.METHYLATION_FILE, index_col=0)
    mirna_df = pd.read_csv(C.MIRNA_FILE,       index_col=0)
    labels_df= pd.read_csv(C.LABELS_FILE,      index_col=0)

    log.info(f"  mRNA         : {mrna_df.shape}  (genes × samples)")
    log.info(f"  Methylation  : {meth_df.shape}  (CpG × samples)")
    log.info(f"  miRNA        : {mirna_df.shape}  (miRNAs × samples)")
    log.info(f"  Labels       : {labels_df.shape}")

    common = (set(mrna_df.columns) & set(meth_df.columns)
              & set(mirna_df.columns) & set(labels_df.index))
    common = sorted(common)
    log.info(f"  Common samples: {len(common)}")

    mrna_df   = mrna_df[common]
    meth_df   = meth_df[common]
    mirna_df  = mirna_df[common]
    labels_df = labels_df.loc[common]

    valid_mask = labels_df["PAM50"].isin(C.CLASS_TO_INT)
    if not valid_mask.all():
        log.warning(f"  Dropping {(~valid_mask).sum()} samples with unknown PAM50")
    labels_df = labels_df[valid_mask]
    common = list(labels_df.index)
    mrna_df  = mrna_df[common]
    meth_df  = meth_df[common]
    mirna_df = mirna_df[common]

    counts = labels_df["PAM50"].value_counts()
    for sub, cnt in counts.items():
        log.info(f"    {sub:8s}: {cnt:4d}  ({100*cnt/len(common):.1f}%)")

    log.checkpoint("data_loaded")
    return mrna_df, meth_df, mirna_df, labels_df, common

def make_patient_splits(labels: np.ndarray, common_samples: List[str], log: RunLogger):
    log.section("STEP 2 — Patient-Level Stratified Split (pre-normalisation)")
    log.info(f"  Strategy : 70/15/15 stratified, {C.N_SEEDS} seeds")

    all_splits = {}
    n = len(labels)
    idx = np.arange(n)

    for seed in C.RANDOM_SEEDS:
        sss1 = StratifiedShuffleSplit(n_splits=1,
                                      test_size=C.VAL_FRAC + C.TEST_FRAC,
                                      random_state=seed)
        train_idx, tmp_idx = next(sss1.split(idx, labels))

        sss2 = StratifiedShuffleSplit(n_splits=1,
                                      test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                                      random_state=seed)
        val_local, test_local = next(sss2.split(tmp_idx, labels[tmp_idx]))
        val_idx  = tmp_idx[val_local]
        test_idx = tmp_idx[test_local]

        all_splits[str(seed)] = {
            "train": train_idx.tolist(),
            "val":   val_idx.tolist(),
            "test":  test_idx.tolist(),
        }
        log.info(f"  seed={seed}  train={len(train_idx)}  val={len(val_idx)}"
                 f"  test={len(test_idx)}")
        for cls_idx, cls_name in C.INT_TO_CLASS.items():
            tr = (labels[train_idx] == cls_idx).sum()
            va = (labels[val_idx]   == cls_idx).sum()
            te = (labels[test_idx]  == cls_idx).sum()
            log.debug(f"    {cls_name}: train={tr}  val={va}  test={te}")

    splits_path = C.RESULTS_DIR / "data_splits.json"
    with open(splits_path, "w") as f:
        json.dump({
            "common_samples": common_samples,
            "splits": all_splits,
            "train_frac": C.TRAIN_FRAC,
            "val_frac":   C.VAL_FRAC,
            "test_frac":  C.TEST_FRAC,
        }, f, indent=2)
    log.info(f"  Splits saved → {splits_path}")
    log.checkpoint("splits_created")
    return all_splits

def _detect_col(df: pd.DataFrame, candidates: list, file_label: str, log: RunLogger) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    log.error(f"  [{file_label}] Could not find any of {candidates} in columns: {list(df.columns)}")
    raise KeyError(f"[{file_label}] Expected one of {candidates}, got {list(df.columns)}")

def load_string_ppi(log: RunLogger):
    log.section("STEP 3a — Loading STRING PPI")

    # Columns vary by STRING version: 'protein1'/'protein2'/'combined_score'
    # or '#string_protein_id_1'/'#string_protein_id_2'/'combined_score'
    ppi_df = pd.read_csv(C.STRING_PPI_FILE, sep="\t")
    log.info(f"  PPI columns detected: {list(ppi_df.columns)}")
    log.info(f"  Raw interactions: {len(ppi_df):,}")

    col_p1    = _detect_col(ppi_df, ["protein1", "#string_protein_id_1"], "PPI", log)
    col_p2    = _detect_col(ppi_df, ["protein2", "#string_protein_id_2"], "PPI", log)
    col_score = _detect_col(ppi_df, ["combined_score"], "PPI", log)

    ppi_df = ppi_df[[col_p1, col_p2, col_score]].copy()
    ppi_df.columns = ["protein1", "protein2", "combined_score"]

    ppi_df = ppi_df[ppi_df["combined_score"] >= C.STRING_MIN_SCORE]
    log.info(f"  After score≥{C.STRING_MIN_SCORE}: {len(ppi_df):,}")

    alias_map: Dict[str, str] = {}

    if C.STRING_INFO_FILE.exists():
        info_df = pd.read_csv(C.STRING_INFO_FILE, sep="\t")
        log.info(f"  Info columns: {list(info_df.columns)}")

        id_col   = _detect_col(info_df,
                               ["#string_protein_id", "string_protein_id",
                                "protein_id", "#protein_id"],
                               "STRING_INFO", log)
        name_col = _detect_col(info_df,
                               ["preferred_name", "gene_name", "name"],
                               "STRING_INFO", log)

        for _, row in info_df.iterrows():
            sid  = str(row[id_col]).strip()
            name = str(row[name_col]).strip()
            if name and name.upper() != "NAN":
                alias_map[sid]              = name
                alias_map[name.upper()]     = name
        log.info(f"  Gene-name mappings from info: {len(alias_map):,}")

    if C.STRING_ALIAS_FILE.exists():

        ali_df = pd.read_csv(C.STRING_ALIAS_FILE, sep="\t",
                             header=0,
                             low_memory=False)
        log.info(f"  Alias columns: {list(ali_df.columns)}")

        cols      = ali_df.columns.tolist()
        ali_id_col   = cols[0]
        ali_name_col = cols[1] if len(cols) > 1 else None

        before = len(alias_map)
        if ali_name_col:
            for _, row in ali_df.iterrows():
                sid   = str(row[ali_id_col]).strip()
                alias = str(row[ali_name_col]).strip()
                if sid not in alias_map and alias and alias.upper() != "NAN":
                    alias_map[sid] = alias
        log.info(f"  Additional from aliases: {len(alias_map)-before:,}")

    log.info(f"  Total alias mappings: {len(alias_map):,}")
    log.checkpoint("string_loaded")
    return ppi_df, alias_map

def load_mirtarbase(log: RunLogger):
    log.section("STEP 3b — Loading miRTarBase")

    try:
        df = pd.read_csv(C.MIRTARBASE_FILE, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(C.MIRTARBASE_FILE, encoding="latin-1", low_memory=False)

    log.info(f"  Raw rows: {len(df):,}")
    log.info(f"  Columns : {list(df.columns)}")

    mir_col     = _detect_col(df, ["miRNA", "mirna", "miRNA_ID"],  "miRTarBase", log)
    gene_col    = _detect_col(df, ["Target Gene", "target_gene", "Target gene", "Gene"], "miRTarBase", log)
    support_col = _detect_col(df, ["Support Type", "support_type", "Support_Type", "Experiments"], "miRTarBase", log)

    df = df[[mir_col, gene_col, support_col]].copy()
    df.columns = ["miRNA", "Target Gene", "Support Type"]

    df = df.dropna(subset=["miRNA", "Target Gene"])
    df = df.drop_duplicates(subset=["miRNA", "Target Gene"])
    log.info(f"  Unique miRNA-target pairs: {len(df):,}")
    log.checkpoint("mirtarbase_loaded")
    return df

def build_node_registry(mrna_df, meth_df, mirna_df, log: RunLogger):
    log.section("STEP 4 — Building Node Registry")

    nodes = {}
    node_id = 0

    for gene in mrna_df.index:
        nodes[f"mRNA::{gene}"] = {
            "node_id": node_id, "name": gene,
            "omic_type": "mRNA", "raw": mrna_df.loc[gene].values.astype(np.float32)
        }
        node_id += 1

    for gene in meth_df.index:
        nodes[f"meth::{gene}"] = {
            "node_id": node_id, "name": gene,
            "omic_type": "methylation", "raw": meth_df.loc[gene].values.astype(np.float32)
        }
        node_id += 1

    for mirna in mirna_df.index:
        nodes[f"miRNA::{mirna}"] = {
            "node_id": node_id, "name": mirna,
            "omic_type": "miRNA", "raw": mirna_df.loc[mirna].values.astype(np.float32)
        }
        node_id += 1

    type_counts = defaultdict(int)
    for v in nodes.values():
        type_counts[v["omic_type"]] += 1

    log.info(f"  Total nodes : {node_id:,}")
    for t, c in type_counts.items():
        log.info(f"    {t:<14}: {c:,}")

    log.checkpoint("node_registry_built")
    return nodes, dict(type_counts)

def normalise_features_safe(
    nodes: dict,
    train_indices: np.ndarray,
    common_samples: List[str],
    log: RunLogger,
) -> Tuple[np.ndarray, dict]:
    log.section("STEP 5 — Leakage-Safe Feature Normalisation")
    log.info("  μ/σ computed on TRAINING patients, applied to ALL patients")

    n_nodes    = len(nodes)
    n_samples  = len(common_samples)
    feat_dim   = 1 + n_samples

    feature_matrix = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    norm_stats: dict = {}

    omic_groups: Dict[str, List[int]] = defaultdict(list)
    for v in nodes.values():
        omic_groups[v["omic_type"]].append(v["node_id"])

    for omic_type, node_ids in omic_groups.items():

        raw = np.stack([nodes[k]["raw"] for k in nodes
                        if nodes[k]["omic_type"] == omic_type and
                           nodes[k]["node_id"] in set(node_ids)],
                       axis=0)

        train_raw = raw[:, train_indices]
        mu, sig   = _zscore_fit(train_raw)

        Z = _zscore_apply(raw, mu, sig)

        norm_stats[omic_type] = {
            "mu":    mu.squeeze().tolist()[:10],
            "sigma": sig.squeeze().tolist()[:10],
        }

        type_id = C.OMIC_TYPE_IDS[omic_type]
        for i, nid in enumerate(node_ids):
            feature_matrix[nid, 0]  = type_id
            feature_matrix[nid, 1:] = Z[i]

        log.info(f"  {omic_type:<14}: {len(node_ids):,} nodes normalised"
                 f"  (train μ range [{mu.min():.3f}, {mu.max():.3f}])")

    log.checkpoint("features_normalised")
    return feature_matrix, norm_stats

def build_edges(nodes: dict, ppi_df, alias_map, mirtarbase_df, log: RunLogger):
    log.section("STEP 6 — Edge Construction")

    mrna_lookup  = {v["name"]: v["node_id"] for v in nodes.values() if v["omic_type"] == "mRNA"}
    meth_lookup  = {v["name"]: v["node_id"] for v in nodes.values() if v["omic_type"] == "methylation"}
    mirna_lookup = {v["name"]: v["node_id"] for v in nodes.values() if v["omic_type"] == "miRNA"}

    all_edges: List[Tuple[int, int, str, float]] = []

    ppi_count = 0
    for _, row in ppi_df.iterrows():
        g1 = alias_map.get(row["protein1"], row["protein1"])
        g2 = alias_map.get(row["protein2"], row["protein2"])
        if g1 in mrna_lookup and g2 in mrna_lookup:
            w = row["combined_score"] / 1000.0
            nid1, nid2 = mrna_lookup[g1], mrna_lookup[g2]
            all_edges.append((nid1, nid2, "PPI", w))
            all_edges.append((nid2, nid1, "PPI", w))
            ppi_count += 2
    log.info(f"  PPI edges         : {ppi_count:,}  (bidirectional)")

    support_weights = {
        "Functional MTI":        1.00,
        "Functional MTI (Weak)": 0.75,
    }
    mirna_count = 0
    for _, row in mirtarbase_df.iterrows():
        mir  = str(row["miRNA"]).strip()
        gene = str(row["Target Gene"]).strip()
        if mir in mirna_lookup and gene in mrna_lookup:
            w = support_weights.get(str(row.get("Support Type", "")), 0.50)
            nid_mir  = mirna_lookup[mir]
            nid_gene = mrna_lookup[gene]
            all_edges.append((nid_mir, nid_gene, "miRNA_target", w))
            mirna_count += 1
    log.info(f"  miRNA-target edges: {mirna_count:,}")

    meth_count = 0
    common_genes = set(meth_lookup.keys()) & set(mrna_lookup.keys())
    for gene in common_genes:
        nid_meth = meth_lookup[gene]
        nid_mrna = mrna_lookup[gene]
        all_edges.append((nid_meth, nid_mrna, "methylation_gene", 1.0))
        all_edges.append((nid_mrna, nid_meth, "methylation_gene", 1.0))
        meth_count += 2
    log.info(f"  Meth-gene edges   : {meth_count:,}  (bidirectional)")
    log.info(f"  TOTAL edges       : {len(all_edges):,}")

    log.checkpoint("edges_built")
    return all_edges

def export_graph(nodes: dict, type_counts: dict,
                 feature_matrix: np.ndarray,
                 all_edges: List[Tuple],
                 labels_df: pd.DataFrame,
                 common_samples: List[str],
                 norm_stats: dict,
                 all_splits: dict,
                 log: RunLogger):
    log.section("STEP 7 — Exporting Graph Artefacts")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    node_rows = sorted(nodes.values(), key=lambda x: x["node_id"])
    node_df = pd.DataFrame([
        {"node_id": v["node_id"], "name": v["name"], "omic_type": v["omic_type"]}
        for v in node_rows
    ])
    node_df.to_csv(C.NODE_METADATA_FILE, index=False)
    np.save(C.NODE_FEATURES_FILE, feature_matrix)
    log.info(f"  node_features.npy  : {feature_matrix.shape}")
    log.info(f"  node_metadata.csv  : {len(node_df):,} rows")

    edge_type_map = {"PPI": 0, "miRNA_target": 1, "methylation_gene": 2}
    src_arr   = np.array([e[0] for e in all_edges], dtype=np.int64)
    dst_arr   = np.array([e[1] for e in all_edges], dtype=np.int64)
    wt_arr    = np.array([e[3] for e in all_edges], dtype=np.float32)
    et_arr    = np.array([edge_type_map.get(e[2], 0) for e in all_edges], dtype=np.int32)

    edge_index = np.stack([src_arr, dst_arr], axis=0)
    np.save(C.RESULTS_DIR / "edge_index.npy",    edge_index)
    np.save(C.RESULTS_DIR / "edge_weights.npy",  wt_arr)
    np.save(C.RESULTS_DIR / "edge_type_ids.npy", et_arr)

    edge_df = pd.DataFrame({
        "src": src_arr, "dst": dst_arr,
        "edge_type": [e[2] for e in all_edges],
        "weight": wt_arr
    })
    edge_df.to_csv(C.EDGES_FILE, index=False)

    with open(C.RESULTS_DIR / "edge_type_mapping.json", "w") as f:
        json.dump(edge_type_map, f, indent=2)
    log.info(f"  edges.csv          : {len(edge_df):,} rows")

    labels_df["PAM50_code"] = labels_df["PAM50"].map(C.CLASS_TO_INT)
    labels_df[["PAM50", "PAM50_code"]].to_csv(C.SAMPLE_LABELS_FILE)
    log.info(f"  sample_labels.csv  : {len(labels_df)} rows")

    with open(C.COMMON_SAMPLES_FILE, "w") as f:
        json.dump(common_samples, f, indent=2)

    degree_count = defaultdict(int)
    for e in all_edges:
        degree_count[e[0]] += 1
    degrees = list(degree_count.values())

    pam50_dist = labels_df["PAM50"].value_counts().to_dict()
    stats = {
        "dataset":            "BRCA",
        "total_nodes":        len(nodes),
        "node_counts":        type_counts,
        "total_edges":        len(all_edges),
        "edge_counts":        {t: int((et_arr == i).sum())
                               for t, i in edge_type_map.items()},
        "feature_dim":        int(feature_matrix.shape[1]),
        "num_samples":        len(common_samples),
        "pam50_distribution": pam50_dist,
        "num_classes":        len(pam50_dist),
        "avg_degree":         float(np.mean(degrees)) if degrees else 0,
        "max_degree":         int(max(degrees)) if degrees else 0,
        "min_degree":         int(min(degrees)) if degrees else 0,
        "nodes_with_edges":   len(degree_count),
        "isolated_nodes":     len(nodes) - len(degree_count),
        "normalisation": {
            "method":         "zscore",
            "split_on":       "training_patients_only",
            "primary_seed":   C.RANDOM_SEEDS[0],
        },
    }
    with open(C.GRAPH_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  graph_statistics.json saved")

    graph_data = {
        "node_features":   feature_matrix,
        "node_metadata":   node_df,
        "edge_index":      edge_index,
        "edge_weights":    wt_arr,
        "edge_types":      np.array([e[2] for e in all_edges]),
        "common_samples":  common_samples,
        "statistics":      stats,
        "data_splits":     all_splits,
        "norm_stats":      norm_stats,
        "config": {
            "dataset":         "BRCA",
            "omic_type_ids":   C.OMIC_TYPE_IDS,
            "class_names":     C.CLASS_NAMES,
            "class_to_int":    C.CLASS_TO_INT,
            "string_min_score":C.STRING_MIN_SCORE,
        },
    }
    with open(C.GRAPH_DATA_FILE, "wb") as f:
        pickle.dump(graph_data, f)
    log.info(f"  graph_data.pkl saved  ({C.GRAPH_DATA_FILE.stat().st_size/1e6:.1f} MB)")

    log.checkpoint("graph_exported")
    return stats

def main():
    log = RunLogger("stage1_graph_construction", C.LOGS_DIR)
    log.section("HOBIT-BRCA  Stage 1 — Heterogeneous Graph Construction")
    log.info(f"  BASE_DIR    : {C.BASE_DIR}")
    log.info(f"  RESULTS_DIR : {C.RESULTS_DIR}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    mrna_df, meth_df, mirna_df, labels_df, common_samples = load_omics_data(log)

    label_codes = np.array([C.CLASS_TO_INT[p] for p in labels_df["PAM50"]])
    all_splits  = make_patient_splits(label_codes, common_samples, log)
    primary_train_idx = np.array(all_splits[str(C.RANDOM_SEEDS[0])]["train"])

    ppi_df, alias_map = load_string_ppi(log)
    mirtarbase_df     = load_mirtarbase(log)

    nodes, type_counts = build_node_registry(mrna_df, meth_df, mirna_df, log)

    feature_matrix, norm_stats = normalise_features_safe(
        nodes, primary_train_idx, common_samples, log
    )

    all_edges = build_edges(nodes, ppi_df, alias_map, mirtarbase_df, log)

    stats = export_graph(nodes, type_counts, feature_matrix,
                         all_edges, labels_df, common_samples,
                         norm_stats, all_splits, log)

    log.section("STAGE 1 SUMMARY")
    log.log_metric("total_nodes",   stats["total_nodes"])
    log.log_metric("total_edges",   stats["total_edges"])
    log.log_metric("num_samples",   stats["num_samples"])
    log.log_metric("avg_degree",    stats["avg_degree"], fmt=".2f")
    log.log_metric("isolated_nodes",stats["isolated_nodes"])
    for subtype, cnt in stats["pam50_distribution"].items():
        log.log_metric(f"n_{subtype}", cnt, fmt="d")

    log.close()
    return stats

if __name__ == "__main__":
    main()
