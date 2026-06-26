import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

def load_omics_data(log: RunLogger):
    log.section("STEP 1 — Loading KIPAN Omics Data")

    mrna  = pd.read_csv(C.MRNA_FILE,        index_col=0)
    meth  = pd.read_csv(C.METHYLATION_FILE, index_col=0)
    mirna = pd.read_csv(C.MIRNA_FILE,       index_col=0)

    log.info(f"  mRNA         : {mrna.shape}  (genes x samples)")
    log.info(f"  Methylation  : {meth.shape}  (CpG x samples)")
    log.info(f"  miRNA        : {mirna.shape}  (miRNAs x samples)")

    #   1. 'subtype' is only filled for KICH (113 rows). KIRC/KIRP subtypes
    #      must be recovered from 'histological_type':
    #        'kidneyclearcellrenalcarcinoma'     -> KIRC  (537 rows)
    #        'kidneypapillaryrenalcellcarcinoma' -> KIRP  (291 rows)
    #        'kidneychromophobe'                 -> KICH  (113 rows)
    #   2. The labels file uses 'sample_id' as a data column (not the index).

    HIST_MAP = {
        "kidneyclearcellrenalcarcinoma":     "KIRC",
        "kidneyclearcelrenalcarcinoma":      "KIRC",
        "kidneyclearcell":                   "KIRC",
        "kidney clear cell":                 "KIRC",
        "kidneypapillaryrenalcellcarcinoma": "KIRP",
        "kidneypapillaryrenalcarcinoma":     "KIRP",
        "kidneypapillary":                   "KIRP",
        "kidney papillary":                  "KIRP",
        "kidneychromophobe":                 "KICH",
        "kidney chromophobe":                "KICH",
    }

    labels_raw = pd.read_csv(C.LABELS_FILE)
    log.info(f"  Labels raw   : {labels_raw.shape}  cols={list(labels_raw.columns)}")

    if C.LABEL_COLUMN in labels_raw.columns and "histological_type" in labels_raw.columns:
        bad = ~labels_raw[C.LABEL_COLUMN].isin(C.CLASS_TO_INT.keys())
        if bad.any():
            hist_vals = (labels_raw.loc[bad, "histological_type"]
                         .astype(str).str.lower().str.strip())
            filled = hist_vals.map(HIST_MAP)
            labels_raw.loc[bad & filled.notna(), C.LABEL_COLUMN] = filled[filled.notna()]
            n_filled = int(filled.notna().sum())
            log.info(f"  Filled {n_filled} subtypes from histological_type")

    log.info(f"  Subtype distribution after fill: "
             f"{labels_raw[C.LABEL_COLUMN].value_counts(dropna=False).to_dict()}")

    if "sample_id" in labels_raw.columns:
        labels_raw = labels_raw.set_index("sample_id")

    labels = labels_raw

    if C.LABEL_COLUMN not in labels.columns:
        raise ValueError(f"Column '{C.LABEL_COLUMN}' not found. "
                         f"Available: {list(labels.columns)}")

    common = sorted(set(mrna.columns) & set(meth.columns) &
                    set(mirna.columns) & set(labels.index))
    log.info(f"  Common samples (before label filter): {len(common)}")

    if len(common) == 0:
        raise ValueError(
            "Zero common samples found between omics files and labels. "
            f"Label index sample: {list(labels.index[:3])}. "
            f"Omics column sample: {list(mrna.columns[:3])}."
        )

    mrna   = mrna[common]
    meth   = meth[common]
    mirna  = mirna[common]
    labels = labels.loc[common]

    if labels.index.duplicated().any():
        n_dup = int(labels.index.duplicated().sum())
        log.warning(f"  Removing {n_dup} duplicate sample IDs (keeping first)")
        labels = labels[~labels.index.duplicated(keep="first")]
        common = sorted(labels.index.tolist())
        mrna = mrna[common]; meth = meth[common]; mirna = mirna[common]

    valid_mask = labels[C.LABEL_COLUMN].isin(C.CLASS_TO_INT.keys())
    n_dropped  = int((~valid_mask).sum())
    if n_dropped > 0:
        bad_vals = labels.loc[~valid_mask, C.LABEL_COLUMN].unique().tolist()
        log.warning(f"  Dropping {n_dropped} samples with unrecognised labels: {bad_vals}")
        labels = labels.loc[valid_mask]
        common = sorted(labels.index.tolist())
        mrna = mrna[common]; meth = meth[common]; mirna = mirna[common]

    log.info(f"  Common samples (after label filter) : {len(common)}")

    dist = labels[C.LABEL_COLUMN].value_counts().to_dict()
    for cls in C.CLASS_NAMES:
        n = dist.get(cls, 0)
        log.info(f"    {cls:<6}: {n:>4}  ({100*n/len(common):.1f}%)")

    log.checkpoint("data_loaded")
    return mrna, meth, mirna, labels, common

def create_patient_splits(labels_df: pd.DataFrame,
                          common_samples: list,
                          log: RunLogger) -> dict:
    log.section("STEP 2 — Patient-Level Stratified Split (pre-normalisation)")
    log.info(f"  Strategy : {int(C.TRAIN_RATIO*100)}/{int(C.VAL_RATIO*100)}/{int(C.TEST_RATIO*100)} "
             f"stratified, {C.N_SEEDS} seeds")

    valid_samples = [s for s in common_samples
                     if labels_df.loc[s, C.LABEL_COLUMN] in C.CLASS_TO_INT]
    if len(valid_samples) < len(common_samples):
        log.warning(f"  Skipping {len(common_samples)-len(valid_samples)} "
                    f"samples with unrecognised labels in patient split")
        common_samples = valid_samples

    sample_arr = np.array(common_samples)
    y = np.array([C.CLASS_TO_INT[labels_df.loc[s, C.LABEL_COLUMN]]
                  for s in common_samples])

    splits = {}
    for seed in C.RANDOM_SEEDS:

        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=(C.VAL_RATIO + C.TEST_RATIO),
            random_state=seed
        )
        tr_idx, temp_idx = next(sss1.split(sample_arr, y))

        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=C.TEST_RATIO / (C.VAL_RATIO + C.TEST_RATIO),
            random_state=seed
        )
        val_local, te_local = next(sss2.split(temp_idx, y[temp_idx]))

        splits[seed] = {
            "train": sample_arr[tr_idx].tolist(),
            "val":   sample_arr[temp_idx[val_local]].tolist(),
            "test":  sample_arr[temp_idx[te_local]].tolist(),
        }
        log.info(f"  seed={seed}  train={len(splits[seed]['train'])}  "
                 f"val={len(splits[seed]['val'])}  "
                 f"test={len(splits[seed]['test'])}")

    with open(C.DATA_SPLITS_FILE, "w") as f:
        json.dump({"splits": {str(k): v for k, v in splits.items()}}, f, indent=2)
    log.info(f"  Splits saved → {C.DATA_SPLITS_FILE}")
    log.checkpoint("splits_created")

    return splits, splits[C.RANDOM_SEEDS[0]]["train"]

def load_string_ppi(log: RunLogger):
    log.section("STEP 3a — Loading STRING PPI")

    log.info(f"  PPI file: {C.STRING_PPI_FILE}")
    ppi_df = pd.read_csv(C.STRING_PPI_FILE, sep="\t")

    cols = list(ppi_df.columns)
    if "protein1" in cols and "protein2" in cols:
        col_a, col_b = "protein1", "protein2"
    else:
        col_a, col_b = cols[0], cols[1]
    score_col = "combined_score" if "combined_score" in cols else cols[2]
    log.info(f"  PPI columns detected: {[col_a, col_b, score_col]}")
    log.info(f"  Raw interactions: {len(ppi_df):,}")

    ppi_df = ppi_df[ppi_df[score_col] >= C.STRING_MIN_SCORE]
    log.info(f"  After score≥{C.STRING_MIN_SCORE}: {len(ppi_df):,}")

    alias_map = {}

    info_path = C.STRING_INFO_FILE
    log.info(f"  Info columns: loading from {info_path.name}")
    info_df = pd.read_csv(info_path, sep="\t")
    info_cols = list(info_df.columns)
    log.info(f"  Info columns: {info_cols}")
    id_col   = [c for c in info_cols if "string_protein_id" in c or "protein_id" in c.lower()][0]
    name_col = [c for c in info_cols if "preferred_name" in c or "gene_name" in c.lower()][0]
    for _, row in info_df.iterrows():
        alias_map[str(row[id_col])] = str(row[name_col])
    log.info(f"  Gene-name mappings from info: {len(alias_map):,}")

    aliases_path = C.STRING_ALIASES_FILE
    log.info(f"  Alias columns: loading from {aliases_path.name}")
    alias_df = pd.read_csv(aliases_path, sep="\t", low_memory=False)
    a_cols = list(alias_df.columns)
    log.info(f"  Alias columns: {a_cols}")
    aid_col   = a_cols[0]
    aalias_col = a_cols[1]
    before = len(alias_map)
    for _, row in alias_df.iterrows():
        sid = str(row[aid_col])
        if sid not in alias_map:
            alias_map[sid] = str(row[aalias_col])
    log.info(f"  Additional from aliases: {len(alias_map)-before:,}")
    log.info(f"  Total alias mappings: {len(alias_map):,}")

    log.checkpoint("string_loaded")
    return ppi_df, alias_map, col_a, col_b

def load_mirtarbase(log: RunLogger) -> pd.DataFrame:
    log.section("STEP 3b — Loading miRTarBase (hsa_MTI.csv)")

    log.info(f"  File: {C.MIRTARBASE_FILE}")

    df = pd.read_csv(C.MIRTARBASE_FILE, low_memory=False)
    log.info(f"  Raw rows: {len(df):,}")
    log.info(f"  Columns : {list(df.columns)}")

    cols = list(df.columns)
    mirna_col  = [c for c in cols if "mirna"   in c.lower() and "species" not in c.lower()][0]
    target_col = [c for c in cols if "target"  in c.lower() and "gene"    in c.lower()  and
                                      "entrez" not in c.lower()][0]

    df = df[[mirna_col, target_col]].dropna()
    df.columns = ["miRNA", "Target_Gene"]
    df["miRNA"]       = df["miRNA"].str.strip()
    df["Target_Gene"] = df["Target_Gene"].str.strip()
    df = df.drop_duplicates()
    log.info(f"  Unique miRNA-target pairs: {len(df):,}")

    log.checkpoint("mirtarbase_loaded")
    return df

def create_node_registry(mrna_df: pd.DataFrame,
                         meth_df: pd.DataFrame,
                         mirna_df: pd.DataFrame,
                         log: RunLogger) -> pd.DataFrame:
    log.section("STEP 4 — Building Node Registry")

    rows = []
    node_id = 0

    for gene in mrna_df.index:
        rows.append({"node_id": node_id, "name": gene, "omic_type": "mRNA"})
        node_id += 1

    for cpg in meth_df.index:
        rows.append({"node_id": node_id, "name": cpg, "omic_type": "methylation"})
        node_id += 1

    for mi in mirna_df.index:
        rows.append({"node_id": node_id, "name": mi, "omic_type": "miRNA"})
        node_id += 1

    node_df = pd.DataFrame(rows)
    log.info(f"  Total nodes : {len(node_df):,}")
    log.info(f"    mRNA          : {(node_df['omic_type']=='mRNA').sum():,}")
    log.info(f"    methylation   : {(node_df['omic_type']=='methylation').sum():,}")
    log.info(f"    miRNA         : {(node_df['omic_type']=='miRNA').sum():,}")
    log.checkpoint("node_registry_built")
    return node_df

def normalise_features(mrna_df: pd.DataFrame,
                        meth_df: pd.DataFrame,
                        mirna_df: pd.DataFrame,
                        train_samples: list,
                        common_samples: list,
                        log: RunLogger):
    log.section("STEP 5 — Leakage-Safe Feature Normalisation")
    log.info("  μ/σ computed on TRAINING patients, applied to ALL patients")

    def zscore(df, tr_samps):
        tr = df[tr_samps]
        mu    = tr.mean(axis=1)
        sigma = tr.std(axis=1).replace(0, 1)
        return df.subtract(mu, axis=0).divide(sigma, axis=0)

    mrna_z  = zscore(mrna_df,  train_samples)
    meth_z  = zscore(meth_df,  train_samples)
    mirna_z = zscore(mirna_df, train_samples)

    log.info(f"  mRNA          : {len(mrna_z)} nodes normalised"
             f"  (train μ range [{mrna_df[train_samples].mean(axis=1).min():.3f},"
             f" {mrna_df[train_samples].mean(axis=1).max():.3f}])")
    log.info(f"  methylation   : {len(meth_z)} nodes normalised"
             f"  (train μ range [{meth_df[train_samples].mean(axis=1).min():.3f},"
             f" {meth_df[train_samples].mean(axis=1).max():.3f}])")
    log.info(f"  miRNA         : {len(mirna_z)} nodes normalised"
             f"  (train μ range [{mirna_df[train_samples].mean(axis=1).min():.3f},"
             f" {mirna_df[train_samples].mean(axis=1).max():.3f}])")

    log.checkpoint("features_normalised")
    return mrna_z, meth_z, mirna_z

def build_edges(node_df: pd.DataFrame,
                ppi_df: pd.DataFrame,
                alias_map: dict,
                col_a: str, col_b: str,
                mirtarbase_df: pd.DataFrame,
                log: RunLogger):
    log.section("STEP 6 — Edge Construction")

    name2id = dict(zip(node_df["name"], node_df["node_id"]))
    mrna_names  = set(node_df[node_df["omic_type"] == "mRNA"]["name"])
    meth_names  = set(node_df[node_df["omic_type"] == "methylation"]["name"])
    mirna_names = set(node_df[node_df["omic_type"] == "miRNA"]["name"])

    prefix = f"{C.STRING_SPECIES}."
    def map_id(string_id):
        key = str(string_id)
        if key in alias_map:
            return alias_map[key]
        stripped = key.replace(prefix, "")
        return alias_map.get(stripped, None)

    src_list, dst_list, wt_list, tp_list = [], [], [], []

    ppi_added = 0
    for _, row in ppi_df.iterrows():
        ga = map_id(row[col_a])
        gb = map_id(row[col_b])
        if ga in name2id and gb in name2id:
            w = float(row["combined_score"]) / 1000.0
            src_list += [name2id[ga], name2id[gb]]
            dst_list += [name2id[gb], name2id[ga]]
            wt_list  += [w, w]
            tp_list  += [0, 0]
            ppi_added += 2
    log.info(f"  PPI edges         : {ppi_added:,}  (bidirectional)")

    mir_added = 0
    for _, row in mirtarbase_df.iterrows():
        mi  = row["miRNA"]
        tgt = row["Target_Gene"]
        if mi in name2id and tgt in name2id:
            src_list.append(name2id[mi])
            dst_list.append(name2id[tgt])
            wt_list.append(0.9)
            tp_list.append(1)
            mir_added += 1
    log.info(f"  miRNA-target edges: {mir_added:,}")

    name_omic_to_id = {
        (row["name"], row["omic_type"]): row["node_id"]
        for _, row in node_df.iterrows()
    }
    meth_added = 0
    for m_name in meth_names:
        if m_name in mrna_names:
            m_id = name_omic_to_id.get((m_name, "methylation"))
            g_id = name_omic_to_id.get((m_name, "mRNA"))
            if m_id is not None and g_id is not None:
                src_list += [m_id, g_id]
                dst_list += [g_id, m_id]
                wt_list  += [0.8, 0.8]
                tp_list  += [2, 2]
                meth_added += 2
    log.info(f"  Meth-gene edges   : {meth_added:,}  (bidirectional)")

    total = len(src_list)
    log.info(f"  TOTAL edges       : {total:,}")
    log.checkpoint("edges_built")

    src_arr = np.array(src_list, dtype=np.int64)
    dst_arr = np.array(dst_list, dtype=np.int64)
    wt_arr  = np.array(wt_list,  dtype=np.float32)
    tp_arr  = np.array(tp_list,  dtype=np.int8)

    edge_counts = {
        "ppi_bidirectional": ppi_added,
        "mirna_target":      mir_added,
        "meth_gene_bidirectional": meth_added,
    }
    return src_arr, dst_arr, wt_arr, tp_arr, edge_counts

def export_graph(node_df: pd.DataFrame,
                 mrna_z: pd.DataFrame,
                 meth_z: pd.DataFrame,
                 mirna_z: pd.DataFrame,
                 src_arr, dst_arr, wt_arr, tp_arr,
                 labels_df: pd.DataFrame,
                 common_samples: list,
                 edge_counts: dict,
                 log: RunLogger):
    log.section("STEP 7 — Exporting Graph Artefacts")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_nodes   = len(node_df)
    n_samples = len(common_samples)
    feat      = np.zeros((n_nodes, 1 + n_samples), dtype=np.float32)

    for _, row in node_df.iterrows():
        feat[row["node_id"], 0] = C.OMIC_TYPE_IDS[row["omic_type"]]

    def fill_omic(df, omic_name):
        sub = node_df[node_df["omic_type"] == omic_name]
        for _, row in sub.iterrows():
            nid  = row["node_id"]
            name = row["name"]
            if name in df.index:
                vals = df.loc[name, common_samples].values.astype(np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                feat[nid, 1:] = vals

    fill_omic(mrna_z,  "mRNA")
    fill_omic(meth_z,  "methylation")
    fill_omic(mirna_z, "miRNA")

    np.save(C.RESULTS_DIR / "node_features.npy",   feat)
    node_df.to_csv(C.RESULTS_DIR / "node_metadata.csv", index=False)

    edge_index = np.stack([src_arr, dst_arr], axis=0)
    np.save(C.RESULTS_DIR / "edge_index.npy",      edge_index)
    np.save(C.RESULTS_DIR / "edge_weights.npy",    wt_arr)
    np.save(C.RESULTS_DIR / "edge_type_ids.npy",   tp_arr)

    edges_df = pd.DataFrame({
        "src": src_arr, "dst": dst_arr,
        "weight": wt_arr, "type": tp_arr
    })
    edges_df.to_csv(C.RESULTS_DIR / "edges.csv", index=False)

    labels_df.to_csv(C.SAMPLE_LABELS_FILE)

    log.info(f"  node_features.npy  : {feat.shape}")
    log.info(f"  node_metadata.csv  : {len(node_df):,} rows")
    log.info(f"  edges.csv          : {len(edges_df):,} rows")
    log.info(f"  sample_labels.csv  : {len(labels_df):,} rows")

    degrees = np.bincount(src_arr, minlength=n_nodes)
    stats = {
        "total_nodes":    n_nodes,
        "total_edges":    len(src_arr),
        "num_samples":    n_samples,
        "avg_degree":     float(degrees.mean()),
        "isolated_nodes": int((degrees == 0).sum()),
        "edge_counts":    {k: int(v) for k, v in edge_counts.items()},
        "node_counts": {
            "mRNA":        int((node_df["omic_type"] == "mRNA").sum()),
            "methylation": int((node_df["omic_type"] == "methylation").sum()),
            "miRNA":       int((node_df["omic_type"] == "miRNA").sum()),
        },
        "subtype_distribution": labels_df[C.LABEL_COLUMN].value_counts().to_dict(),
    }
    with open(C.RESULTS_DIR / "graph_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  graph_statistics.json saved")

    graph_data = {
        "node_features":  feat,
        "node_metadata":  node_df,
        "edge_index":     edge_index,
        "edge_weights":   wt_arr,
        "edge_types":     tp_arr,
        "common_samples": common_samples,
        "statistics":     stats,
    }
    with open(C.GRAPH_DATA_FILE, "wb") as f:
        pickle.dump(graph_data, f)
    sz = C.GRAPH_DATA_FILE.stat().st_size / 1e6
    log.info(f"  graph_data.pkl saved  ({sz:.1f} MB)")
    log.checkpoint("graph_exported")

    return stats

def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log = RunLogger("stage1_graph_construction", C.LOGS_DIR)
    log.section("HOBIT-KIPAN  Stage 1 — Heterogeneous Graph Construction")
    log.info(f"  BASE_DIR    : {C.BASE_DIR}")
    log.info(f"  RESULTS_DIR : {C.RESULTS_DIR}")

    mrna, meth, mirna, labels, common = load_omics_data(log)

    splits, train_samples = create_patient_splits(labels, common, log)

    ppi_df, alias_map, col_a, col_b = load_string_ppi(log)

    mirtarbase_df = load_mirtarbase(log)

    node_df = create_node_registry(mrna, meth, mirna, log)

    mrna_z, meth_z, mirna_z = normalise_features(
        mrna, meth, mirna, train_samples, common, log
    )

    src_arr, dst_arr, wt_arr, tp_arr, edge_counts = build_edges(
        node_df, ppi_df, alias_map, col_a, col_b, mirtarbase_df, log
    )

    stats = export_graph(
        node_df, mrna_z, meth_z, mirna_z,
        src_arr, dst_arr, wt_arr, tp_arr,
        labels, common, edge_counts, log
    )

    log.section("STAGE 1 SUMMARY")
    log.log_metric("total_nodes",  stats["total_nodes"], fmt="d")
    log.log_metric("total_edges",  stats["total_edges"], fmt="d")
    log.log_metric("num_samples",  stats["num_samples"], fmt="d")
    log.log_metric("avg_degree",   stats["avg_degree"])
    log.log_metric("isolated_nodes", stats["isolated_nodes"], fmt="d")
    for cls in C.CLASS_NAMES:
        n = stats["subtype_distribution"].get(cls, 0)
        log.log_metric(f"n_{cls}", n, fmt="d")

    log.close()
    return stats

if __name__ == "__main__":
    main()
