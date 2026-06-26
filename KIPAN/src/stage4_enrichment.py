import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

def load_stage2_data(log: RunLogger):
    log.section("Loading Stage 2 biomarker rankings")

    ranking_df = pd.read_pickle(C.BIOMARKER_RANK_FILE)
    log.info(f"  Total biomarkers ranked  : {len(ranking_df):,}")

    with open(C.GRAPH_DATA_FILE, "rb") as f:
        gd = pickle.load(f)

    node_metadata = gd["node_metadata"]
    node_features = gd["node_features"]
    common_samples = gd["common_samples"]

    mrna_mask = node_metadata["omic_type"] == "mRNA"
    mrna_ids  = node_metadata[mrna_mask]["node_id"].values
    mrna_names = node_metadata[mrna_mask]["name"].values
    X_mrna    = node_features[mrna_ids, 1:].T

    log.info(f"  mRNA expression matrix   : {X_mrna.shape}")

    n_high = (ranking_df["importance_label"] == "High").sum()
    log.info(f"  High-importance biomarkers: {n_high:,}")

    labels_df = pd.read_csv(C.SAMPLE_LABELS_FILE, index_col=0)

    return ranking_df, gd, node_metadata, X_mrna, mrna_names, common_samples, labels_df

# STEP 2 — ORA (Fisher's Exact, BH-FDR)

def run_ora(ranking_df: pd.DataFrame, log: RunLogger) -> pd.DataFrame:
    log.section("Over-Representation Analysis (ORA) — Fisher's Exact Test")

    high_genes = set(
        ranking_df[ranking_df["importance_label"] == "High"]["name"].values
    )
    all_genes  = set(ranking_df[ranking_df["omic_type"] == "mRNA"]["name"].values)
    N = len(all_genes)
    k = len(high_genes)

    rows = []
    for gs_name, gs_genes in C.KIPAN_GENE_SETS.items():
        if not isinstance(gs_genes, (set, frozenset)):
            gs_genes = set(gs_genes)
        gs_in_data = gs_genes & all_genes
        M = len(gs_in_data)
        if M == 0:
            continue
        overlap = high_genes & gs_in_data
        x = len(overlap)
        # Fisher's exact (one-sided: over-representation)
        table = [[x, k-x], [M-x, N-k-M+x]]
        try:
            _, pval = scipy_stats.fisher_exact(table, alternative="greater")
        except Exception:
            pval = 1.0
        fold = (x/k) / (M/N) if M/N > 0 else 0
        rows.append({
            "Gene_Set":          gs_name,
            "GS_size_in_data":   M,
            "High_importance_N": k,
            "Overlap":           x,
            "Overlap_genes":     ",".join(sorted(overlap)[:20]),
            "Fold_Enrichment":   round(fold, 3),
            "P_value":           pval,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        log.warning("  No ORA results")
        return df

    _, padj, _, _ = multipletests(df["P_value"].values, method="fdr_bh")
    df["FDR_BH"]  = padj
    df["Sig_FDR5"] = df["FDR_BH"] < 0.05
    df = df.sort_values("P_value")
    df.to_csv(C.RESULTS_DIR / "enrichment_results.csv", index=False)

    n_sig = df["Sig_FDR5"].sum()
    log.info(f"  Gene sets tested      : {len(df)}")
    log.info(f"  Significant (FDR<0.05): {n_sig}")

    for _, row in df[df["Sig_FDR5"]].iterrows():
        log.info(f"    {row['Gene_Set']:<45} "
                 f"FE={row['Fold_Enrichment']:.2f}  "
                 f"p={row['P_value']:.2e}  "
                 f"FDR={row['FDR_BH']:.3f}  "
                 f"overlap={row['Overlap']}")

    log.checkpoint("ora_complete")
    return df

def run_subtype_specificity(ranking_df: pd.DataFrame,
                             X_mrna: np.ndarray,
                             mrna_names: np.ndarray,
                             common_samples: list,
                             labels_df: pd.DataFrame,
                             log: RunLogger) -> pd.DataFrame:
    log.section("Subtype-Specificity Analysis (Kruskal-Wallis, 3 RCC subtypes)")

    y_vec = np.array([
        labels_df.loc[s, C.LABEL_COLUMN] if s in labels_df.index else "NA"
        for s in common_samples
    ])

    masks = {cls: (y_vec == cls) for cls in C.CLASS_NAMES}
    log.info("  Sample counts per subtype: " +
             "  ".join(f"{cls}={masks[cls].sum()}" for cls in C.CLASS_NAMES))

    top_mrna = ranking_df[ranking_df["omic_type"] == "mRNA"].head(200)
    mrna_name2idx = {n: i for i, n in enumerate(mrna_names)}

    rows = []
    for _, row in top_mrna.iterrows():
        gene = row["name"]
        if gene not in mrna_name2idx:
            continue
        idx = mrna_name2idx[gene]
        expr = X_mrna[:, idx]

        groups = [expr[masks[cls]] for cls in C.CLASS_NAMES]
        if any(len(g) < 3 for g in groups):
            continue

        try:
            stat, pval = kruskal(*groups)
        except Exception:
            continue

        N = sum(len(g) for g in groups)
        k = len(groups)
        eta2 = max(0.0, (stat - k + 1) / (N - k))

        means = {cls: float(groups[i].mean()) for i, cls in enumerate(C.CLASS_NAMES)}
        dominant = max(means, key=means.get)

        rows.append({
            "gene":             gene,
            "kruskal_stat":     round(stat, 4),
            "p_value":          pval,
            "eta_squared":      round(eta2, 4),
            "dominant_subtype": dominant,
            "is_hub":           gene in C.HUB_HOUSEKEEPING_GENES,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        log.warning("  No subtype-specificity results")
        return df

    _, padj, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["fdr_bh"]   = padj
    df["sig_fdr5"] = df["fdr_bh"] < 0.05
    df = df.sort_values("p_value")

    n_sig     = df["sig_fdr5"].sum()
    n_hub_sig = (df[df["sig_fdr5"]]["is_hub"]).sum()
    log.info(f"  Genes tested              : {len(df)}")
    log.info(f"  Differentially expressed  : {n_sig}  (FDR<5%)")
    log.info(f"  Sig. hub genes (caution)  : {n_hub_sig}")

    df.to_csv(C.TABLES_DIR / "stage4_subtype_specificity.csv", index=False)
    log.checkpoint("subtype_specificity_done")
    return df

def build_extended_table6(ranking_df: pd.DataFrame,
                           specificity_df: pd.DataFrame,
                           ora_df: pd.DataFrame,
                           log: RunLogger) -> pd.DataFrame:
    log.section("Building Extended Table 6 (Biomarker Analysis)")

    spec_lookup = {}
    if len(specificity_df) > 0:
        spec_lookup = dict(zip(
            specificity_df["gene"],
            specificity_df[["kruskal_stat","p_value","fdr_bh","sig_fdr5",
                             "dominant_subtype","is_hub","eta_squared"]].to_dict("records")
        ))

    top_n   = 20
    high_df = ranking_df[ranking_df["importance_label"] == "High"].head(top_n).copy()
    high_df = high_df.reset_index(drop=True)

    rows = []
    for display_rank, (_, r) in enumerate(high_df.iterrows(), start=1):
        gene      = r["name"]
        omic_type = r["omic_type"]
        score     = r["confidence_penalized_score"]
        ann       = C.KIPAN_BIOMARKER_ANNOTATIONS.get(gene, {})
        sp        = spec_lookup.get(gene, {})

        in_cgc  = gene in C.COSMIC_CGC_GENES
        in_kirc = gene in C.KIRC_SIGNATURE
        in_kirp = gene in C.KIRP_SIGNATURE
        in_kich = gene in C.KICH_SIGNATURE
        is_hub  = gene in C.HUB_HOUSEKEEPING_GENES

        rows.append({
            "Rank":                display_rank,
            "Biomarker":           gene,
            "Omic_Type":           omic_type,
            "Importance_Score":    round(float(score), 4),
            "GMM_Confidence":      round(float(r.get("gmm_confidence", 1.0)), 4),
            "General_Cancer_Role": ann.get("general_role", "—"),
            "RCC_Subtype_Role":    ann.get("subtype_role",  "—"),
            "Key_Reference":       ann.get("reference",     "—"),
            "KW_stat":   round(sp.get("kruskal_stat", float("nan")), 3)
                         if sp else float("nan"),
            "KW_FDR":    round(sp.get("fdr_bh", float("nan")), 4)
                         if sp else float("nan"),
            "Subtype_Specific": "✓" if sp.get("sig_fdr5", False) else "",
            "Dominant_Subtype":  sp.get("dominant_subtype", "—") if sp else "—",
            "Is_Hub_Gene":  "⚠" if is_hub else "",
            "In_CGC":       "✓" if in_cgc else "",
            "In_KIRC_Sig":  "✓" if in_kirc else "",
            "In_KIRP_Sig":  "✓" if in_kirp else "",
            "In_KICH_Sig":  "✓" if in_kich else "",
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(C.TABLES_DIR / "stage4_extended_table6.csv", index=False)

    n_specific = sum(1 for r in rows if r["Subtype_Specific"] == "✓")
    n_hub      = sum(1 for r in rows if r["Is_Hub_Gene"] == "⚠")
    n_cgc      = sum(1 for r in rows if r["In_CGC"] == "✓")
    n_kirc_sig = sum(1 for r in rows if r["In_KIRC_Sig"] == "✓")
    log.info(f"  Extended Table 6 rows: {len(table_df)}")
    log.info(f"  Subtype-specific (KW FDR<5%): {n_specific}")
    log.info(f"  Hub genes flagged: {n_hub}")
    log.info(f"  In COSMIC CGC: {n_cgc}")
    log.info(f"  In KIRC Signature: {n_kirc_sig}")

    log.info(f"\n  Top-10 Biomarkers (Extended Table 6 preview):")
    log.info(f"  {'Rank':<5}{'Gene':<12}{'Omic':<16}{'RCC Role':<28}"
             f"{'Spec':>7}{'Hub':>5}{'CGC':>5}{'KIRC':>6}")
    for _, row in table_df.head(10).iterrows():
        log.info(f"  {row['Rank']:<5}{row['Biomarker']:<12}{row['Omic_Type']:<16}"
                 f"{row['RCC_Subtype_Role']:<28}"
                 f"{row['Subtype_Specific']:>7}"
                 f"{row['Is_Hub_Gene']:>5}"
                 f"{row['In_CGC']:>5}"
                 f"{row['In_KIRC_Sig']:>6}")

    log.checkpoint("table6_built")
    return table_df

def generate_stage4_figures(ranking_df: pd.DataFrame,
                             ora_df: pd.DataFrame,
                             specificity_df: pd.DataFrame,
                             log: RunLogger):
    log.section("Generating Stage 4 Figures")
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(ora_df) > 0 and ora_df["Sig_FDR5"].sum() > 0:
        sig = ora_df[ora_df["Sig_FDR5"]].copy()
        sig = sig.sort_values("Fold_Enrichment")
        max_fe = sig["Fold_Enrichment"].max()

        colors = plt.cm.YlOrRd(
            np.linspace(0.4, 0.9, len(sig)))[::-1]

        fig, ax = plt.subplots(figsize=(14, max(8, len(sig) * 0.55)))
        bars = ax.barh(sig["Gene_Set"], sig["Fold_Enrichment"],
                       color=colors, edgecolor="white")
        ax.axvline(x=1, color="black", linestyle="--", lw=1.2)
        for bar, (_, row) in zip(bars, sig.iterrows()):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"p={row['P_value']:.1e}  n={row['Overlap']}",
                    va="center", fontsize=8)
        ax.set_xlabel("Fold Enrichment")
        ax.set_title("Over-Representation Analysis — Significant Gene Sets (FDR<5%)")
        ax.set_xlim(0, max_fe * 1.35)
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / "stage4_ora_enrichment.png",
                    dpi=300, bbox_inches="tight"); plt.close()
        log.info("  stage4_ora_enrichment.png")

    omic_colors = {"mRNA": "#E53935", "methylation": "#1E88E5", "miRNA": "#43A047"}
    for omic in ["mRNA", "methylation", "miRNA"]:
        sub = ranking_df[ranking_df["omic_type"] == omic].head(25).copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("confidence_penalized_score", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(sub) * 0.38)))
        hub_mask = sub["name"].isin(C.HUB_HOUSEKEEPING_GENES)
        clrs = [("#FF8A65" if h else omic_colors[omic]) for h in hub_mask]
        ax.barh(sub["name"], sub["confidence_penalized_score"],
                color=clrs, alpha=0.85)
        ax.set_xlabel("Confidence-Penalized Importance Score")
        ax.set_title(f"Top 25 KIPAN {omic} Biomarkers (HOBIT)")
        patches = [
            mpatches.Patch(color=omic_colors[omic], label="Biomarker"),
            mpatches.Patch(color="#FF8A65", label="Hub/Housekeeping"),
        ]
        ax.legend(handles=patches, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / f"stage4_top_{omic}_biomarkers.png",
                    dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage4_top_[omic]_biomarkers.png")

    if len(specificity_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 8))
        df = specificity_df.copy()
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(1e-60))
        sig_fdr = df["sig_fdr5"]
        hub_sig = df["sig_fdr5"] & df["is_hub"]
        not_sig = ~df["sig_fdr5"]

        ax.scatter(df.loc[not_sig, "eta_squared"],
                   df.loc[not_sig, "neg_log10_p"],
                   c="gray", s=15, alpha=0.5, label="Not significant")
        ax.scatter(df.loc[sig_fdr & ~hub_sig, "eta_squared"],
                   df.loc[sig_fdr & ~hub_sig, "neg_log10_p"],
                   c="#E53935", s=20, alpha=0.7, label="Significant (FDR<5%)")
        ax.scatter(df.loc[hub_sig, "eta_squared"],
                   df.loc[hub_sig, "neg_log10_p"],
                   c="#FF8A65", s=30, alpha=0.9, label="Hub/Housekeeping gene")

        top5 = df[sig_fdr].head(5)
        for _, r in top5.iterrows():
            ax.annotate(r["gene"],
                        (r["eta_squared"], r["neg_log10_p"]),
                        fontsize=8, ha="left",
                        xytext=(5, 2), textcoords="offset points")

        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", lw=1)
        ax.set_xlabel("η² (Effect Size)"); ax.set_ylabel("-log₁₀(p)")
        ax.set_title("Subtype-Specificity: Kruskal-Wallis Analysis\n"
                     "(KIPAN: KIRC vs KIRP vs KICH)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / "stage4_subtype_specificity.png",
                    dpi=300, bbox_inches="tight"); plt.close()
        log.info("  stage4_subtype_specificity.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    high_df = ranking_df[ranking_df["importance_label"] == "High"]
    omic_counts = high_df["omic_type"].value_counts()
    axes[0].pie(omic_counts.values, labels=omic_counts.index,
                autopct="%1.1f%%",
                colors=["#E53935","#1E88E5","#43A047"])
    axes[0].set_title("High-Importance Biomarkers by Omic Type")

    high_mrna = set(high_df[high_df["omic_type"] == "mRNA"]["name"])
    cat_counts = {
        "CGC Genes":         len(high_mrna & C.COSMIC_CGC_GENES),
        "KIRC Signature":    len(high_mrna & C.KIRC_SIGNATURE),
        "KIRP Signature":    len(high_mrna & C.KIRP_SIGNATURE),
        "KICH Signature":    len(high_mrna & C.KICH_SIGNATURE),
        "Hub/HK Genes":      len(high_mrna & C.HUB_HOUSEKEEPING_GENES),
        "Other":             len(high_mrna - C.COSMIC_CGC_GENES - C.KIRC_SIGNATURE
                                 - C.KIRP_SIGNATURE - C.KICH_SIGNATURE
                                 - C.HUB_HOUSEKEEPING_GENES),
    }
    cats, vals = list(cat_counts.keys()), list(cat_counts.values())
    bar_colors = ["#AB47BC","#EF5350","#42A5F5","#26A69A","#FF7043","#90A4AE"]
    bars = axes[1].bar(cats, vals, color=bar_colors, edgecolor="white")
    axes[1].set_ylabel("Count (mRNA high-importance)")
    axes[1].set_title("Gene Set Membership of High-Importance mRNA Biomarkers")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.5,
                     str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage4_biomarker_composition.png",
                dpi=300, bbox_inches="tight"); plt.close()
    log.info("  stage4_biomarker_composition.png")

    log.checkpoint("stage4_figures_saved")

def main():
    log = RunLogger("stage4_enrichment", C.LOGS_DIR)
    log.section("HOBIT-KIPAN  Stage 4 — Biomarker Enrichment & Validation")

    ranking_df, gd, node_metadata, X_mrna, mrna_names, \
        common_samples, labels_df = load_stage2_data(log)

    ora_df = run_ora(ranking_df, log)

    spec_df = run_subtype_specificity(
        ranking_df, X_mrna, mrna_names, common_samples, labels_df, log
    )

    table6 = build_extended_table6(ranking_df, spec_df, ora_df, log)

    generate_stage4_figures(ranking_df, ora_df, spec_df, log)

    high_df = ranking_df[ranking_df["importance_label"] == "High"].head(20).copy()
    high_df = high_df.reset_index(drop=True)
    high_df.to_csv(C.TABLES_DIR / "biomarker_analysis.csv", index=False)

    n_high = (ranking_df["importance_label"] == "High").sum()
    n_ora_sig = int(ora_df["Sig_FDR5"].sum()) if len(ora_df) > 0 else 0
    n_sub_sig = int(spec_df["sig_fdr5"].sum()) if len(spec_df) > 0 else 0
    cgc_high  = len(set(ranking_df[ranking_df["importance_label"]=="High"]["name"])
                    & C.COSMIC_CGC_GENES)
    cgc_total = len(C.COSMIC_CGC_GENES & set(
        ranking_df[ranking_df["omic_type"]=="mRNA"]["name"]))
    cgc_rate  = cgc_high / max(cgc_total, 1)
    n_hub_high = (ranking_df[ranking_df["importance_label"]=="High"]["name"]
                  .isin(C.HUB_HOUSEKEEPING_GENES)).sum()

    log.log_metric("n_high_importance",       n_high, fmt="d")
    log.log_metric("n_ora_sig_fdr5",          n_ora_sig, fmt="d")
    log.log_metric("n_subtype_specific_fdr5", n_sub_sig, fmt="d")
    log.log_metric("CGC_in_data",             cgc_total, fmt="d")
    log.log_metric("CGC_recovered_high",      cgc_high, fmt="d")
    log.log_metric("CGC_recovery_rate",       cgc_rate)
    log.log_metric("n_hub_genes_high_importance", int(n_hub_high), fmt="d")

    summary = {
        "n_high_importance":       int(n_high),
        "n_ora_significant_fdr5":  n_ora_sig,
        "n_subtype_specific_fdr5": n_sub_sig,
        "CGC_recovery_rate":       round(cgc_rate, 4),
        "n_hub_genes_high":        int(n_hub_high),
    }
    with open(C.RESULTS_DIR / "stage4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.close()
    return summary

if __name__ == "__main__":
    main()
