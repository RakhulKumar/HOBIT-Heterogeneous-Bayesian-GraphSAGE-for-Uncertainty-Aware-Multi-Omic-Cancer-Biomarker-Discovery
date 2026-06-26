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
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from logger import RunLogger

HALLMARK_GENE_SETS = {
    "HALLMARK_E2F_TARGETS": {
        "E2F1","E2F2","E2F3","PCNA","RB1","CCNA2","CCNE1","CDC6","MCM2",
        "MCM3","MCM4","MCM5","MCM6","MCM7","CDK1","CCNB1","MKI67","TOP2A",
        "TYMS","DHFR","RRM1","RRM2","BIRC5","CENPF","CDKN1A","CDKN2A",
    },
    "HALLMARK_G2M_CHECKPOINT": {
        "CCNB1","CCNB2","CDK1","PLK1","AURKA","AURKB","BUB1","BUB1B",
        "CDC20","CENPF","KIF2C","KNTC2","MKI67","ESPL1","PTTG1","TTK",
        "ZWINT","TOP2A","SMC2","SMC4",
    },
    "HALLMARK_MYC_TARGETS_V1": {
        "MYC","NPM1","CDK4","EIF4E","PCNA","TP53","BRCA1","MDM2","RB1",
        "CCND1","CCND2","CCNE1","CDK2","CUL1","FBXW7","CDKN1B","CDKN1A",
        "EEF2","RPS6","RPL11","RPL13","RPL23","RPL26","RPL35A",
    },
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING": {
        "PIK3CA","PIK3CB","PIK3CD","PIK3R1","AKT1","AKT2","AKT3","PTEN",
        "MTOR","RPS6KB1","EIF4EBP1","FOXO1","FOXO3","GSK3A","GSK3B",
        "MDM2","CCND1","BCL2","BAD","TSC1","TSC2","RHEB","EGFR","ERBB2",
        "IGF1R","PDPK1","SGK1","INSR",
    },
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": {
        "VIM","CDH1","CDH2","FN1","SNAI1","SNAI2","TWIST1","ZEB1","ZEB2",
        "MMP2","MMP9","MMP14","ITGB1","ITGB3","ITGAV","FAK","SRC",
        "TGM2","CTNNB1","WNT5A","EGFR","TGFB1","TGFB2",
    },
    "HALLMARK_APOPTOSIS": {
        "TP53","BCL2","BAX","BAK1","MCL1","BCL2L1","PUMA","NOXA","CASP3",
        "CASP8","CASP9","CASP7","FADD","FAS","FASL","TRAIL","TNFRSF10A",
        "TNFRSF10B","CYCS","APAF1","BID","DIABLO","XIAP",
    },
    "HALLMARK_HYPOXIA": {
        "HIF1A","HIF2A","VEGFA","VEGFB","LDHA","LDHB","PGK1","ENO1",
        "GAPDH","ALDOA","PFKL","PKM","GLUT1","SLC2A1","CA9","BNIP3",
        "BNIP3L","EPO","EPAS1","PDK1",
    },
}

BRCA_SUBTYPE_SIGNATURES = {
    "Basal": {
        "KRT5","KRT14","KRT17","TP53","BRCA1","BRCA2","FOXC1","MIA",
        "SFRP1","CDKN2A","CDH3","LAMC2","SLC52A3","EGFR","MYBL2",
    },
    "Her2": {
        "ERBB2","GRB7","STARD3","PSMD3","CSN1S1","ESAD1","ORMDL3",
        "MIEN1","PNMT","GSDMB","ZPBP2","IKBKE","MED24",
    },
    "LumA": {
        "ESR1","PGR","BCL2","GATA3","FOXA1","TFF1","TFF3","MYB",
        "NAT1","SLC39A6","MLPH","MAPT","CXXC5","DCN","BAG1",
    },
    "LumB": {
        "CCNB1","MKI67","UBE2C","MELK","CENPF","NDC80","EXO1","NUF2",
        "CDC20","ORC6L","CDCA1","CEP55","ANLN","TYMS","RRM2","LMNB2",
    },
}

HUB_HOUSEKEEPING = {
    "ACTB","GAPDH","B2M","HPRT1","SDHA","UBC","ACTG1",
    "RPS27A","RPS4X","RPS18","RPS6","RPL10","RPL39","RPL11",
    "RPS3","RPL5","RPL13A","EEF1A1","EEF2","HSP90AA1","HSPA8",
    "HSPA5","HSPA1A","HSPA1B","VIM","TUBB","TUBA1B",
}

BRCA_BIOMARKER_ANNOTATIONS = {
    "TP53":   {"subtype_role": "Mutated in 80% Basal-like; guardian of genome",
               "subtype": "Basal", "key_ref": "Cancer Genome Atlas Network 2012"},
    "BRCA1":  {"subtype_role": "Germline/somatic mutations drive Basal-like; homologous recombination",
               "subtype": "Basal", "key_ref": "Foulkes et al. 2010"},
    "ERBB2":  {"subtype_role": "Amplified in ~20% HER2-enriched; druggable with trastuzumab",
               "subtype": "Her2",  "key_ref": "Slamon et al. 2001"},
    "ESR1":   {"subtype_role": "Estrogen receptor; defines Luminal A/B; targeted by tamoxifen",
               "subtype": "LumA",  "key_ref": "Harvey et al. 1999"},
    "PGR":    {"subtype_role": "Progesterone receptor; Luminal A biomarker; predicts endocrine response",
               "subtype": "LumA",  "key_ref": "Bardou et al. 2003"},
    "PIK3CA": {"subtype_role": "Most common somatic mutation in Luminal cancers; PI3K/AKT pathway",
               "subtype": "LumA/B","key_ref": "TCGA 2012; Kandoth et al. 2013"},
    "MYC":    {"subtype_role": "Amplified in Basal-like and LumB; drives cell proliferation",
               "subtype": "Basal/LumB","key_ref": "Chandriani et al. 2009"},
    "AKT1":   {"subtype_role": "PI3K/AKT survival signaling; E17K mutation in Luminal",
               "subtype": "LumA",  "key_ref": "Carpten et al. 2007"},
    "EGFR":   {"subtype_role": "Overexpressed in Basal-like triple-negative; EGFR-targeted therapy investigated",
               "subtype": "Basal", "key_ref": "Bhargava et al. 2005"},
    "CCND1":  {"subtype_role": "Cyclin D1; amplified in Luminal B and HER2+; CDK4/6 target",
               "subtype": "LumB/Her2","key_ref": "Buckley et al. 1993"},
    "CTNNB1": {"subtype_role": "β-catenin; Wnt pathway; invasive Luminal phenotype",
               "subtype": "LumA",  "key_ref": "Khramtsov et al. 2010"},
    "CDH1":   {"subtype_role": "E-cadherin; ILC defining marker; lost in lobular subtype",
               "subtype": "LumA",  "key_ref": "Berx et al. 1995"},
    "GATA3":  {"subtype_role": "Luminal A master transcription factor; predicts endocrine sensitivity",
               "subtype": "LumA",  "key_ref": "Sørlie et al. 2001"},
    "FOXA1":  {"subtype_role": "Pioneer factor for ESR1; Luminal A differentiation",
               "subtype": "LumA",  "key_ref": "Carroll et al. 2005"},
    "MKI67":  {"subtype_role": "Proliferation marker; high in Basal and LumB; Ki-67 prognostic index",
               "subtype": "Basal/LumB","key_ref": "Yerushalmi et al. 2010"},
    "BIRC5":  {"subtype_role": "Survivin; anti-apoptotic; overexpressed in HER2/Basal; poor prognosis",
               "subtype": "Basal/Her2","key_ref": "Tanaka et al. 2000"},
    "RRM2":   {"subtype_role": "Ribonucleotide reductase; elevated in Basal-like; replication stress",
               "subtype": "Basal", "key_ref": "Liu et al. 2012"},
    "KRT5":   {"subtype_role": "Basal cytokeratin; canonical Basal-like marker",
               "subtype": "Basal", "key_ref": "Nielsen et al. 2004"},
    "KRT17":  {"subtype_role": "Basal cytokeratin; Basal-like immunohistochemical marker",
               "subtype": "Basal", "key_ref": "Bhargava et al. 2005"},
    "SRC":    {"subtype_role": "Non-receptor tyrosine kinase; invasion; Basal-like signaling hub",
               "subtype": "Basal", "key_ref": "Zhang et al. 2007"},

    "UBC":    {"subtype_role": "Ubiquitin; network hub gene — connectivity artifact likely",
               "subtype": "N/A (hub)", "key_ref": "Hausser & Jucker 2010"},
    "UBA52":  {"subtype_role": "Ubiquitin-60S ribosomal; network hub gene",
               "subtype": "N/A (hub)", "key_ref": ""},
    "ACTB":   {"subtype_role": "Actin; housekeeping gene — high connectivity, not subtype-specific",
               "subtype": "N/A (hub)", "key_ref": ""},
    "RPS27A": {"subtype_role": "Ribosomal protein; ubiquitous expression — likely network hub artifact",
               "subtype": "N/A (hub)", "key_ref": ""},
    "HSP90AA1":{"subtype_role": "Chaperone; oncogene stabiliser; elevated across multiple subtypes",
                "subtype": "pan-cancer","key_ref": "Neckers 2002"},
}

def load_data(log: RunLogger):
    log.section("Loading Stage 2 biomarker rankings")

    ranking_df = pd.read_pickle(C.BIOMARKER_RANK_FILE)
    labels_df  = pd.read_csv(C.SAMPLE_LABELS_FILE, index_col=0)

    with open(C.GRAPH_DATA_FILE, "rb") as f:
        gd = pickle.load(f)
    node_features  = gd["node_features"]
    node_metadata  = gd["node_metadata"]
    common_samples = gd["common_samples"]

    mrna_mask  = node_metadata["omic_type"] == "mRNA"
    mrna_ids   = node_metadata[mrna_mask]["node_id"].values
    mrna_names = node_metadata[mrna_mask]["name"].values
    X_mrna = node_features[mrna_ids, 1:]

    log.info(f"  Total biomarkers ranked  : {len(ranking_df):,}")
    log.info(f"  mRNA expression matrix   : {X_mrna.shape}")
    log.info(f"  High-importance biomarkers: "
             f"{(ranking_df['importance_label']=='High').sum():,}")

    return ranking_df, labels_df, X_mrna, mrna_names, common_samples

# STEP 2 — ORA (Fisher's Exact)

def run_ora(ranking_df: pd.DataFrame, log: RunLogger) -> pd.DataFrame:
    log.section("Over-Representation Analysis (ORA) — Fisher's Exact Test")

    high_genes = set(
        ranking_df[ranking_df["importance_label"] == "High"]["name"].values
    )
    all_genes   = set(ranking_df[ranking_df["omic_type"] == "mRNA"]["name"].values)
    N   = len(all_genes)
    k   = len(high_genes)

    gene_sets = {
        **HALLMARK_GENE_SETS,
        "COSMIC_CGC":       C.COSMIC_CGC_GENES,
        "PAM50_SIGNATURE":  C.PAM50_GENE_SET,
        "BASAL_SIGNATURE":  BRCA_SUBTYPE_SIGNATURES["Basal"],
        "HER2_SIGNATURE":   BRCA_SUBTYPE_SIGNATURES["Her2"],
        "LUMA_SIGNATURE":   BRCA_SUBTYPE_SIGNATURES["LumA"],
        "LUMB_SIGNATURE":   BRCA_SUBTYPE_SIGNATURES["LumB"],
        "HUB_HOUSEKEEPING": HUB_HOUSEKEEPING,
    }

    rows = []
    for gs_name, gs_genes in gene_sets.items():
        gs_in_data = gs_genes & all_genes
        M = len(gs_in_data)
        if M == 0:
            continue
        overlap = high_genes & gs_in_data
        x = len(overlap)

        table = [[x, k - x], [M - x, N - k - M + x]]
        try:
            _, pval = scipy_stats.fisher_exact(table, alternative="greater")
        except Exception:
            pval = 1.0
        fold = (x / k) / (M / N) if (M / N) > 0 else 0
        rows.append({
            "Gene_Set":            gs_name,
            "GS_size_in_data":     M,
            "High_importance_N":   k,
            "Overlap":             x,
            "Overlap_genes":       ",".join(sorted(overlap)[:20]),
            "Fold_Enrichment":     round(fold, 3),
            "P_value":             pval,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        log.warning("  No ORA results — check gene set membership")
        return df

    _, padj, _, _ = multipletests(df["P_value"].values, method="fdr_bh")
    df["FDR_BH"]  = padj
    df["Sig_FDR5"]= df["FDR_BH"] < 0.05

    df = df.sort_values("P_value")

    log.info(f"  Gene sets tested      : {len(df)}")
    log.info(f"  Significant (FDR<0.05): {df['Sig_FDR5'].sum()}")
    for _, r in df[df["Sig_FDR5"]].iterrows():
        log.info(f"    {r['Gene_Set']:<40} FE={r['Fold_Enrichment']:.2f}"
                 f"  p={r['P_value']:.2e}  FDR={r['FDR_BH']:.3f}"
                 f"  overlap={r['Overlap']}")

    df.to_csv(C.ENRICHMENT_RESULTS_FILE, index=False)
    log.info(f"  Saved → {C.ENRICHMENT_RESULTS_FILE}")

    log.checkpoint("ora_complete")
    return df

def subtype_specificity_analysis(X_mrna: np.ndarray,
                                  mrna_names: np.ndarray,
                                  labels_df: pd.DataFrame,
                                  common_samples: list,
                                  ranking_df: pd.DataFrame,
                                  log: RunLogger) -> pd.DataFrame:
    log.section("Subtype-Specificity Analysis (Kruskal-Wallis)")

    labels = np.array([
        C.CLASS_TO_INT.get(labels_df.loc[s, "PAM50"], -1)
        if s in labels_df.index else -1
        for s in common_samples
    ])
    valid  = labels >= 0
    X_valid = X_mrna[:, valid]
    y_valid = labels[valid]

    top_mrna = ranking_df[ranking_df["omic_type"] == "mRNA"].head(200)["name"].values
    mrna_name_to_idx = {n: i for i, n in enumerate(mrna_names)}

    rows = []
    for gene in top_mrna:
        if gene not in mrna_name_to_idx:
            continue
        idx  = mrna_name_to_idx[gene]
        expr = X_valid[idx]
        groups = [expr[y_valid == c] for c in range(C.NUM_CLASSES)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        try:
            stat, pval = scipy_stats.kruskal(*groups)
        except Exception:
            pval = 1.0; stat = 0.0

        N = sum(len(g) for g in groups)
        k_gr = len(groups)
        eta2 = (stat - k_gr + 1) / (N - k_gr) if N > k_gr else 0

        group_means = {C.CLASS_NAMES[c]: X_valid[idx, y_valid==c].mean()
                       for c in range(C.NUM_CLASSES)
                       if (y_valid==c).sum() > 0}
        dominant_subtype = max(group_means, key=group_means.get)

        rows.append({
            "gene":              gene,
            "kruskal_stat":      round(float(stat), 4),
            "p_value":           float(pval),
            "eta_squared":       round(float(eta2), 4),
            "dominant_subtype":  dominant_subtype,
            "is_hub":            gene in HUB_HOUSEKEEPING,
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
        spec_lookup = dict(zip(specificity_df["gene"],
                               specificity_df[["kruskal_stat","p_value",
                                               "fdr_bh","sig_fdr5",
                                               "dominant_subtype","is_hub",
                                               "eta_squared"]].to_dict("records")))

    top_n   = 20
    high_df = ranking_df[ranking_df["importance_label"] == "High"].head(top_n).copy()

    high_df = high_df.reset_index(drop=True)

    rows = []
    for display_rank, (_, r) in enumerate(high_df.iterrows(), start=1):
        gene      = r["name"]
        omic_type = r["omic_type"]
        score     = r["confidence_penalized_score"]
        ann       = BRCA_BIOMARKER_ANNOTATIONS.get(gene, {})
        sp        = spec_lookup.get(gene, {})

        rows.append({
            "Rank":                display_rank,
            "Biomarker":           gene,
            "Omic_Type":           omic_type,
            "Importance_Score":    round(float(score), 4),
            "GMM_Confidence":      round(float(r.get("gmm_confidence", 0)), 3),
            "General_Cancer_Role": ann.get("subtype_role", "—"),
            "PAM50_Subtype":       ann.get("subtype", "—"),
            "Key_Reference":       ann.get("key_ref", "—"),
            "KW_stat":             round(sp.get("kruskal_stat", float("nan")), 3)
                                   if sp else float("nan"),
            "KW_FDR":              round(sp.get("fdr_bh", float("nan")), 4)
                                   if sp else float("nan"),
            "Subtype_Specific":    bool(sp.get("sig_fdr5", False)) if sp else None,
            "Dominant_Subtype":    sp.get("dominant_subtype", "—") if sp else "—",
            "Is_Hub_Gene":         gene in HUB_HOUSEKEEPING,
            "In_CGC":              gene in C.COSMIC_CGC_GENES,
            "In_PAM50_Sig":        gene in C.PAM50_GENE_SET,
        })

    df = pd.DataFrame(rows)
    df.to_csv(C.BIOMARKER_ANALYSIS_FILE, index=False)
    df.to_csv(C.TABLES_DIR / "stage4_extended_table6.csv", index=False)

    log.info(f"  Extended Table 6 rows: {len(df)}")
    log.info(f"  Subtype-specific (KW FDR<5%): {df['Subtype_Specific'].sum()}")
    log.info(f"  Hub genes flagged: {df['Is_Hub_Gene'].sum()}")
    log.info(f"  In COSMIC CGC: {df['In_CGC'].sum()}")
    log.info(f"  In PAM50 Signature: {df['In_PAM50_Sig'].sum()}")

    log.info("\n  Top-10 Biomarkers (Extended Table 6 preview):")
    log.info(f"  {'Rank':<5}{'Gene':<12}{'Omic':<16}{'PAM50 Role':<30}"
             f"{'Specific':<10}{'Hub':<6}{'CGC':<6}{'PAM50sig'}")
    for _, r in df.head(10).iterrows():
        log.info(f"  {int(r['Rank']):<5}{r['Biomarker']:<12}{r['Omic_Type']:<16}"
                 f"{str(r['PAM50_Subtype'])[:28]:<30}"
                 f"{'✓' if r['Subtype_Specific'] else '':<10}"
                 f"{'⚠' if r['Is_Hub_Gene'] else '':<6}"
                 f"{'✓' if r['In_CGC'] else '':<6}"
                 f"{'✓' if r['In_PAM50_Sig'] else ''}")

    log.checkpoint("table6_built")
    return df

def generate_stage4_figures(ranking_df: pd.DataFrame,
                             ora_df: pd.DataFrame,
                             extended_table6: pd.DataFrame,
                             specificity_df: pd.DataFrame,
                             log: RunLogger):
    log.section("Generating Stage 4 Figures")
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(ora_df) > 0:
        sig_ora = ora_df[ora_df["Sig_FDR5"]].head(15).sort_values("Fold_Enrichment")
        if len(sig_ora) > 0:
            fig, ax = plt.subplots(figsize=(10, max(5, len(sig_ora)*0.45)))
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(sig_ora)))
            ax.barh(sig_ora["Gene_Set"], sig_ora["Fold_Enrichment"],
                    color=colors, edgecolor="white")
            ax.set_xlabel("Fold Enrichment")
            ax.set_title("Over-Representation Analysis — Significant Gene Sets (FDR<5%)")
            ax.axvline(1, color="black", lw=1, ls="--")
            for i, (_, row) in enumerate(sig_ora.iterrows()):
                ax.text(row["Fold_Enrichment"] + 0.02, i,
                        f"p={row['P_value']:.1e}  n={row['Overlap']}",
                        va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(C.FIGURES_DIR / "stage4_ora_enrichment.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
            log.info("  stage4_ora_enrichment.png")

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    omic_colors = {"mRNA": "#E53935", "methylation": "#1E88E5", "miRNA": "#43A047"}

    for ax, omic in zip(axes, ["mRNA", "methylation", "miRNA"]):
        sub = ranking_df[ranking_df["omic_type"] == omic].head(25)
        sub = sub.sort_values("confidence_penalized_score", ascending=True)

        colors = [("#FF8A65" if g in HUB_HOUSEKEEPING else omic_colors[omic])
                  for g in sub["name"]]
        ax.barh(sub["name"], sub["confidence_penalized_score"],
                color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top 25 — {omic}")
        ax.grid(axis="x", alpha=0.3)

        handles = [
            mpatches.Patch(color=omic_colors[omic], label="Biomarker"),
            mpatches.Patch(color="#FF8A65",          label="Hub/Housekeeping"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.suptitle("Top 25 BRCA Biomarkers by Omic Type (HOBIT)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage4_top_biomarkers_by_omic.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage4_top_biomarkers_by_omic.png")

    if len(specificity_df) > 0:
        sp_top = specificity_df.head(200).copy()
        sp_top["-log10_p"] = -np.log10(sp_top["p_value"] + 1e-300)
        sp_top["color"]    = sp_top.apply(
            lambda r: ("#FF8A65" if r["is_hub"]
                       else ("#E53935" if r["sig_fdr5"] else "#BDBDBD")), axis=1
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(sp_top["eta_squared"], sp_top["-log10_p"],
                   c=sp_top["color"], alpha=0.7, s=40)
        ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=1,
                   label="p=0.05 threshold")
        ax.set_xlabel("η² (Effect Size)")
        ax.set_ylabel("−log₁₀(p)")
        ax.set_title("Subtype-Specificity: Kruskal-Wallis Analysis\n(BRCA PAM50 subtypes)")
        handles = [
            mpatches.Patch(color="#E53935", label="Significant (FDR<5%)"),
            mpatches.Patch(color="#FF8A65", label="Hub/Housekeeping gene"),
            mpatches.Patch(color="#BDBDBD", label="Not significant"),
        ]
        ax.legend(handles=handles)

        labeled = sp_top[sp_top["sig_fdr5"] & ~sp_top["is_hub"]].head(5)
        for _, r in labeled.iterrows():
            ax.annotate(r["gene"],
                        xy=(r["eta_squared"], r["-log10_p"]),
                        xytext=(r["eta_squared"]+0.002, r["-log10_p"]+0.2),
                        fontsize=8)
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / "stage4_subtype_specificity.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  stage4_subtype_specificity.png")

    high_df    = ranking_df[ranking_df["importance_label"] == "High"]
    omic_counts= high_df["omic_type"].value_counts()
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(omic_counts.values, labels=omic_counts.index,
                colors=["#E53935","#1E88E5","#43A047"],
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("High-Importance Biomarkers by Omic Type")

    n_high_mrna  = (high_df["omic_type"] == "mRNA").sum()
    n_cgc        = high_df["name"].isin(C.COSMIC_CGC_GENES).sum()
    n_pam50      = high_df["name"].isin(C.PAM50_GENE_SET).sum()
    n_hub        = high_df["name"].isin(HUB_HOUSEKEEPING).sum()
    n_other      = n_high_mrna - n_cgc - n_pam50 - n_hub
    axes[1].bar(["CGC\nGenes", "PAM50\nSignature", "Hub/HK\nGenes", "Other"],
                [n_cgc, n_pam50, n_hub, max(0, n_other)],
                color=["#AB47BC","#26A69A","#FF7043","#78909C"])
    axes[1].set_ylabel("Count (mRNA high-importance)")
    axes[1].set_title("Gene Set Membership of High-Importance mRNA Biomarkers")
    for i, v in enumerate([n_cgc, n_pam50, n_hub, max(0, n_other)]):
        axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "stage4_biomarker_composition.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  stage4_biomarker_composition.png")

    log.checkpoint("stage4_figures_saved")

def main():
    log = RunLogger("stage4_enrichment", C.LOGS_DIR)
    log.section("HOBIT-BRCA  Stage 4 — Biomarker Enrichment & Validation")

    ranking_df, labels_df, X_mrna, mrna_names, common_samples = load_data(log)

    ora_df = run_ora(ranking_df, log)

    specificity_df = subtype_specificity_analysis(
        X_mrna, mrna_names, labels_df, common_samples, ranking_df, log
    )

    extended_table6 = build_extended_table6(ranking_df, specificity_df, ora_df, log)

    generate_stage4_figures(ranking_df, ora_df, extended_table6,
                             specificity_df, log)

    n_high = (ranking_df["importance_label"] == "High").sum()
    n_cgc_recovered = ranking_df[
        (ranking_df["importance_label"] == "High") &
        (ranking_df["name"].isin(C.COSMIC_CGC_GENES))
    ].shape[0]
    cgc_in_data = (ranking_df["omic_type"] == "mRNA") & \
                   ranking_df["name"].isin(C.COSMIC_CGC_GENES)
    n_cgc_data = cgc_in_data.sum()

    summary = {
        "n_high_importance":           int(n_high),
        "n_ora_sig_fdr5":              int(ora_df["Sig_FDR5"].sum()) if len(ora_df) > 0 else 0,
        "n_subtype_specific_fdr5":     int(specificity_df["sig_fdr5"].sum())
                                        if len(specificity_df) > 0 else 0,
        "CGC_in_data":                 int(n_cgc_data),
        "CGC_recovered_high":          int(n_cgc_recovered),
        "CGC_recovery_rate":           round(n_cgc_recovered / max(n_cgc_data, 1), 4),
        "n_hub_genes_high_importance": int((ranking_df[ranking_df["importance_label"]=="High"]
                                           ["name"].isin(HUB_HOUSEKEEPING)).sum()),
    }
    with open(C.STAGE4_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    log.log_metrics(summary)
    log.close()
    return summary

if __name__ == "__main__":
    main()
