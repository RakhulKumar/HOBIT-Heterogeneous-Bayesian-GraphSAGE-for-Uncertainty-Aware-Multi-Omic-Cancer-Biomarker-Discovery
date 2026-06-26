import os
import torch
from pathlib import Path

BASE_DIR = Path(os.environ.get("KIPAN_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR      = BASE_DIR / "data" / "csv"
DATABASE_DIR  = BASE_DIR / "data" / "databases"

MRNA_FILE         = DATA_DIR / "KIPAN_mRNA.csv"
METHYLATION_FILE  = DATA_DIR / "KIPAN_methylation.csv"
MIRNA_FILE        = DATA_DIR / "KIPAN_miRNA.csv"
LABELS_FILE       = DATA_DIR / "KIPAN_labels.csv"

STRING_PPI_FILE       = DATABASE_DIR / "string_human_ppi.tsv"
STRING_INFO_FILE      = DATABASE_DIR / "string_human_info.tsv"
STRING_ALIASES_FILE   = DATABASE_DIR / "string_human_aliases.tsv"
MIRTARBASE_FILE       = DATABASE_DIR / "hsa_MTI.csv"

RESULTS_DIR   = BASE_DIR / "results"
LOGS_DIR      = RESULTS_DIR / "logs"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
MODELS_DIR    = RESULTS_DIR / "models"

GRAPH_DATA_FILE       = RESULTS_DIR / "graph_data.pkl"
SAMPLE_LABELS_FILE    = RESULTS_DIR / "sample_labels.csv"
DATA_SPLITS_FILE      = RESULTS_DIR / "data_splits.json"
BIOMARKER_RANK_FILE   = RESULTS_DIR / "biomarker_importance_rankings.pkl"
EDGE_PRED_FILE        = RESULTS_DIR / "edge_predictions.npy"
GMM_MODEL_FILE        = RESULTS_DIR / "gmm_model.pkl"
STAGE2_SUMMARY_FILE   = RESULTS_DIR / "stage2_summary.json"
STAGE3_SUMMARY_FILE   = RESULTS_DIR / "stage3_summary.json"

LINK_MODEL_FILE       = RESULTS_DIR / "link_pred_model.pt"

STAGE                 = "KIPAN"

LABEL_COLUMN  = "subtype"

CLASS_NAMES   = ["KIRC", "KIRP", "KICH"]
CLASS_TO_INT  = {"KIRC": 0, "KIRP": 1, "KICH": 2}
INT_TO_CLASS  = {0: "KIRC", 1: "KIRP", 2: "KICH"}
NUM_CLASSES   = 3

OMIC_TYPE_IDS = {"mRNA": 0, "methylation": 1, "miRNA": 2}

STRING_MIN_SCORE  = 700
STRING_SPECIES    = "9606"

RANDOM_SEEDS   = [42, 123, 456, 789, 1024]
N_SEEDS        = len(RANDOM_SEEDS)
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15

LP_HIDDEN_DIM     = 256
LP_N_LAYERS       = 2
LP_NUM_LAYERS     = LP_N_LAYERS
LP_DROPOUT        = 0.3
LP_LR             = 0.005
LP_WEIGHT_DECAY   = 1e-5
LP_EPOCHS         = 300
LP_PATIENCE       = 40
LP_TRAIN_RATIO    = 0.80
LP_VAL_RATIO      = 0.10
LP_TEST_RATIO     = 0.10
LP_TRAIN_FRAC     = LP_TRAIN_RATIO
LP_VAL_FRAC       = LP_VAL_RATIO
LP_PRED_THRESHOLD = 0.5

GMM_K_RANGE             = [2, 3, 4]
UNCERTAINTY_PENALTY_WEIGHT = 0.1

GMM_ALPHA_WDC     = 0.4
GMM_BETA_PR       = 0.4
GMM_GAMMA_BC      = 0.2

CLF_HIDDEN_DIM      = 512
CLF_N_LAYERS        = 3
CLF_DROPOUT         = 0.3
CLF_LR              = 5e-4
CLF_WEIGHT_DECAY    = 1e-4
CLF_EPOCHS          = 400
CLF_PATIENCE        = 60
CLF_WARMUP_EPOCHS   = 20
CLF_FOCAL_GAMMA     = 2.0
CLF_LABEL_SMOOTHING = 0.05
CLF_N_MC_SAMPLES    = 50
ECE_N_BINS          = 15

OT_EPSILON          = 0.1
OT_MAX_ITER         = 500
OT_STOP_THRESHOLD   = 1e-7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COSMIC_CGC_GENES = {
    "VHL","PBRM1","SETD2","BAP1","KDM5C","MTOR","PIK3CA","TP53","PTEN",
    "TSC1","TSC2","MET","FH","FLCN","SDHA","SDHB","SDHC","SDHD","NF2",
    "EGFR","AKT1","KRAS","NRAS","BRAF","RB1","CDKN2A","CDK4","MDM2",
    "MYC","MYCN","CCND1","CCND2","CCNE1","SMARCB1","ARID1A","KIT","PDGFRA",
    "FGFR3","IDH1","IDH2","JAK2","STAT3","NFE2L2","KEAP1","STK11",
    "NF1","RET","ALK","ROS1","ERBB2","BRCA1","BRCA2",
}

VHL_HIF_PATHWAY = {
    "VHL","HIF1A","HIF2A","EPAS1","EGLN1","EGLN2","EGLN3","VHL",
    "HIF1AN","ARNT","VEGFA","VEGFB","VEGFC","VEGFD","PGF",
    "GLUT1","SLC2A1","CA9","CA12","LDHA","PDK1","PFKL","ENO1",
    "BNIP3","BNIP3L","CITED2","ADORA2B","EPO","EPOR",
}

MTOR_PATHWAY = {
    "MTOR","RPTOR","RICTOR","TSC1","TSC2","PTEN","PIK3CA","PIK3CB",
    "PIK3R1","AKT1","AKT2","AKT3","S6K1","RPS6KB1","RPS6KB2",
    "EIF4EBP1","MLST8","DEPTOR","ULK1","BECN1","ATG13","AMPK",
    "PRKAA1","PRKAA2","STK11","RHEB","REDD1","DDIT4",
    "PDK1","PDPK1","SGK1","FOXO1","FOXO3","GSK3B",
}

CHROMATIN_REMODELLING = {
    "PBRM1","SETD2","BAP1","KDM5C","KDM6A","ARID1A","ARID1B","ARID2",
    "SMARCA4","SMARCB1","SMARCC1","SMARCC2","SMARCD1","SMARCE1",
    "KDM1A","KDM4C","EZH2","EED","SUZ12","DNMT3A","DNMT3B","TET2",
    "HDAC1","HDAC2","HDAC3","EP300","CREBBP","SETDB1","NSD1",
}

MET_PATHWAY = {
    "MET","HGF","HGFAC","PLAUR","PLAU","ST14","PTPRJ","SEMA4D",
    "GAB1","GRB2","SOS1","KRAS","NRAS","HRAS","BRAF","MAP2K1",
    "MAP2K2","MAPK1","MAPK3","FRS2","EGF","EGFR","ERBB2","ERBB3",
    "STAT3","SRC","FAK","PTK2","PIK3R1","CRKL","CRK",
}

HALLMARK_RENAL_CARCINOMA = {
    "VHL","PBRM1","SETD2","BAP1","KDM5C","MTOR","TSC1","TSC2",
    "PTEN","MET","FH","FLCN","SDHA","SDHB","HIF1A","EPAS1",
    "CA9","CA12","VEGFA","CCND1","MYC","CDKN2A","TP53","EGFR",
}

HALLMARK_OXIDATIVE_PHOSPH = {
    "MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-ND5","MT-ND6",
    "MT-CYB","MT-CO1","MT-CO2","MT-CO3","MT-ATP6","MT-ATP8",
    "NDUFA1","NDUFA2","NDUFA3","NDUFA4","NDUFA5","NDUFA6",
    "NDUFB1","NDUFB2","NDUFB3","NDUFB4","NDUFB5","NDUFB6",
    "UQCRC1","UQCRC2","COX4I1","COX5A","COX6A1","ATP5A1",
    "SDHA","SDHB","SDHC","SDHD","ETFA","ETFB",
}

HALLMARK_MYC_TARGETS = {
    "MYC","MYCN","MYCL","MAX","MXD1","MXD2","MXD3","MXD4",
    "MXI1","MNT","MLX","MLXIP","MLXIPL","FBXW7","AURKA","AURKB",
    "CDK4","CCND1","CCND2","CCND3","CCNE1","CCNE2","CDK2",
    "RPS6KB1","RPS6KB2","NPM1","NCL","NOP56","NOP58","BYSL",
    "RPS3","RPS9","RPS11","RPS18","RPS27A","RPL5","RPL6",
}

HALLMARK_G2M_CHECKPOINT = {
    "CCNA2","CCNB1","CCNB2","CDK1","CDC20","CDC25A","CDC25B","CDC25C",
    "PLK1","BUB1","BUB1B","BUB3","MAD1L1","MAD2L1","CENPA","CENPB",
    "KIF11","KIF2C","ESPL1","AURKB","INCENP","BIRC5","UBE2C",
    "RAD51","BRCA1","BRCA2","CHEK1","CHEK2","ATM","ATR","WEE1",
}

HALLMARK_APOPTOSIS = {
    "BCL2","BCL2L1","BCL2L2","MCL1","BAX","BAK1","BID","PUMA",
    "NOXA","BMF","HRK","BIK","BAD","CASP3","CASP7","CASP8",
    "CASP9","APAF1","CYCS","SMAC","DIABLO","XIAP","BIRC2","BIRC3",
    "TP53","MDM2","MDM4","PTEN","AKT1","BCL2L11","BBC3","PMAIP1",
}

HALLMARK_EMT = {
    "CDH1","CDH2","VIM","FN1","SNAI1","SNAI2","ZEB1","ZEB2","TWIST1",
    "TWIST2","MMP2","MMP9","MMP14","TGFB1","TGFB2","TGFB3","TGFBR1",
    "TGFBR2","SMAD2","SMAD3","SMAD4","ACTA2","COL1A1","COL3A1",
    "CTNNB1","AXIN1","APC","WNT1","WNT3A","WNT5A","FZD1",
}

HUB_HOUSEKEEPING_GENES = {
    "ACTB","GAPDH","HSPD1","HSP90AA1","HSP90AB1","HSPA1A","HSPA1B",
    "UBC","UBA52","UBB","RPS27A","RPS18","RPS11","RPS9","RPS6",
    "RPS3","RPL5","RPL6","RPL7","RPS16","RPS23","RPS3A",
    "EEF1A1","EEF2","YWHAZ","YWHAB","ACTG1",
}

KIRC_SIGNATURE = {
    "VHL","CA9","CA12","EPAS1","HIF1A","VEGFA","LDHA","PDK1","SLC2A1",
    "CCND1","MET","PBRM1","SETD2","BAP1","KDM5C","MTOR","TP53BP1",
}

KIRP_SIGNATURE = {
    "MET","KRAS","NRF2","NFE2L2","KEAP1","CUL3","CDKN2A","SETD2",
    "BAP1","TFE3","TFEB","MITF","FLCN","FNIP1","FNIP2","AMPK",
    "PRKAA1","STK11","CDKN1A","CDKN1B",
}

KICH_SIGNATURE = {
    "TP53","PTEN","CDKN2A","RB1","CCND1","CDK4","CDK6","TERT",
    "MTOR","TSC1","TSC2","FOXA1","FOXA2","KIT","PDGFRA","EGFR",
    "NF2","LZTR1","SMAD4",
}

KIPAN_GENE_SETS = {
    "VHL_HIF_PATHWAY":          VHL_HIF_PATHWAY,
    "MTOR_PI3K_PATHWAY":        MTOR_PATHWAY,
    "CHROMATIN_REMODELLING":    CHROMATIN_REMODELLING,
    "MET_PAPILLARY_PATHWAY":    MET_PATHWAY,
    "HALLMARK_RENAL_CARCINOMA": HALLMARK_RENAL_CARCINOMA,
    "HALLMARK_OXPHOS":          HALLMARK_OXIDATIVE_PHOSPH,
    "HALLMARK_MYC_TARGETS":     HALLMARK_MYC_TARGETS,
    "HALLMARK_G2M_CHECKPOINT":  HALLMARK_G2M_CHECKPOINT,
    "HALLMARK_APOPTOSIS":       HALLMARK_APOPTOSIS,
    "HALLMARK_EMT":             HALLMARK_EMT,
    "COSMIC_CGC":               COSMIC_CGC_GENES,
    "KIRC_SIGNATURE":           KIRC_SIGNATURE,
    "KIRP_SIGNATURE":           KIRP_SIGNATURE,
    "KICH_SIGNATURE":           KICH_SIGNATURE,
    "HUB_HOUSEKEEPING":         HUB_HOUSEKEEPING_GENES,
}

KIPAN_BIOMARKER_ANNOTATIONS = {

    "VHL":     {"general_role": "Tumour suppressor; E3 ubiquitin ligase; mutated in >75% KIRC",
                "subtype_role": "KIRC", "reference": "Gnarra et al. 1994"},
    "EPAS1":   {"general_role": "HIF-2α; drives KIRC progression under VHL loss",
                "subtype_role": "KIRC", "reference": "Gordan & Simon 2007"},
    "HIF1A":   {"general_role": "Hypoxia master regulator; transcribes VEGF, GLUT1, CA9",
                "subtype_role": "KIRC", "reference": "Semenza 2012"},
    "CA9":     {"general_role": "Carbonic anhydrase IX; hypoxia biomarker; KIRC diagnostic",
                "subtype_role": "KIRC", "reference": "Leibovich et al. 2007"},
    "VEGFA":   {"general_role": "VEGF-A; angiogenesis driver; sunitinib target in KIRC",
                "subtype_role": "KIRC", "reference": "Escudier et al. 2007"},

    "PBRM1":   {"general_role": "SWI/SNF complex; 2nd most mutated in KIRC after VHL",
                "subtype_role": "KIRC", "reference": "Varela et al. 2011"},
    "SETD2":   {"general_role": "H3K36 methyltransferase; mutated in KIRC and KIRP",
                "subtype_role": "KIRC/KIRP", "reference": "Dalgliesh et al. 2010"},
    "BAP1":    {"general_role": "Deubiquitinase; KIRC tumour suppressor; poor prognosis",
                "subtype_role": "KIRC", "reference": "Pena-Llopis et al. 2012"},
    "KDM5C":   {"general_role": "H3K4 demethylase; mutated in KIRC",
                "subtype_role": "KIRC", "reference": "Dalgliesh et al. 2010"},

    "MTOR":    {"general_role": "mTOR kinase; temsirolimus target; activated in all RCC",
                "subtype_role": "pan-RCC", "reference": "Hudes et al. 2007"},
    "TSC1":    {"general_role": "Hamartin; TSC1/2 complex inhibits mTOR; mutated in KICH",
                "subtype_role": "KICH", "reference": "Guo et al. 2012"},
    "TSC2":    {"general_role": "Tuberin; mTOR regulator; mutated in chromophobe RCC",
                "subtype_role": "KICH", "reference": "Guo et al. 2012"},
    "PTEN":    {"general_role": "Phosphatase; PI3K/AKT suppressor; mutated in RCC",
                "subtype_role": "pan-RCC", "reference": "Henske & El-Hashemite 2012"},

    "MET":     {"general_role": "MET kinase; hereditary KIRP driver; cabozantinib target",
                "subtype_role": "KIRP", "reference": "Choueiri et al. 2015"},
    "NFE2L2":  {"general_role": "NRF2; oxidative stress master regulator; KIRP type 2",
                "subtype_role": "KIRP", "reference": "Linehan & Ricketts 2013"},
    "CDKN2A":  {"general_role": "p16/ARF; cell cycle brake; mutated in KICH",
                "subtype_role": "KICH", "reference": "Guo et al. 2012"},

    "TP53":    {"general_role": "Guardian of genome; ~30% KICH; rare in KIRC/KIRP",
                "subtype_role": "KICH", "reference": "Teng et al. 2018"},

    "ACTB":    {"general_role": "Actin; housekeeping — network connectivity artifact",
                "subtype_role": "N/A (hub)", "reference": ""},
    "GAPDH":   {"general_role": "Glycolytic enzyme; housekeeping — network hub artifact",
                "subtype_role": "N/A (hub)", "reference": ""},
    "UBC":     {"general_role": "Ubiquitin; network hub gene",
                "subtype_role": "N/A (hub)", "reference": ""},

    "EGFR":    {"general_role": "EGF receptor; overexpressed in KICH; erlotinib studied",
                "subtype_role": "KICH", "reference": "Choueiri et al. 2008"},
    "MYC":     {"general_role": "Oncogene; drives ribosome biogenesis; pan-cancer",
                "subtype_role": "pan-RCC", "reference": "Dang 2012"},
    "CCND1":   {"general_role": "Cyclin D1; G1/S checkpoint; elevated in KICH",
                "subtype_role": "KICH", "reference": "Guo et al. 2012"},
    "CTNNB1":  {"general_role": "β-catenin; Wnt pathway; activated in some KIRP",
                "subtype_role": "KIRP", "reference": "Linehan & Ricketts 2013"},
    "AKT1":    {"general_role": "PI3K/AKT kinase; survival signalling; pan-RCC",
                "subtype_role": "pan-RCC", "reference": "Henske & El-Hashemite 2012"},
}
