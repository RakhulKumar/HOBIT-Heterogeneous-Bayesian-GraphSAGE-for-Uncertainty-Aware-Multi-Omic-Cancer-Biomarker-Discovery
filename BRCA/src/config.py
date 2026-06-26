import os
import torch
from pathlib import Path

BASE_DIR = Path(os.environ.get("BRCA_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR        = BASE_DIR / "data" / "csv"
DB_DIR          = BASE_DIR / "data" / "databases"
RESULTS_DIR     = BASE_DIR / "results"
FIGURES_DIR     = RESULTS_DIR / "figures"
TABLES_DIR      = RESULTS_DIR / "tables"
LOGS_DIR        = RESULTS_DIR / "logs"
MODELS_DIR      = RESULTS_DIR / "models"

MRNA_FILE        = DATA_DIR / "BRCA_mRNA_filtered.csv"
METHYLATION_FILE = DATA_DIR / "BRCA_methylation_filtered.csv"
MIRNA_FILE       = DATA_DIR / "BRCA_miRNA_filtered.csv"
LABELS_FILE      = DATA_DIR / "BRCA_labels_filtered.csv"

STRING_PPI_FILE     = DB_DIR / "string_human_ppi.tsv"
STRING_INFO_FILE    = DB_DIR / "string_human_info.tsv"
STRING_ALIAS_FILE   = DB_DIR / "string_human_aliases.tsv"
MIRTARBASE_FILE     = DB_DIR / "hsa_MTI.csv"

GRAPH_DATA_FILE     = RESULTS_DIR / "graph_data.pkl"
NODE_FEATURES_FILE  = RESULTS_DIR / "node_features.npy"
NODE_METADATA_FILE  = RESULTS_DIR / "node_metadata.csv"
EDGES_FILE          = RESULTS_DIR / "edges.csv"
GRAPH_STATS_FILE    = RESULTS_DIR / "graph_statistics.json"
SAMPLE_LABELS_FILE  = RESULTS_DIR / "sample_labels.csv"
COMMON_SAMPLES_FILE = RESULTS_DIR / "common_samples.json"

LINK_MODEL_FILE     = MODELS_DIR / "link_prediction_model.pt"
EDGE_PRED_FILE      = RESULTS_DIR / "edge_predictions.npy"
GMM_MODEL_FILE      = RESULTS_DIR / "gmm_model.pkl"
BIOMARKER_RANK_FILE = RESULTS_DIR / "biomarker_importance_rankings.pkl"
STAGE2_SUMMARY_FILE = RESULTS_DIR / "stage2_summary.json"

CLASSIFIER_MODEL_FILE  = MODELS_DIR / "hobit_classifier.pth"
PREDICTIONS_FILE       = RESULTS_DIR / "pam50_predictions.csv"
STAGE3_SUMMARY_FILE    = RESULTS_DIR / "stage3_summary.json"
TRAINING_HISTORY_FILE  = RESULTS_DIR / "training_history.json"

ENRICHMENT_RESULTS_FILE   = RESULTS_DIR / "enrichment_results.csv"
BIOMARKER_ANALYSIS_FILE   = RESULTS_DIR / "biomarker_analysis.csv"
STAGE4_SUMMARY_FILE       = RESULTS_DIR / "stage4_summary.json"

DATASET_NAME   = "BRCA"
NUM_CLASSES    = 4
CLASS_NAMES    = ["Basal", "Her2", "LumA", "LumB"]
CLASS_TO_INT   = {"Basal": 0, "Her2": 1, "LumA": 2, "LumB": 3}
INT_TO_CLASS   = {0: "Basal", 1: "Her2", 2: "LumA", 3: "LumB"}

OMIC_TYPE_IDS  = {"mRNA": 0, "methylation": 1, "miRNA": 2}

STRING_MIN_SCORE   = 700

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
N_SEEDS    = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

LP_TRAIN_FRAC       = 0.80
LP_VAL_FRAC         = 0.10
LP_TEST_FRAC        = 0.10
LP_HIDDEN_DIM       = 128
LP_NUM_LAYERS       = 2
LP_DROPOUT          = 0.30
LP_LR               = 1e-3
LP_WEIGHT_DECAY     = 1e-4
LP_EPOCHS           = 300
LP_PATIENCE         = 20
LP_NEG_RATIO        = 1.0
LP_PRED_THRESHOLD   = 0.50

GMM_K_RANGE      = [2, 3, 4]
GMM_ALPHA_WDC    = 0.50
GMM_BETA_PR      = 0.30
GMM_GAMMA_BC     = 0.20

OT_EPSILON          = 0.01
OT_MAX_ITER         = 100
OT_STOP_THRESHOLD   = 1e-9

CLF_HIDDEN_DIM      = 512
CLF_N_LAYERS        = 4
CLF_DROPOUT         = 0.25
CLF_LR              = 1e-3
CLF_WEIGHT_DECAY    = 5e-6
CLF_EPOCHS          = 400
CLF_PATIENCE        = 60
CLF_WARMUP_EPOCHS   = 20
CLF_FOCAL_GAMMA     = 1.5
CLF_LABEL_SMOOTHING = 0.02
CLF_N_MC_SAMPLES    = 50

ECE_N_BINS          = 15

HIGH_IMPORTANCE_PROB_THRESHOLD   = 0.70
HIGH_IMPORTANCE_CONF_THRESHOLD   = 0.70
HIGH_IMPORTANCE_ENTROPY_MAX      = 0.60
UNCERTAINTY_PENALTY_WEIGHT       = 0.20

COSMIC_CGC_GENES = {
    "TP53", "BRCA1", "BRCA2", "PIK3CA", "PTEN", "AKT1", "ERBB2",
    "CDH1", "MAP3K1", "GATA3", "MYC", "CCND1", "FGFR1", "RB1",
    "CDKN2A", "MDM2", "EGFR", "CTNNB1", "KRAS", "NRAS", "BRAF",
    "NF1", "TSC1", "TSC2", "MTOR", "EP300", "CREBBP", "KMT2D",
    "ARID1A", "NCOR1", "SF3B1", "TBX3", "RUNX1", "CBFB", "FOXA1",
    "ESR1", "ERBB3", "FGFR2", "PDGFRA", "KIT", "MET", "RET",
    "SMAD4", "SMAD2", "TGFBR2", "PBRM1", "BAP1", "VHL",
}

PAM50_GENE_SET = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1",
    "CCNE1", "CDC20", "CDC6", "CDCA1", "CDH3", "CENPF", "CEP55",
    "CXXC5", "DCN", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160",
    "GRB7", "KIF2C", "KNTC2", "KRT14", "KRT17", "KRT5", "LMNB2",
    "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC",
    "NAT1", "NDC80", "NUF2", "ORC6L", "PGR", "PHGDH", "PTTG1",
    "RRM2", "SFRP1", "SLC39A6", "SLC52A3", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_LEVEL = "INFO"
