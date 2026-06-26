# Link Prediction for Multi-Omic Cancer Subtype Classification

This repository contains the source code for a heterogeneous graph-based pipeline applied to two cancer datasets: **BRCA** (breast cancer) and **KIPAN** (pan-kidney renal cell carcinoma). The pipeline integrates multi-omic data (mRNA expression, DNA methylation, miRNA) with biological interaction networks to perform cancer subtype classification with uncertainty quantification.

## Pipeline Overview

Each dataset folder (`BRCA/`, `KIPAN/`) is self-contained and follows the same four-stage pipeline:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `src/stage1_graph_construction.py` | Builds heterogeneous multi-omic graph from expression data + STRING PPI + miRTarBase |
| 2 | `src/stage2_link_prediction.py` | Trains GraphSAGE link predictor; ranks biomarkers via GMM + centrality scoring |
| 3 | `src/stage3_classifier.py` | Bayesian OT-enhanced GraphSAGE classifier with MC-Dropout uncertainty quantification |
| 4 | `src/stage4_enrichment.py` | Over-representation analysis (ORA), subtype-specificity testing, biomarker validation |

## Datasets

### BRCA
- **Task**: PAM50 breast cancer subtype classification (Basal, Her2, LumA, LumB)
- **Data**: TCGA-BRCA mRNA, methylation (450K), miRNA profiles

### KIPAN
- **Task**: Renal cell carcinoma subtype classification (KIRC, KIRP, KICH)
- **Data**: TCGA pan-kidney mRNA, methylation, miRNA profiles

## Requirements

```bash
pip install -r requirements.txt
```

DGL must be installed separately following the [official instructions](https://www.dgl.ai/pages/start.html) for your platform and CUDA version.

## Data Setup

Place the following files in each dataset's `data/` folder before running:

```
BRCA/
└── data/
    ├── csv/
    │   ├── BRCA_mRNA_filtered.csv
    │   ├── BRCA_methylation_filtered.csv
    │   ├── BRCA_miRNA_filtered.csv
    │   └── BRCA_labels_filtered.csv
    └── databases/
        ├── string_human_ppi.tsv
        ├── string_human_info.tsv
        ├── string_human_aliases.tsv
        └── hsa_MTI.csv

KIPAN/
└── data/
    ├── csv/
    │   ├── KIPAN_mRNA.csv
    │   ├── KIPAN_methylation.csv
    │   ├── KIPAN_miRNA.csv
    │   └── KIPAN_labels.csv
    └── databases/
        ├── string_human_ppi.tsv
        ├── string_human_info.tsv
        ├── string_human_aliases.tsv
        └── hsa_MTI.csv
```

**Database sources:**
- STRING PPI: [string-db.org](https://string-db.org) — Human protein network (v12.0, score ≥ 700)
- miRTarBase: [mirtarbase.cuhk.edu.cn](https://mirtarbase.cuhk.edu.cn) — `hsa_MTI.csv`

## Running the Pipeline

Set the root environment variable, then run:

```bash
# BRCA
export BRCA_ROOT=/path/to/BRCA
cd BRCA
python run_pipeline.py              # run all stages
python run_pipeline.py --from 2    # resume from stage 2
python run_pipeline.py --stage 3   # run only stage 3

# KIPAN
export KIPAN_ROOT=/path/to/KIPAN
cd KIPAN
python run_pipeline.py
python run_pipeline.py --from 2
python run_pipeline.py --stages 3 4
```

Outputs are written to `results/` (figures, tables, model checkpoints, logs).

## Repository Structure

```
├── README.md
├── requirements.txt
├── BRCA/
│   ├── run_pipeline.py
│   ├── data/
│   │   ├── csv/          ← place omics CSV files here
│   │   └── databases/    ← place STRING + miRTarBase files here
│   ├── results/          ← generated outputs
│   └── src/
│       ├── config.py
│       ├── logger.py
│       ├── stage1_graph_construction.py
│       ├── stage2_link_prediction.py
│       ├── stage3_classifier.py
│       └── stage4_enrichment.py
└── KIPAN/
    ├── run_pipeline.py
    ├── data/
    │   ├── csv/
    │   └── databases/
    ├── results/
    └── src/
        ├── config.py
        ├── logger.py
        ├── stage1_graph_construction.py
        ├── stage2_link_prediction.py
        ├── stage3_classifier.py
        └── stage4_enrichment.py
```

## Key Design Decisions

- **Leakage-safe splits**: The 70/15/15 patient-level stratified split is performed before any feature normalization. Z-score statistics are computed from training patients only and applied to validation/test sets.
- **Multiple seeds**: All classification results are averaged over 5 random seeds (42, 123, 456, 789, 1024).
- **Uncertainty quantification**: MC-Dropout with 50 forward passes provides epistemic uncertainty estimates per prediction.
- **Optimal Transport**: Sinkhorn algorithm aligns multi-omic feature spaces before classification.
