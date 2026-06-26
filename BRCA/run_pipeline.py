import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import config as C
from logger import RunLogger


STAGES = [
    {
        "id": 1,
        "name": "Graph Construction",
        "module": "stage1_graph_construction",
        "required_outputs": [C.GRAPH_DATA_FILE, C.SAMPLE_LABELS_FILE,
                              C.RESULTS_DIR / "data_splits.json"],
        "description": "Build heterogeneous multi-omic graph with leakage-safe patient split",
    },
    {
        "id": 2,
        "name": "Link Prediction & GMM",
        "module": "stage2_link_prediction",
        "required_outputs": [C.EDGE_PRED_FILE, C.GMM_MODEL_FILE, C.BIOMARKER_RANK_FILE],
        "description": "Train GraphSAGE link predictor, compute centralities, fit GMM",
    },
    {
        "id": 3,
        "name": "Bayesian OT Classifier",
        "module": "stage3_classifier",
        "required_outputs": [C.STAGE3_SUMMARY_FILE],
        "description": "Multi-seed Bayesian OT GraphSAGE PAM50 classification + calibration",
    },
    {
        "id": 4,
        "name": "Enrichment & Biomarker Validation",
        "module": "stage4_enrichment",
        "required_outputs": [C.ENRICHMENT_RESULTS_FILE, C.BIOMARKER_ANALYSIS_FILE],
        "description": "ORA/Fisher enrichment, subtype-specificity, extended Table 6",
    },
]

def check_stage_outputs(stage: dict) -> bool:
    return all(Path(p).exists() for p in stage["required_outputs"])

def run_stage(stage: dict, log: RunLogger) -> dict:
    import importlib
    t0 = time.time()
    try:
        mod = importlib.import_module(stage["module"])
        result = mod.main()
        elapsed = time.time() - t0
        log.info(f"  Stage {stage['id']} completed in {elapsed:.1f}s")
        return {"status": "OK", "elapsed_s": round(elapsed, 2), "result": result}
    except Exception as ex:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log.error(f"  Stage {stage['id']} FAILED after {elapsed:.1f}s")
        log.error(f"  {ex}")
        log.error(tb)
        return {"status": "FAILED", "elapsed_s": round(elapsed, 2),
                "error": str(ex), "traceback": tb}

def generate_final_report(stage_outcomes: dict, log: RunLogger):
    log.section("GENERATING FINAL CONSOLIDATED REPORT")

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "dataset":       "BRCA",
        "n_seeds":       C.N_SEEDS,
        "stages":        {},
    }

    for sid in [1, 2, 3, 4]:
        outcome = stage_outcomes.get(sid, {})
        report["stages"][f"stage{sid}"] = {
            "status":    outcome.get("status", "SKIPPED"),
            "elapsed_s": outcome.get("elapsed_s", 0),
        }

    if C.GRAPH_STATS_FILE.exists():
        with open(C.GRAPH_STATS_FILE) as f:
            gs = json.load(f)
        report["graph"] = {
            "total_nodes":       gs["total_nodes"],
            "total_edges":       gs["total_edges"],
            "num_samples":       gs["num_samples"],
            "avg_degree":        gs["avg_degree"],
            "isolated_nodes":    gs["isolated_nodes"],
            "pam50_distribution":gs["pam50_distribution"],
        }

    if C.STAGE2_SUMMARY_FILE.exists():
        with open(C.STAGE2_SUMMARY_FILE) as f:
            s2 = json.load(f)
        report["link_prediction"] = {
            "test_AUC":        s2.get("link_prediction", {}).get("test_auc"),
            "test_AP":         s2.get("link_prediction", {}).get("test_ap"),
            "edges_added":     s2.get("edges_added"),
            "density_increase":s2.get("network_density_increase_pct"),
            "gmm_k":           s2.get("gmm_k"),
            "cgc_recovery_rate":s2.get("gmm_validation", {}).get("cgc_recovery_rate"),
            "pam50_recovery_rate":s2.get("gmm_validation",{}).get("pam50_rate"),
            "n_high_importance":s2.get("n_high_importance"),
        }

    if C.STAGE3_SUMMARY_FILE.exists():
        with open(C.STAGE3_SUMMARY_FILE) as f:
            s3 = json.load(f)
        agg = s3.get("aggregate", {}).get("mean", {})
        std = s3.get("aggregate", {}).get("std",  {})
        report["classification"] = {
            "accuracy_mean":      agg.get("accuracy"),
            "accuracy_std":       std.get("accuracy"),
            "balanced_acc_mean":  agg.get("balanced_acc"),
            "F1_macro_mean":      agg.get("F1_macro"),
            "F1_macro_std":       std.get("F1_macro"),
            "AUC_macro_mean":     agg.get("AUC_macro"),
            "AUC_macro_std":      std.get("AUC_macro"),
            "ECE_mean":           agg.get("ECE"),
            "ECE_std":            std.get("ECE"),
            "MCE_mean":           agg.get("MCE"),
            "MI_mean":            agg.get("MI_mean"),
            "spearman_rho_mean":  agg.get("spearman_rho"),
            "ot_alignment_used":  s3.get("ot_used"),
            "mc_samples":         s3.get("mc_samples"),
        }

    if C.STAGE4_SUMMARY_FILE.exists():
        with open(C.STAGE4_SUMMARY_FILE) as f:
            s4 = json.load(f)
        report["biomarker_validation"] = {
            "n_high_importance":           s4.get("n_high_importance"),
            "n_ora_significant_fdr5":      s4.get("n_ora_sig_fdr5"),
            "n_subtype_specific_fdr5":     s4.get("n_subtype_specific_fdr5"),
            "CGC_recovery_rate":           s4.get("CGC_recovery_rate"),
            "n_hub_genes_high_importance": s4.get("n_hub_genes_high_importance"),
        }

    report_path = C.RESULTS_DIR / "final_pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("\n" + "="*70)
    log.info("  HOBIT-BRCA FINAL PIPELINE REPORT")
    log.info("="*70)

    if "graph" in report:
        g = report["graph"]
        log.info(f"\n  GRAPH")
        log.info(f"    Nodes         : {g['total_nodes']:,}")
        log.info(f"    Edges         : {g['total_edges']:,}")
        log.info(f"    Samples       : {g['num_samples']}")
        log.info(f"    Avg degree    : {g['avg_degree']:.2f}")
        log.info(f"    PAM50 dist.   : {g['pam50_distribution']}")

    if "link_prediction" in report:
        lp = report["link_prediction"]
        log.info(f"\n  LINK PREDICTION")
        log.info(f"    Test AUC-ROC   : {lp['test_AUC']:.4f}")
        log.info(f"    Test Avg Prec  : {lp['test_AP']:.4f}")
        log.info(f"    Edges added    : {lp['edges_added']:,}")
        log.info(f"    Density +%     : {lp['density_increase']:.1f}%")
        log.info(f"    GMM k          : {lp['gmm_k']}")
        log.info(f"    CGC recovery   : {lp['cgc_recovery_rate']:.1%}")
        log.info(f"    PAM50 recovery : {lp['pam50_recovery_rate']:.1%}")

    if "classification" in report:
        cl = report["classification"]
        log.info(f"\n  PAM50 CLASSIFICATION  (mean±SD, N={C.N_SEEDS} seeds)")
        log.info(f"    Accuracy        : {cl['accuracy_mean']:.4f} ± {cl['accuracy_std']:.4f}")
        log.info(f"    Balanced Acc.   : {cl['balanced_acc_mean']:.4f}")
        log.info(f"    F1 Macro        : {cl['F1_macro_mean']:.4f} ± {cl['F1_macro_std']:.4f}")
        log.info(f"    AUC-ROC Macro   : {cl['AUC_macro_mean']:.4f} ± {cl['AUC_macro_std']:.4f}")
        log.info(f"    ECE             : {cl['ECE_mean']:.4f} ± {cl['ECE_std']:.4f}")
        log.info(f"    MCE             : {cl['MCE_mean']:.4f}")
        log.info(f"    Mut. Info (mean): {cl['MI_mean']:.4f}")
        log.info(f"    Spearman ρ(H,err): {cl['spearman_rho_mean']:.4f}")
        log.info(f"    OT alignment    : {'YES' if cl['ot_alignment_used'] else 'NO (POT not installed)'}")
        log.info(f"    MC samples      : {cl['mc_samples']}")

    if "biomarker_validation" in report:
        bv = report["biomarker_validation"]
        log.info(f"\n  BIOMARKER VALIDATION")
        log.info(f"    High-importance : {bv['n_high_importance']:,}")
        log.info(f"    ORA sig. (FDR<5%): {bv['n_ora_significant_fdr5']}")
        log.info(f"    Subtype-specific: {bv['n_subtype_specific_fdr5']}")
        log.info(f"    CGC recovery    : {bv['CGC_recovery_rate']:.1%}")
        log.info(f"    Hub genes flagged: {bv['n_hub_genes_high_importance']}")

    log.info(f"\n  OUTPUTS → {C.RESULTS_DIR}")
    log.info(f"    final_pipeline_report.json")
    log.info(f"    stage3_summary.json  (main classification results)")
    log.info(f"    tables/              ({len(list(C.TABLES_DIR.glob('*.csv')))} CSV tables)")
    log.info(f"    figures/             ({len(list(C.FIGURES_DIR.glob('*.png')))} figures)")

    log.info("="*70)
    return report_path

def main():
    parser = argparse.ArgumentParser(description="HOBIT-BRCA Pipeline Runner")
    parser.add_argument("--stage",   type=int, default=None,
                        help="Run only this stage (1-4)")
    parser.add_argument("--from",    type=int, default=1, dest="from_stage",
                        help="Start from this stage (default: 1)")
    parser.add_argument("--to",      type=int, default=4,
                        help="Stop after this stage (default: 4)")
    parser.add_argument("--check",   action="store_true",
                        help="Skip stages whose outputs already exist")
    args = parser.parse_args()

    for d in [C.RESULTS_DIR, C.FIGURES_DIR, C.TABLES_DIR,
              C.LOGS_DIR, C.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log = RunLogger("master_pipeline", C.LOGS_DIR)
    log.section("HOBIT-BRCA MASTER PIPELINE")
    log.info(f"  BASE_DIR      : {C.BASE_DIR}")
    log.info(f"  RESULTS_DIR   : {C.RESULTS_DIR}")
    log.info(f"  N seeds       : {C.N_SEEDS}")
    log.info(f"  Device        : {C.DEVICE}")

    if args.stage is not None:
        stages_to_run = [s for s in STAGES if s["id"] == args.stage]
    else:
        stages_to_run = [s for s in STAGES
                         if args.from_stage <= s["id"] <= args.to]

    log.info(f"\n  Stages to run : {[s['id'] for s in stages_to_run]}")

    stage_outcomes: dict = {}
    total_start = time.time()

    for stage in stages_to_run:
        log.section(f"STAGE {stage['id']} — {stage['name']}")
        log.info(f"  {stage['description']}")

        if args.check and check_stage_outputs(stage):
            log.info(f"  All outputs exist — SKIPPING (--check mode)")
            stage_outcomes[stage["id"]] = {"status": "SKIPPED", "elapsed_s": 0}
            continue

        outcome = run_stage(stage, log)
        stage_outcomes[stage["id"]] = outcome

        if outcome["status"] == "FAILED":
            log.error(f"\n  *** Stage {stage['id']} failed. "
                      "Stopping pipeline to prevent downstream errors. ***")
            log.error("  Fix the issue above and re-run with --from "
                      f"{stage['id']}")
            break

        log.checkpoint(f"stage{stage['id']}_done")

    total_elapsed = time.time() - total_start

    report_path = generate_final_report(stage_outcomes, log)

    log.info(f"\n  Total pipeline time : {total_elapsed:.1f}s  ({total_elapsed/60:.1f}min)")
    log.info(f"  Report saved        : {report_path}")

    all_ok = all(v.get("status") in ("OK", "SKIPPED")
                 for v in stage_outcomes.values())
    if not all_ok:
        log.error("  One or more stages FAILED.")
        sys.exit(1)

    log.info("\n  ✓  All stages completed successfully.")
    log.close()

if __name__ == "__main__":
    main()
