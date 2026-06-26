import sys
import time
import argparse
import traceback
import importlib
from pathlib import Path
import os


sys.path.insert(0, str(Path(__file__).parent / "src"))
import config as C
from logger import RunLogger

STAGE_REGISTRY = {
    1: {
        "module":      "stage1_graph_construction",
        "description": "Build heterogeneous multi-omic graph with leakage-safe patient split",
    },
    2: {
        "module":      "stage2_link_prediction",
        "description": "Train GraphSAGE link predictor, compute centralities, fit GMM",
    },
    3: {
        "module":      "stage3_classifier",
        "description": "Multi-seed Bayesian OT GraphSAGE RCC subtype classification + calibration",
    },
    4: {
        "module":      "stage4_enrichment",
        "description": "ORA/Fisher enrichment, subtype-specificity, extended Table 6",
    },
}

def run_stage(stage_num: int, log: RunLogger) -> dict:
    info = STAGE_REGISTRY[stage_num]
    mod  = importlib.import_module(info["module"])
    return mod.main()

def build_final_report(stage_summaries: dict, elapsed_total: float, log: RunLogger):

    s1 = stage_summaries.get(1, {})
    s2 = stage_summaries.get(2, {})
    s3 = stage_summaries.get(3, {})
    s4 = stage_summaries.get(4, {})

    report_lines = [
        "",
        "=" * 70,
        "  HOBIT-KIPAN FINAL PIPELINE REPORT",
        "=" * 70,
        "",
        "  GRAPH",
    ]

    if s1:
        report_lines += [
            f"      Nodes         : {s1.get('total_nodes', 'N/A'):,}",
            f"      Edges         : {s1.get('total_edges', 'N/A'):,}",
            f"      Samples       : {s1.get('num_samples', 'N/A')}",
            f"      Avg degree    : {s1.get('avg_degree', 0):.2f}",
            f"      RCC dist.     : "
            + "  ".join(f"{cls}={s1.get('subtype_distribution', {}).get(cls, '?')}"
                        for cls in C.CLASS_NAMES),
        ]

    report_lines.append("")
    report_lines.append("  LINK PREDICTION")

    if s2:
        lp = s2.get("link_prediction", {})
        report_lines += [
            f"      Test AUC-ROC   : {lp.get('test_auc', float('nan')):.4f}",
            f"      Test Avg Prec  : {lp.get('test_ap',  float('nan')):.4f}",
            f"      Edges added    : {s2.get('edges_added', 'N/A'):,}",
            f"      Density +%     : {s2.get('network_density_increase_pct', 0):.1f}%",
            f"      GMM k          : {s2.get('gmm_k', 'N/A')}",
            f"      CGC recovery   : {s2.get('cgc_recovery_rate', 0)*100:.1f}%",
        ]

    report_lines.append("")
    report_lines.append(
        f"  RCC SUBTYPE CLASSIFICATION  (mean±SD, N={C.N_SEEDS} seeds)"
    )

    if s3:
        agg = s3.get("aggregate", {})
        mn  = agg.get("mean", {})
        sd  = agg.get("std",  {})
        ci  = agg.get("ci95", {})
        report_lines += [
            f"      Accuracy        : {mn.get('accuracy',0):.4f} ± {sd.get('accuracy',0):.4f}",
            f"      Balanced Acc.   : {mn.get('balanced_acc',0):.4f}",
            f"      F1 Macro        : {mn.get('F1_macro',0):.4f} ± {sd.get('F1_macro',0):.4f}",
            f"      AUC-ROC Macro   : {mn.get('AUC_macro',0):.4f} ± {sd.get('AUC_macro',0):.4f}",
            f"      ECE             : {mn.get('ECE',0):.4f} ± {sd.get('ECE',0):.4f}",
            f"      MCE             : {mn.get('MCE',0):.4f}",
            f"      Mut. Info (mean): {mn.get('MI_mean',0):.4f}",
            f"      Spearman ρ(H,err): {mn.get('spearman_rho',0):.4f}",
            f"      OT alignment    : {'YES' if s3.get('ot_used') else 'NO'}",
            f"      MC samples      : {s3.get('mc_samples', C.CLF_N_MC_SAMPLES)}",
        ]

    report_lines.append("")
    report_lines.append("  BIOMARKER VALIDATION")

    if s4:
        report_lines += [
            f"      High-importance : {s4.get('n_high_importance', 'N/A')}",
            f"      ORA sig. (FDR<5%): {s4.get('n_ora_significant_fdr5', 'N/A')}",
            f"      Subtype-specific: {s4.get('n_subtype_specific_fdr5', 'N/A')}",
            f"      CGC recovery    : {s4.get('CGC_recovery_rate', 0)*100:.1f}%",
            f"      Hub genes flagged: {s4.get('n_hub_genes_high', 'N/A')}",
        ]

    report_lines += [
        "",
        f"  OUTPUTS → {C.RESULTS_DIR}",
        "      final_pipeline_report.json",
        "      stage3_summary.json  (main classification results)",
        f"      tables/              ({len(list(C.TABLES_DIR.glob('*.csv')))} CSV tables)",
        f"      figures/             ({len(list(C.FIGURES_DIR.glob('*.png')))} figures)",
        "=" * 70,
        "",
        f"  Total pipeline time : {elapsed_total:.1f}s  ({elapsed_total/60:.1f}min)",
    ]

    for line in report_lines:
        log.info(line)

    import json
    report = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "KIPAN",
        "n_seeds": C.N_SEEDS,
        "stages": {
            str(k): {"status": "OK", "elapsed_s": stage_summaries.get(k, {}).get("_elapsed", 0)}
            for k in stage_summaries
        },
        "graph":          s1,
        "link_prediction": s2,
        "classification": s3.get("aggregate", {}) if s3 else {},
        "biomarker_validation": s4,
    }
    with open(C.RESULTS_DIR / "final_pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"  Report saved        : {C.RESULTS_DIR / 'final_pipeline_report.json'}")

def main():
    parser = argparse.ArgumentParser(description="HOBIT-KIPAN Pipeline")
    parser.add_argument("--from", dest="from_stage", type=int, default=1,
                        help="Resume from this stage (1-4)")
    parser.add_argument("--stages", dest="stages", type=int, nargs="+",
                        help="Run only these specific stages (e.g. --stages 3 4)")
    args = parser.parse_args()

    if args.stages:
        stages_to_run = sorted(args.stages)
    else:
        stages_to_run = list(range(args.from_stage, 5))

    for d in [C.RESULTS_DIR, C.LOGS_DIR, C.FIGURES_DIR,
              C.TABLES_DIR, C.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log = RunLogger("master_pipeline", C.LOGS_DIR)
    log.section("HOBIT-KIPAN MASTER PIPELINE")
    log.info(f"  BASE_DIR      : {C.BASE_DIR}")
    log.info(f"  RESULTS_DIR   : {C.RESULTS_DIR}")
    log.info(f"  N seeds       : {C.N_SEEDS}")
    log.info(f"  Device        : {C.DEVICE}")
    log.info(f"\n  Stages to run : {stages_to_run}")
    log.info("")

    pipeline_start  = time.time()
    stage_summaries = {}
    failed_stage    = None

    for stage_num in stages_to_run:
        info = STAGE_REGISTRY[stage_num]
        log.section(f"STAGE {stage_num} — {info['description'].split(',')[0]}")
        log.info(f"  {info['description']}")

        stage_start = time.time()
        try:
            summary = run_stage(stage_num, log)
            elapsed = time.time() - stage_start
            if isinstance(summary, dict):
                summary["_elapsed"] = elapsed
            else:
                summary = {"_elapsed": elapsed}
            stage_summaries[stage_num] = summary
            log.info(f"  Stage {stage_num} completed in {elapsed:.1f}s")
            log.checkpoint(f"stage{stage_num}_done")
        except Exception as exc:
            elapsed = time.time() - stage_start
            log.error(f"  Stage {stage_num} FAILED after {elapsed:.1f}s")
            log.error(f"  {exc}")
            log.error(f"{traceback.format_exc()}")
            log.error(f"\n  *** Stage {stage_num} failed. "
                      f"Stopping pipeline to prevent downstream errors. ***")
            log.error(f"  Fix the issue above and re-run with --from {stage_num}")
            failed_stage = stage_num
            break
        log.info("")

    elapsed_total = time.time() - pipeline_start

    log.section("GENERATING FINAL CONSOLIDATED REPORT")
    build_final_report(stage_summaries, elapsed_total, log)
    log.info(f"\n  Total pipeline time : {elapsed_total:.1f}s  ({elapsed_total/60:.1f}min)")

    if failed_stage:
        log.error(f"  One or more stages FAILED.")
    else:
        log.info(f"\n  ✓  All stages completed successfully.")

    log.close()
    return 0 if failed_stage is None else 1

if __name__ == "__main__":
    sys.exit(main())
