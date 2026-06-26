import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime

class RunLogger:

    def __init__(self, stage_name: str, logs_dir: Path, level: str = "INFO"):
        self.stage_name = stage_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = logs_dir / f"{stage_name}_{ts}.log"

        self.logger = logging.getLogger(stage_name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        self._start_time = time.time()
        self._checkpoints: dict = {}
        self._metrics: dict = {}

        self.info(f"{'='*70}")
        self.info(f"  STAGE : {stage_name}")
        self.info(f"  LOG   : {self.log_path}")
        self.info(f"  START : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"{'='*70}")

    def info(self, msg: str):    self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str):   self.logger.error(msg)
    def debug(self, msg: str):   self.logger.debug(msg)

    def section(self, title: str):
        self.info("")
        self.info("=" * 70)
        self.info(f"  {title}")
        self.info("=" * 70)

    def checkpoint(self, name: str):
        elapsed = time.time() - self._start_time
        self._checkpoints[name] = elapsed
        self.info(f"  ✓ CHECKPOINT [{name}]  elapsed={elapsed:.1f}s")

    def log_metric(self, key: str, value, fmt: str = ".4f"):
        self._metrics[key] = value
        if isinstance(value, float):
            self.info(f"  METRIC  {key:<40} = {value:{fmt}}")
        else:
            self.info(f"  METRIC  {key:<40} = {value}")

    def log_metrics(self, metrics: dict, fmt: str = ".4f"):
        for k, v in metrics.items():
            self.log_metric(k, v, fmt)

    def save_summary(self, extra: dict = None):
        elapsed = time.time() - self._start_time
        summary = {
            "stage": self.stage_name,
            "start_time": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "checkpoints_seconds": self._checkpoints,
            "metrics": {
                k: (float(v) if hasattr(v, "__float__") else str(v))
                for k, v in self._metrics.items()
            },
        }
        if extra:
            summary.update(extra)

        summary_path = self.log_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.info(f"  Summary saved → {summary_path}")
        return summary

    def close(self):
        elapsed = time.time() - self._start_time
        self.info("")
        self.info("=" * 70)
        self.info(f"  STAGE COMPLETE : {self.stage_name}")
        self.info(f"  TOTAL ELAPSED  : {elapsed:.1f}s  ({elapsed/60:.1f}min)")
        self.info("=" * 70)
        self.save_summary()
