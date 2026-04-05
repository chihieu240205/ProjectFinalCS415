#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight fixed subset benchmark from subset_manifest.csv.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True, help="Path to subset_manifest.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory for subset evaluation outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on selected rows for quick dry runs.")
    parser.add_argument("--grounding_ckpt", default=None, help="Optional override for the GroundingDINO checkpoint path.")
    parser.add_argument("--sam2_ckpt", default=None, help="Optional override for the SAM2 checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional runtime device override, for example `cuda` or `cpu`.")
    args = parser.parse_args()

    from src.eval.eval_rvos import run_eval_subset
    from src.utils.io import load_project_config
    from src.utils.logger import configure_logger

    logger = configure_logger()
    config = load_project_config(
        ROOT / args.config,
        ROOT / "configs" / "grounding_dino.yaml",
        ROOT / "configs" / "sam2.yaml",
    )
    if args.grounding_ckpt is not None:
        config["grounding_dino"]["checkpoint_path"] = args.grounding_ckpt
    if args.sam2_ckpt is not None:
        config["sam2"]["checkpoint_path"] = args.sam2_ckpt
    if args.device is not None:
        config["runtime"]["device"] = args.device
        config["grounding_dino"]["device"] = args.device
        config["sam2"]["device"] = args.device

    summary = run_eval_subset(
        manifest_path=args.manifest,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    logger.info("Subset run complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
