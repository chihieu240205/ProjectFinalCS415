#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image or video inference for smoke or custom qualitative validation.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base runtime config.")
    parser.add_argument("--input_video", required=True, help="Path to the input image or video.")
    parser.add_argument("--prompt", required=True, help="Text prompt for GroundingDINO.")
    parser.add_argument("--output_dir", required=True, help="Directory for saved artifacts.")
    parser.add_argument("--run_name", default=None, help="Optional subdirectory name for this run under output_dir.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frame override for videos.")
    parser.add_argument("--grounding_ckpt", default=None, help="Optional override for the GroundingDINO checkpoint path.")
    parser.add_argument("--sam2_ckpt", default=None, help="Optional override for the SAM2 checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional runtime device override, for example `cuda` or `cpu`.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from src.models.pipeline import run_inference
    from src.utils.io import load_project_config
    from src.utils.logger import configure_logger

    logger = configure_logger()
    config = load_project_config(
        ROOT / args.config,
        ROOT / "configs" / "grounding_dino.yaml",
        ROOT / "configs" / "sam2.yaml",
    )
    if args.max_frames is not None:
        config["runtime"]["max_frames"] = args.max_frames
    if args.grounding_ckpt is not None:
        config["grounding_dino"]["checkpoint_path"] = args.grounding_ckpt
    if args.sam2_ckpt is not None:
        config["sam2"]["checkpoint_path"] = args.sam2_ckpt
    if args.device is not None:
        config["runtime"]["device"] = args.device
        config["grounding_dino"]["device"] = args.device
        config["sam2"]["device"] = args.device

    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name

    summary = run_inference(
        input_path=args.input_video,
        prompt=args.prompt,
        config=config,
        output_dir=output_dir,
    )
    logger.info("Smoke run complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
