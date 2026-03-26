from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load_summary(summary_or_path: Mapping[str, Any] | str | Path) -> dict:
    if isinstance(summary_or_path, Mapping):
        return dict(summary_or_path)
    return json.loads(Path(summary_or_path).read_text(encoding="utf-8"))


def summarize_failures(summary_or_path: Mapping[str, Any] | str | Path) -> dict:
    summary = _load_summary(summary_or_path)
    artifacts = summary.get("artifacts", {})
    issues = []

    if summary.get("input_type") == "video" and "video_overlay" not in artifacts:
        issues.append("missing_video_overlay")
    if summary.get("input_type") == "image" and "image_overlay" not in artifacts:
        issues.append("missing_image_overlay")
    if summary.get("num_frames", 0) == 0:
        issues.append("empty_run")
    if "fallback_reason" in artifacts:
        issues.append("fallback_triggered")
    if summary.get("runtime_sec", 0) <= 0:
        issues.append("invalid_runtime")

    if not issues:
        status = "ok"
        message = "No obvious qualitative failure signals detected in run_summary.json."
    else:
        status = "needs_review"
        message = "Qualitative run completed with issues that should be reviewed."

    return {
        "status": status,
        "issues": issues,
        "video_mode": artifacts.get("video_mode"),
        "message": message,
        "input_path": summary.get("input_path"),
        "prompt": summary.get("prompt"),
    }
