from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.data.build_subset import ALLOWED_PRIMARY_TAGS, MANIFEST_COLUMNS
from src.models.pipeline import run_inference
from src.utils.io import ensure_dir, write_json


def load_subset_manifest(manifest_path: str | Path) -> List[dict]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    return rows


def _is_selected(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validate_row(row: dict) -> None:
    if not row["clip_id"].strip():
        raise ValueError("Each selected manifest row must have a clip_id.")
    if not row["video_path"].strip():
        raise ValueError(f"Manifest row {row['clip_id']} is missing video_path.")
    primary_tag = row["primary_tag"].strip()
    if primary_tag not in ALLOWED_PRIMARY_TAGS:
        raise ValueError(
            f"Manifest row {row['clip_id']} has invalid primary_tag '{primary_tag}'. "
            f"Allowed tags: {sorted(ALLOWED_PRIMARY_TAGS)}"
        )


def _resolve_prompt(row: dict) -> str:
    notes = str(row.get("notes", "")).strip()
    if notes.startswith("prompt="):
        prompt_section = notes.split("|", 1)[0].strip()
        prompt = prompt_section[len("prompt=") :].strip()
        if prompt:
            return prompt
    if notes:
        return notes
    return str(row.get("primary_tag", "")).strip()


def selected_rows(rows: Iterable[dict], limit: int | None = None) -> List[dict]:
    selected = [row for row in rows if _is_selected(row.get("selected", "0"))]
    if limit is not None:
        selected = selected[:limit]
    for row in selected:
        _validate_row(row)
    return selected


def run_eval_subset(
    manifest_path: str | Path,
    config: Dict[str, Any],
    output_dir: str | Path,
    limit: int | None = None,
    summary_filename: str = "subset_run_summary.json",
    summary_overrides: Dict[str, Any] | None = None,
) -> dict:
    output_root = ensure_dir(output_dir)
    rows = load_subset_manifest(manifest_path)
    chosen_rows = selected_rows(rows, limit=limit)

    aggregate: List[dict] = []
    tag_counts: dict[str, int] = {}

    for row in chosen_rows:
        clip_output_dir = output_root / row["clip_id"]
        prompt = _resolve_prompt(row)
        summary = run_inference(
            input_path=row["video_path"],
            prompt=prompt,
            config=config,
            output_dir=clip_output_dir,
        )
        summary["clip_id"] = row["clip_id"]
        summary["primary_tag"] = row["primary_tag"]
        summary["manifest_notes"] = row["notes"]
        summary["manifest_prompt"] = prompt
        aggregate.append(summary)
        tag_counts[row["primary_tag"]] = tag_counts.get(row["primary_tag"], 0) + 1

    run_summary = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "num_selected": len(chosen_rows),
        "num_completed": len(aggregate),
        "tag_counts": tag_counts,
        "clips": aggregate,
    }
    if summary_overrides:
        run_summary.update(summary_overrides)
    write_json(output_root / summary_filename, run_summary)
    return run_summary
