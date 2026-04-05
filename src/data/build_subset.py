from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List

from src.data.dataset_utils import SUPPORTED_VIDEO_EXTENSIONS


MANIFEST_COLUMNS = ["clip_id", "video_path", "primary_tag", "notes", "selected"]
ALLOWED_PRIMARY_TAGS = {"easy", "occlusion", "small_object", "crowded"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "clip"


def discover_video_files(source_root: str | Path) -> List[Path]:
    source_path = Path(source_root)
    return sorted(path for path in source_path.rglob("*") if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


def _dedupe_clip_ids(paths: Iterable[Path]) -> List[str]:
    seen: dict[str, int] = {}
    clip_ids: List[str] = []
    for path in paths:
        base_slug = _slugify(path.stem)
        count = seen.get(base_slug, 0)
        seen[base_slug] = count + 1
        clip_ids.append(base_slug if count == 0 else f"{base_slug}_{count + 1}")
    return clip_ids


def build_subset_manifest(source_root: str | Path, target_path: str | Path) -> dict:
    source = Path(source_root)
    manifest_path = Path(target_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    videos = discover_video_files(source)
    clip_ids = _dedupe_clip_ids(videos)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for clip_id, video_path in zip(clip_ids, videos):
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "video_path": str(video_path.resolve()),
                    "primary_tag": "",
                    "notes": "",
                    "selected": "0",
                }
            )

    return {
        "source_root": str(source.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "num_discovered_clips": len(videos),
        "allowed_primary_tags": sorted(ALLOWED_PRIMARY_TAGS),
    }
