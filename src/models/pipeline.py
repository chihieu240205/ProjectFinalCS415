from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.data.extract_frames import extract_video_frames, read_image_rgb
from src.eval.metrics import box_mask_iou
from src.models.grounding import load_grounding_model, predict_boxes
from src.models.sam2_wrapper import predict_image_masks, propagate_video_masks
from src.utils.io import ensure_dir, write_json
from src.vis.overlay_masks import draw_boxes, overlay_mask
from src.vis.save_video import save_video


def _merge_masks(masks: List[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=bool)
    return np.any(np.stack([mask.astype(bool) for mask in masks], axis=0), axis=0)


def _mask_to_box(mask: np.ndarray, fallback_box: list[float] | np.ndarray) -> list[float]:
    mask_array = np.asarray(mask).astype(bool)
    points = np.argwhere(mask_array)
    if points.size == 0:
        return np.asarray(fallback_box, dtype=float).tolist()
    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def _scheduled_reground_frames(num_frames: int, interval_frames: int) -> List[int]:
    if interval_frames <= 0:
        return []
    return list(range(interval_frames, num_frames, interval_frames))


def _run_video_with_periodic_regrounding(
    frames: List[np.ndarray],
    prompt: str,
    grounding_cfg: Dict[str, Any],
    sam2_cfg: Dict[str, Any],
    ablation_cfg: Dict[str, Any],
    grounding_model,
) -> Dict[str, Any]:
    reground_cfg = ablation_cfg.get("regrounding", {})
    interval_frames = int(reground_cfg.get("interval_frames", 10))
    min_match_iou = float(reground_cfg.get("min_match_iou", 0.1))
    attempt_frames = _scheduled_reground_frames(len(frames), interval_frames)
    current_seed_box = None

    initial_boxes = predict_boxes(frames[0], prompt, grounding_cfg, model=grounding_model)
    if not initial_boxes["boxes_xyxy"]:
        raise RuntimeError(
            f"No detections found in the first frame for prompt: {prompt}. Try a simpler prompt such as `person`."
        )
    current_seed_box = np.asarray(initial_boxes["boxes_xyxy"][0], dtype=float)

    collected_masks: List[np.ndarray] = []
    boxes_per_frame: List[list[float]] = []
    attempted_frames: List[int] = []
    successful_frames: List[int] = []
    used_fallback = False
    fallback_reasons: List[str] = []
    segment_start = 0

    for attempt_frame in attempt_frames + [len(frames) - 1]:
        segment_frames = frames[segment_start : attempt_frame + 1]
        video_masks = propagate_video_masks(segment_frames, [current_seed_box.tolist()], sam2_cfg)
        segment_masks = [np.asarray(mask).astype(bool) for mask in video_masks["masks"]]
        if video_masks["mode"] != "sam2_video_predictor":
            used_fallback = True
            if "fallback_reason" in video_masks:
                fallback_reasons.append(str(video_masks["fallback_reason"]))

        if segment_start == 0:
            collected_masks.extend(segment_masks)
            boxes_per_frame.extend([current_seed_box.tolist()] * len(segment_masks))
        else:
            collected_masks.extend(segment_masks[1:])
            boxes_per_frame.extend([current_seed_box.tolist()] * max(len(segment_masks) - 1, 0))

        if attempt_frame == len(frames) - 1:
            break

        current_mask = segment_masks[-1] if segment_masks else np.zeros(frames[0].shape[:2], dtype=bool)
        attempted_frames.append(attempt_frame)
        candidate_boxes = predict_boxes(frames[attempt_frame], prompt, grounding_cfg, model=grounding_model)
        best_box = None
        best_iou = -1.0
        for candidate_box in candidate_boxes["boxes_xyxy"]:
            candidate_iou = box_mask_iou(candidate_box, current_mask)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_box = candidate_box

        if best_box is not None and best_iou >= min_match_iou:
            current_seed_box = np.asarray(best_box, dtype=float)
            successful_frames.append(attempt_frame)
        else:
            current_seed_box = np.asarray(_mask_to_box(current_mask, current_seed_box), dtype=float)

        segment_start = attempt_frame

    result: Dict[str, Any] = {
        "masks": collected_masks,
        "mode": "sam2_video_predictor_regrounding" if not used_fallback else "regrounding_with_fallback",
        "boxes_per_frame": boxes_per_frame,
        "num_reground_attempts": len(attempted_frames),
        "num_reground_successes": len(successful_frames),
        "reground_frames": attempted_frames if reground_cfg.get("record_frames", True) else [],
        "reground_success_frames": successful_frames,
        "reground_matching": str(reground_cfg.get("matching", "iou")),
        "min_match_iou": min_match_iou,
        "reground_interval": interval_frames,
    }
    if used_fallback:
        result["fallback_reason"] = "; ".join(dict.fromkeys(fallback_reasons)) if fallback_reasons else "segment fallback"
    return result


def run_inference(
    input_path: str | Path,
    prompt: str,
    config: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_root = ensure_dir(output_dir)
    runtime_cfg = config["runtime"]
    grounding_cfg = config["grounding_dino"]
    sam2_cfg = config["sam2"]
    ablation_cfg = config.get("ablation", {})
    regrounding_cfg = ablation_cfg.get("regrounding", {})

    start_time = time.time()
    suffix = Path(input_path).suffix.lower()
    model_stack = f'{config["models"]["active_stack"]["detector"]}+{config["models"]["active_stack"]["segmenter"]}'
    grounding_model = load_grounding_model(grounding_cfg)

    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        image_rgb = read_image_rgb(input_path)
        boxes = predict_boxes(image_rgb, prompt, grounding_cfg, model=grounding_model)
        if not boxes["boxes_xyxy"]:
            raise RuntimeError(f"No detections found for prompt: {prompt}. Try a simpler prompt such as `person` or `dog`.")
        masks = predict_image_masks(image_rgb, boxes["boxes_xyxy"], sam2_cfg)
        merged_mask = _merge_masks(masks, image_rgb.shape[:2])
        overlaid = overlay_mask(image_rgb, merged_mask, alpha=float(runtime_cfg["overlay_alpha"]))
        overlaid = draw_boxes(overlaid, boxes["boxes_xyxy"], boxes["phrases"])
        image_path = output_root / "smoke_image_overlay.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        num_frames = 1
        artifacts = {"image_overlay": str(image_path)}
        input_type = "image"
    else:
        frames = extract_video_frames(
            input_path,
            max_frames=int(runtime_cfg["max_frames"]),
            frame_stride=int(runtime_cfg["frame_stride"]),
        )
        if not frames:
            raise RuntimeError(f"No frames could be extracted from {input_path}")
        if regrounding_cfg.get("enabled", False):
            video_masks = _run_video_with_periodic_regrounding(
                frames=frames,
                prompt=prompt,
                grounding_cfg=grounding_cfg,
                sam2_cfg=sam2_cfg,
                ablation_cfg=ablation_cfg,
                grounding_model=grounding_model,
            )
            frame_boxes = video_masks.get("boxes_per_frame", [])
        else:
            initial_boxes = predict_boxes(frames[0], prompt, grounding_cfg, model=grounding_model)
            if not initial_boxes["boxes_xyxy"]:
                raise RuntimeError(
                    f"No detections found in the first frame for prompt: {prompt}. Try a simpler prompt such as `person`."
                )
            video_masks = propagate_video_masks(frames, initial_boxes["boxes_xyxy"], sam2_cfg)
            frame_boxes = [initial_boxes["boxes_xyxy"][0]] * len(frames)
        overlaid_frames = []
        for frame_index, (frame_rgb, mask) in enumerate(zip(frames, video_masks["masks"])):
            combined = np.asarray(mask).astype(bool)
            frame_overlay = overlay_mask(frame_rgb, combined, alpha=float(runtime_cfg["overlay_alpha"]))
            if frame_index < len(frame_boxes):
                frame_overlay = draw_boxes(frame_overlay, [frame_boxes[frame_index]], [prompt])
            overlaid_frames.append(frame_overlay)
        video_path = save_video(overlaid_frames, output_root / "smoke_video_overlay.mp4")
        num_frames = len(frames)
        artifacts = {
            "video_overlay": str(video_path),
            "video_mode": video_masks["mode"],
        }
        if "fallback_reason" in video_masks:
            artifacts["fallback_reason"] = video_masks["fallback_reason"]
        input_type = "video"

    runtime_sec = round(time.time() - start_time, 3)
    summary = {
        "prompt": prompt,
        "input_path": str(input_path),
        "input_type": input_type,
        "num_frames": num_frames,
        "runtime_sec": runtime_sec,
        "model_stack": model_stack,
        "artifacts": artifacts,
    }
    if input_type == "video" and regrounding_cfg.get("enabled", False):
        summary.update(
            {
                "ablation_variant": str(ablation_cfg.get("variant", "periodic_regrounding")),
                "regrounding_enabled": True,
                "reground_interval": int(video_masks.get("reground_interval", regrounding_cfg.get("interval_frames", 10))),
                "reground_matching": str(video_masks.get("reground_matching", regrounding_cfg.get("matching", "iou"))),
                "min_match_iou": float(video_masks.get("min_match_iou", regrounding_cfg.get("min_match_iou", 0.1))),
                "num_reground_attempts": int(video_masks.get("num_reground_attempts", 0)),
                "num_reground_successes": int(video_masks.get("num_reground_successes", 0)),
                "reground_frames": list(video_masks.get("reground_frames", [])),
            }
        )
    write_json(output_root / "run_summary.json", summary)
    return summary
