# Report Outline

- Problem framing
- Baseline approach
- Smoke test
- Quantitative evaluation
- Failure analysis
- Ablation study
- Final discussion

## Validation Log

### D1 Runtime Validation

- D1 smoke test passed on Colab with the primary SAM2 propagation path.
- Image smoke test produced an annotated image and valid `run_summary.json`.
- Video smoke test produced an overlay MP4 and valid `run_summary.json`.
- Confirmed `artifacts.video_mode == "sam2_video_predictor"` on the happy path.

### Stage 1 Custom Videos

- `2026-04-01 | hieu_con_quang_ninh | prompt=person | video_mode=sam2_video_predictor | pass`
- `2026-04-01 | khanh_sky_uong_nuoc | prompt=fridge | video_mode=sam2_video_predictor | pass`

### Notes for Write-up

- Primary runtime stack: `grounding_dino+sam2`
- Custom-video qualitative validation succeeded on two user videos.
- Both successful custom runs stayed on the primary SAM2 video path without fallback.
