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

- `2026-04-01 | hieu_con_quang_ninh | prompt=person | video_mode=sam2_video_predictor | review=good tracking | pass`
- `2026-04-01 | khanh_sky_uong_nuoc | prompt=fridge | video_mode=sam2_video_predictor | review=wrong object | pass`

### Fixed Subset Benchmark Seed

- `2026-04-13 | subset_manifest.csv | selected=20 | completed=20 | primary_path=sam2_video_predictor | pass`
- Tag distribution for the fixed subset seed: `easy=6`, `occlusion=5`, `crowded=5`, `small_object=4`.
- All completed subset runs used the primary `sam2_video_predictor` video path and wrote overlay artifacts under `results/quantitative/subset_eval/`.
- Current prompt source in the subset manifest is the `notes` column, which is being used as the inference prompt for each selected clip.
- One duplicate `clip_id` (`853810`) exists in the current manifest/output summary; this was left as-is for now and should be treated as a reporting caveat rather than a corrected benchmark row.

### Official Baseline Run

- `2026-04-14 | grounding_dino+sam2_no_regrounding | selected=19 | completed=19 | primary_path=sam2_video_predictor | pass`
- Official baseline tag distribution: `easy=5`, `occlusion=5`, `crowded=5`, `small_object=4`.
- Local reviewed baseline counts: `good_tracking=7`, `partial_tracking=7`, `drift=2`, `wrong_object=1`, `no_detection=1`, `fallback=1`.
- Selected 5 success cases: `10360251`, `853810`, `16436839`, `5730870`, `6326811`.
- Selected 5 failure cases: `11998127`, `12699538`, `5220726`, `5630823`, `6664239`.
- Final export of sample artifacts on Drive was not cleanly finalized in Colab, but the reviewed baseline table and 5/5 success-failure split were completed locally.

### Notes for Write-up

- Primary runtime stack: `grounding_dino+sam2`
- Custom-video qualitative validation succeeded on two user videos.
- Both successful custom runs stayed on the primary SAM2 video path without fallback.
- Observed failure mode on `khanh_sky_uong_nuoc`: when the fridge is behind the person, detection is no longer accurate and tends toward the wrong object category.
- First subset benchmark seed completed end-to-end on 20 selected rows with balanced qualitative tags across easy, occlusion, crowded, and small-object clips.
- The current subset run is suitable as an experiment scaffold, but any formal quantitative table should note the existing duplicate `clip_id` in the manifest summary.
- Official qualitative baseline was completed on a locked 19-clip subset using `Grounding DINO + SAM2` without re-grounding.
- All 19 official baseline runs stayed on `sam2_video_predictor`, so the main qualitative failure modes are now object ambiguity, drift, no-detection, and occlusion-related degradation rather than pipeline fallback.

## Baseline Results

The official baseline uses `Grounding DINO + SAM2` without any re-grounding step on a fixed 19-clip qualitative subset. The subset is balanced across four coarse difficulty tags: `easy=5`, `occlusion=5`, `crowded=5`, and `small_object=4`. All 19 runs completed successfully and stayed on the primary `sam2_video_predictor` path, which means the baseline is now stable enough to discuss model behavior rather than debugging the runtime stack.

At the qualitative review level, the baseline breaks down into `7 good_tracking`, `7 partial_tracking`, `2 drift`, `1 wrong_object`, `1 no_detection`, and `1 fallback`. This distribution suggests that the main bottleneck is not catastrophic pipeline failure but degraded tracking quality under harder scenes. In other words, the baseline is strong enough on clean targets to serve as a reference point, but it is not yet robust to dense distractors, severe occlusion, and very small targets.

Representative success cases for the write-up:

- `10360251` (`small_object`, prompt=`car`): tracks small cars stably in a wide scene.
- `853810` (`easy`, prompt=`dog`): tracks a single target with low ambiguity.
- `16436839` (`easy`, prompt=`bicycle`): handles a dominant foreground target cleanly.
- `5730870` (`easy`, prompt=`dog`): near-perfect tracking on a clear single target.
- `6326811` (`easy`, prompt=`backpack`): stable tracking in portrait framing with a clean target.

Representative failure cases for the write-up:

- `11998127` (`crowded`, prompt=`person`): `drift` caused by many visually similar distractors.
- `12699538` (`crowded`, prompt=`person`): `wrong_object` when the tracker switches targets in a dense overlapping crowd.
- `5220726` (`occlusion`, prompt=`bicycle`): `fallback`-like failure under heavy mutual occlusion.
- `5630823` (`small_object`, prompt=`dog`): `drift` toward a moving distractor while tracking a small target.
- `6664239` (`crowded`, prompt=`person`): `no_detection` in a dense night scene with heavy motion and distractors.

## Ablation Results

The main ablation adds naive periodic re-grounding every 10 frames on top of the same `Grounding DINO + SAM2` stack and evaluates it on the same locked 19-clip subset. This keeps the comparison clean: the prompt set, clip set, checkpoints, and qualitative review procedure are unchanged, so any difference can be attributed to the re-grounding policy itself rather than data drift.

At the clip level, the re-grounding variant produced `4 improved`, `8 same`, and `7 worse` outcomes relative to the no-re-grounding baseline. The reviewed re-grounding labels collapsed to `18 partial_tracking` and `1 fallback`, which is a strong sign that periodic reseeding reduced a few severe failures but also destabilized many clips that were previously solid. In effect, the ablation traded several hard failures for a broad shift toward middling tracking quality.

The improvement cases are still meaningful. Re-grounding helped several baseline failure clips move upward into `partial_tracking`, including `11998127` (`drift -> partial_tracking`), `12699538` (`wrong_object -> partial_tracking`), `5220726` (`fallback -> partial_tracking`), and `5630823` (`drift -> partial_tracking`). These are exactly the kinds of crowded, occluded, and small-object scenes where periodic re-detection can plausibly recover from accumulated tracking drift.

However, the negative side of the ablation is stronger than the positive side in this first version. Several clips that were previously labeled `good_tracking` degraded to `partial_tracking`, including `10360251`, `853810`, `16436839`, `5730870`, `6326811`, `7825225`, and `9910242`. This pattern suggests that the current re-grounding policy is too aggressive: detector refreshes are being accepted often enough to introduce extra target jitter or unnecessary resets even when propagation is already stable.

The runtime pattern supports that interpretation. Most clips show `4` re-ground attempts and often `4` successes, meaning the detector is finding an acceptable box at nearly every scheduled refresh. With `min_match_iou = 0.1`, the current acceptance rule is probably too permissive. The ablation therefore produces a useful negative result: **naive periodic re-grounding every 10 frames does not outperform the baseline overall**, even though it can rescue a subset of difficult clips.

## Failure Analysis

The 10 reviewed example cases show that the dominant failure modes are semantic ambiguity and target persistence, not infrastructure instability. Because all official baseline runs stayed on `sam2_video_predictor`, the failure analysis can focus on scene difficulty instead of implementation bugs.

The first recurring pattern is **crowd-driven ambiguity**. In clips such as `11998127`, `12699538`, and `6664239`, the prompt `person` is semantically correct but underspecified relative to the scene. When many people are present with similar appearance and motion, the detector can either drift to a neighboring instance, switch identities entirely, or fail to localize a confident starting target. This is the clearest argument for later improvements such as re-grounding or stronger prompt conditioning.

The second pattern is **occlusion-driven instability**. Clips such as `4992551`, `4992557`, `5021553`, and especially `5220726` show that once the target is partially hidden by other actors or framing, the baseline often degrades from `good_tracking` to `partial_tracking`, and in the worst case loses the target entirely. Even when the system does not fully collapse, mask quality and identity consistency degrade noticeably during crossings and mutual overlap.

The third pattern is **small-object fragility**. The contrast between `10360251` and `11073730` for `car`, and between `853810`/`5730870` and `5630823`/`6413967` for `dog`, suggests that small targets are not uniformly hard; they become hard when small scale is combined with distractors or wide framing. The system can track a small object when it remains visually isolated, but performance deteriorates quickly when the same object competes with clutter or secondary motion cues.

The practical conclusion after the ablation is more specific. The baseline is already good enough to establish a credible first result section, but the first periodic re-grounding variant is not a net improvement. Instead of cleanly fixing hard cases, it often rescues severe failures only by pulling many easy or already-stable clips down to `partial_tracking`. That shifts the next experimental direction: rather than adding more naive periodic schedules, the more promising follow-up is selective or trigger-based re-grounding with a stricter matching rule.
