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

## Report Tables

### Table 1. Baseline Summary

| subset_size | tag_distribution | baseline_review_counts | interpretation |
| --- | --- | --- | --- |
| 19 | easy=5, occlusion=5, crowded=5, small_object=4 | good_tracking=7, partial_tracking=7, drift=2, wrong_object=1, no_detection=1, fallback=1 | The no-re-grounding baseline is reliable on clean single-target clips but degrades under crowding, occlusion, and small-object settings. |

### Table 2. Ablation Delta Summary

| improved | same | worse | reground_review_counts | interpretation |
| --- | --- | --- | --- | --- |
| 4 | 8 | 7 | partial_tracking=18, fallback=1 | Periodic re-grounding every 10 frames is not a net win; it rescues a few hard failures but destabilizes many clips that were already strong under the baseline. |

## Figure Package

Lock the report/slides figure set to exactly six items so the narrative stays fixed.

- `F1_baseline_success_small_object`: `10360251`, prompt=`car`, label=`good_tracking`. Chosen because it shows the baseline can still succeed on a small-object clip when the target remains visually separable.
- `F2_baseline_success_easy`: `853810`, prompt=`dog`, label=`good_tracking`. Chosen because it is the cleanest single-target success case and makes the baseline strength easy to explain.
- `F3_baseline_failure_wrong_object`: `12699538`, prompt=`person`, label=`wrong_object`. Chosen because it visualizes crowd-driven identity switching clearly.
- `F4_baseline_failure_drift`: `5630823`, prompt=`dog`, label=`drift`. Chosen because it is a compact example of small-object tracking collapsing toward a distractor.
- `F5_ablation_improved`: `12699538`, prompt=`person`, `wrong_object -> partial_tracking`. Chosen because it is the clearest case where periodic re-grounding helps a severe baseline failure.
- `F6_ablation_worse`: `10360251`, prompt=`car`, `good_tracking -> partial_tracking`. Chosen because it cleanly demonstrates the cost of naive periodic re-grounding on a clip that baseline already handled well.

The slide mapping should stay derivative from this report package:

- Slide 1: Table 1 baseline summary
- Slide 2: F1-F4 baseline success/failure examples
- Slide 3: Table 2 ablation delta summary
- Slide 4: F5-F6 ablation improved vs worse examples
- Slide 5: Final discussion takeaway bullets

## Final Discussion

The project now has a coherent result story rather than just a runnable pipeline. The official no-re-grounding baseline establishes that `Grounding DINO + SAM2` is already a credible reference system on the locked subset: it completes all runs, stays on the primary `sam2_video_predictor` path, and produces several genuinely strong qualitative cases on clear single-target clips. That matters because it means later comparisons are not being made against a broken or unstable baseline.

At the same time, the baseline exposes a clear and defensible weakness profile. Performance degrades most often when prompts are underspecified relative to the scene, especially in crowds, partial occlusion, and small-object settings. The failure analysis shows that the core issue is not infrastructure reliability but target identity preservation under ambiguity. This gives the project a clean problem statement for the experimental section: the question is how to recover from drift and wrong-object switches without destabilizing already-good tracks.

The periodic re-grounding ablation answers that question in an informative but negative way. Re-detecting every 10 frames does help a few severe failure clips, especially crowded and occluded cases that were previously labeled `drift`, `wrong_object`, or `fallback`. However, it also downgrades many clips that were already `good_tracking` under the baseline. The resulting `4 improved / 8 same / 7 worse` split is strong evidence that naive periodic re-grounding is too blunt an intervention for this setting.

That negative result is still useful. It narrows the design space and rules out one simplistic fix: periodic refresh on a fixed schedule with a permissive IoU threshold is not the right default policy. The main lesson is that re-grounding must be more selective than the current implementation. If the project were extended, the most justified next step would be a trigger-based or more conservative re-grounding strategy, for example one with stricter matching, quality-based gating, or re-detection only when propagation confidence visibly degrades.

The final project-level conclusion is therefore straightforward. The stronger default result is the no-re-grounding baseline. It is stable enough to serve as the main system and honest enough to reveal real weaknesses. The periodic re-grounding ablation is valuable not because it wins, but because it shows exactly why naive refresh does not automatically solve RVOS failure modes: it can rescue hard failures, but it can also inject unnecessary detector noise and destabilize clips that propagation already handled correctly.
