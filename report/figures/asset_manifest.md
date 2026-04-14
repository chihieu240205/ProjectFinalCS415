# Report Asset Manifest

This file locks the report-first asset package so the same tables and figures can be reused in slides without changing the narrative.

## Tables

| id | title | content | slide_use |
| --- | --- | --- | --- |
| T1 | Baseline Summary | Subset size, tag distribution, baseline review counts, one-sentence interpretation | Slide 1 |
| T2 | Ablation Delta Summary | Improved/same/worse counts, re-grounding review counts, one-sentence interpretation | Slide 3 |

## Figures

| id | clip_id | prompt | label_or_delta | why_this_case | slide_use |
| --- | --- | --- | --- | --- | --- |
| F1_baseline_success_small_object | 10360251 | car | good_tracking | Shows that the baseline can still succeed on a small-object case when the target remains visually separable. | Slide 2 |
| F2_baseline_success_easy | 853810 | dog | good_tracking | Clean single-target success; easy to explain the baseline's strongest behavior. | Slide 2 |
| F3_baseline_failure_wrong_object | 12699538 | person | wrong_object | Clear crowd-driven identity switch in a dense overlapping scene. | Slide 2 |
| F4_baseline_failure_drift | 5630823 | dog | drift | Compact example of small-object tracking collapsing toward a distractor. | Slide 2 |
| F5_ablation_improved | 12699538 | person | wrong_object -> partial_tracking | Best example that periodic re-grounding can help a severe baseline failure. | Slide 4 |
| F6_ablation_worse | 10360251 | car | good_tracking -> partial_tracking | Best example that naive periodic re-grounding can hurt a clip that baseline already handled well. | Slide 4 |

## Slide Package

- Slide 1: T1 baseline summary
- Slide 2: F1-F4 baseline success/failure panel
- Slide 3: T2 ablation delta summary
- Slide 4: F5-F6 ablation improved-vs-worse panel
- Slide 5: Final discussion takeaway bullets

## Follow-up Recommendation

Do not add another experiment in the report-first closeout phase. If there is time after the write-up is stable, the only justified next experiment is a stricter or trigger-based re-grounding policy rather than another naive periodic schedule.
