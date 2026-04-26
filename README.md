# Text-Prompted Video Object Tracking with Grounding DINO and SAM2

This project explores text-prompted video object tracking using a pipeline built with Grounding DINO and SAM2. The goal is to detect an object from a natural-language prompt, initialize a mask, and then track that object across video frames. I built the project in Python using repository-based scripts, configuration files, lightweight notebooks, and Google Colab for the heavier model stack and inference runs. I also tested different re-grounding strategies to compare how reseeding affected tracking quality on a locked subset of video clips. Along the way, I learned a lot about evaluation design, failure analysis, and the challenge of making computer vision pipelines both accurate and reproducible.

## Highlights

- Uses **Grounding DINO** to localize an object from a text prompt
- Uses **SAM2** to convert detections into masks and propagate them through video
- Separates local development from heavier Colab-based inference workflows
- Supports structured configs, reusable scripts, and reproducible experiment organization
- Includes space for smoke tests, ablations, qualitative outputs, and report assets

## Project Overview

This repository is the source of truth for the project code, configs, and lightweight notebooks.

- MacBook Air M4 is the development environment for code, Git, lightweight data work, visualization, local wiring checks, and report assets.
- Google Colab Pro is the compute environment for installing the heavy model stack, downloading checkpoints, running real video inference, and benchmarking.
- Deliverable 0 is repo wiring only: folder structure, docs, config schema, CLI stubs, notebook scaffolds, and a lightweight local validation script.
- Deliverable 1 is the first full model run: GroundingDINO generates boxes from text prompts, SAM2 converts them to masks, and SAM2 video propagation is the default video path.

The repository is organized so that:

- core logic lives in `src/` and `scripts/`
- notebooks call scripts instead of holding business logic
- checkpoints, raw data, and generated outputs stay outside git

## Demo Video

[Watch the screencast demo with audio explanation]

## Folder Structure
```text
ProjectFinalCS415/
├── README.md
├── requirements.txt
├── .gitignore
├── setup_colab.sh
├── checkpoints/
├── notebooks/
├── configs/
├── data/
├── src/
├── scripts/
├── results/
└── report/
```

- `notebooks/`: thin orchestration notebooks for smoke test, subset eval, and ablations.
- `configs/`: shared YAML config schema for paths, runtime settings, and model-specific parameters.
- `configs/external/`: repo-owned vendor copies of the exact GroundingDINO and SAM2 model configs used by the smoke test.
- `data/`: local-only input staging for raw data, processed data, and very short sample clips.
- `src/`: reusable Python modules for data IO, model wrappers, evaluation, visualization, and utilities.
- `scripts/`: command-line entrypoints used both locally and from Colab notebooks.
- `results/`: qualitative outputs, quantitative summaries, and report figures. These are ignored by git.
- `report/`: outline, references, and exported figures for the write-up.

## Local Workflow
Deliverable 0 should be reproducible without installing GroundingDINO or SAM2.

1. Create and activate a Python environment.
2. Install the pinned base dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run the local wiring check:

```bash
python3 scripts/check_env.py --config configs/base.yaml --output-dir outputs/check_env
```

What this validates:

- core package imports
- config loading and schema merging
- synthetic overlay rendering
- JSON artifact writing

Expected output:

- `outputs/check_env/check_overlay.png`
- `outputs/check_env/check_summary.json`

For local experimentation with very short clips, you can also dry-run the CLI shape:

```bash
python3 scripts/run_custom_video.py \
  --config configs/base.yaml \
  --input_video /path/to/short_clip.mp4 \
  --prompt "person with red backpack" \
  --output_dir outputs/smoke_local \
  --max_frames 24
```

This command becomes fully runnable after the D1 Colab model stack is installed and checkpoints are available.

## Colab Workflow
Colab is treated as a compute runner, not the place where project logic lives.

1. Mount Google Drive.
2. Clone this repository.
3. Run the base setup:

```bash
bash setup_colab.sh
```

4. For Deliverable 1, install the heavy model stack:

```bash
bash setup_colab.sh --with-models
```

This now also downloads the pinned smoke-test checkpoints into Drive if they are missing.

5. Put checkpoints and inputs in Drive, for example:

```text
/content/drive/MyDrive/cv-final-project/
├── checkpoints/
├── inputs/
└── results/
```

6. Open `notebooks/01_smoke_test.ipynb`, set:

- `PROMPT`
- `INPUT_IMAGE`
- `INPUT_VIDEO`
- `CHECKPOINT_DIR`
- `OUTPUT_DIR`

7. Run the notebook cells. The notebook calls repository scripts and should produce:

- one annotated image with box and mask
- one short overlay video
- one `run_summary.json` with prompt, frame count, runtime, model stack, and artifact paths

## Notes
- The D1 baseline is intentionally limited to `GroundingDINO + SAM2`.
- `Florence-2` is scaffolded in configs and notebooks but is not part of the first runnable smoke test.
- The supported D1 video path is SAM2 video propagation. Frame-by-frame inference exists only as an internal fallback when the official video predictor cannot initialize.
