# Low-Light Enhancement Lab

A web app that runs an uploaded low-light image through 4 models
(Zero-DCE, Zero-DCE++, RetinexNet, MIRNet), scores each result, and
highlights the best one — with PSNR/SSIM when a ground-truth image is
supplied, or a no-reference quality score otherwise.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** — works from any device on your network
(phone, another PC) if you replace `localhost` with your machine's LAN IP.

## Current state

The app is fully runnable **right now**, before your models finish
training. Each model file in `models/` falls back to a classical
enhancement (CLAHE, gamma correction, Retinex-style, or bilateral+CLAHE)
when no trained checkpoint is found, so you can demo the full pipeline
and UI today.

## Plugging in your trained models

Each file in `models/` (`zero_dce.py`, `zero_dce_pp.py`, `retinexnet.py`,
`mirnet.py`) has:

1. A `CHECKPOINT_PATH` constant at the top — point it at your `.pth` file.
2. A `sys.path.append(...)` line — point it at the folder containing that
   model's `model.py` (its architecture definition).
3. A `TODO` marked block in `_try_load()` — match the class names/constructor
   args to your actual repo's code (these differ per model, e.g. RetinexNet
   uses two networks: `DecomNet` + `RelightNet`).

Once the checkpoint path exists and loads without error, the app
automatically switches from the classical fallback to your real model —
no other code changes needed.

## Why there's no PSNR/SSIM by default

PSNR and SSIM are **full-reference** metrics — they need a known-correct
"ground truth" image to compare against. A user uploading an arbitrary
photo has no such reference, so the app can't compute them in that case.

- **Live app / arbitrary uploads**: ranked using `score_no_reference()` in
  `metrics.py` — a composite of brightness balance, contrast, entropy
  (detail), sharpness, and colorfulness.
- **Optional ground-truth upload**: if the user attaches a reference image
  (same scene, ideal exposure), real PSNR/SSIM are computed and used for
  selection instead.
- **Paper results**: run your models offline against paired test sets that
  have real ground truth (LOL-v1 / LOL-v2 test splits) and report PSNR/SSIM
  from that — this is the number reviewers will expect, not the app's
  no-reference score.

## Hybrid framework integration

Once you build the hybrid post-processing pipeline (CLAHE + denoising +
adaptive brightness/contrast), add it as a 5th entry in the `MODELS` dict
in `app.py`, following the same `enhance(image_bgr) -> image_bgr` interface
as the other four.
