# AnoLLM MPS reproduction

This repository records my Apple Silicon adaptation of [AnoLLM](https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection), a language-model approach to tabular anomaly detection.

## Scope

- Added an MPS/CPU device path for training and inference
- Disabled bfloat16 on MPS
- Preserved the original tabular-to-text preprocessing and permutation-based anomaly scoring
- Ran a smoke-test experiment with SmolLM-135M on the seismic dataset

The smoke test completed and produced model and score artifacts locally. Those artifacts are not checked in because they are generated files; this repository focuses on the adapted code path.

## Run

Prepare a supported dataset under `data/`, then run:

```bash
pip install -r requirements.txt
python train_anollm_mps.py \
  --dataset seismic \
  --model smol \
  --max_steps 2000

python evaluate_anollm_mps.py \
  --dataset seismic \
  --model smol \
  --split_idx 0
```

On an Apple Silicon machine the scripts select `mps`; otherwise they fall back to CPU.

## Limitations

- This is a single-machine adaptation, not a reproduction of the paper's full benchmark matrix.
- The dataset, W&B logs, trained `.pt` weights, and generated NumPy score arrays are excluded.
- Exact results depend on the selected split, model download, and dependency versions.

## Attribution

AnoLLM and the base implementation are by Liu et al. and Amazon Science. The original repository's Apache-2.0 license, notice, and third-party notices are retained. My contribution here is limited to the local MPS reproduction path and its documentation.
