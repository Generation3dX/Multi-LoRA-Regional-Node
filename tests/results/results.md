# Phase 2 identity matrix

ArcFace cosine similarity of the face in each box to each character's solo-LoRA reference (mean over seeds [1000, 1001, 1002, 1003]).
`diag` is the mean similarity to the right character, `off` to the wrong one; `margin` = diag - off. `faces` is the number of faces detected in each whole image.

## Yogi V1.5 int8_convrot (transfer: base model unknown)

Contact sheet: `yogi_sheet.jpg`

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.280 | 0.402 | 0.160 | 0.546 | 0.413 | 0.281 | 0.132 | 0 | 2,2,2,2 |
| stacked | 0.757 | 0.379 | 0.692 | 0.348 | 0.553 | 0.536 | 0.017 | 0 | 2,2,2,2 |
| regional | 0.778 | 0.333 | 0.351 | 0.745 | 0.762 | 0.342 | 0.420 | 0 | 2,2,2,2 |
| regional_text | 0.774 | 0.319 | 0.447 | 0.682 | 0.728 | 0.383 | 0.345 | 0 | 2,2,2,2 |

## Base bf16

Contact sheet: `base_sheet.jpg`

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.072 | -0.018 | 0.004 | 0.060 | 0.066 | -0.007 | 0.073 | 0 | 2,2,2,2 |
| stacked | 0.676 | 0.317 | 0.579 | 0.337 | 0.506 | 0.448 | 0.058 | 0 | 2,2,2,2 |
| regional | 0.589 | 0.210 | 0.150 | 0.386 | 0.487 | 0.180 | 0.307 | 0 | 2,2,2,2 |
| regional_text | 0.687 | 0.204 | 0.293 | 0.490 | 0.589 | 0.248 | 0.341 | 0 | 2,2,2,2 |

## SFT bf16 (cross-checkpoint transfer experiment)

Contact sheet: `sft_sheet.jpg`

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.211 | 0.195 | 0.132 | 0.370 | 0.291 | 0.163 | 0.127 | 0 | 2,2,2,2 |
| stacked | 0.755 | 0.382 | 0.649 | 0.369 | 0.562 | 0.516 | 0.046 | 0 | 2,2,2,2 |
| regional | 0.729 | 0.307 | 0.312 | 0.618 | 0.674 | 0.309 | 0.364 | 0 | 2,2,2,2 |
