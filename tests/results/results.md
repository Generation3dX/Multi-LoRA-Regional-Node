# Identity matrix

ArcFace cosine similarity to each character's solo global-LoRA reference, mean over seeds [1000, 1001, 1002, 1003].

## MOSS Yogi V1.5 int8_convrot (transfer: base model unknown)

Contact sheet: `yogi_sheet.jpg`

Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):

| config | vs joni | vs sansa | missing faces | faces |
|---|---|---|---|---|
| ref_joni | 0.903 | 0.296 | 0 | 1,1,1,1 |
| ref_sansa | 0.290 | 0.885 | 0 | 1,1,1,1 |
| solo_regional_joni | 0.867 | 0.307 | 0 | 1,1,1,1 |
| solo_regional_sansa | 0.264 | 0.873 | 0 | 1,1,1,1 |

Two people, similarity of the face in each box to each reference; `margin` = diag - off:

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.280 | 0.402 | 0.160 | 0.546 | 0.413 | 0.281 | 0.132 | 0 | 2,2,2,2 |
| stacked | 0.757 | 0.379 | 0.692 | 0.348 | 0.553 | 0.536 | 0.017 | 0 | 2,2,2,2 |
| regional | 0.767 | 0.340 | 0.367 | 0.737 | 0.752 | 0.353 | 0.399 | 0 | 2,2,2,2 |
| regional_text | 0.787 | 0.299 | 0.430 | 0.684 | 0.736 | 0.365 | 0.371 | 0 | 2,2,2,2 |

## MOSS Base bf16

Contact sheet: `base_sheet.jpg`

Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):

| config | vs joni | vs sansa | missing faces | faces |
|---|---|---|---|---|
| ref_joni | 0.827 | 0.121 | 0 | 1,1,1,1 |
| ref_sansa | 0.105 | 0.718 | 0 | 1,1,1,1 |

Two people, similarity of the face in each box to each reference; `margin` = diag - off:

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.072 | -0.018 | 0.004 | 0.060 | 0.066 | -0.007 | 0.073 | 0 | 2,2,2,2 |
| stacked | 0.676 | 0.317 | 0.579 | 0.337 | 0.506 | 0.448 | 0.058 | 0 | 2,2,2,2 |
| regional | 0.560 | 0.206 | 0.179 | 0.472 | 0.516 | 0.192 | 0.324 | 0 | 2,2,2,2 |
| regional_text | 0.677 | 0.205 | 0.249 | 0.473 | 0.575 | 0.227 | 0.348 | 0 | 2,2,2,2 |

## MOSS SFT bf16 (cross-checkpoint transfer experiment)

Contact sheet: `sft_sheet.jpg`

Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):

| config | vs joni | vs sansa | missing faces | faces |
|---|---|---|---|---|
| ref_joni | 0.904 | 0.237 | 0 | 1,1,1,1 |
| ref_sansa | 0.221 | 0.844 | 0 | 1,1,1,1 |

Two people, similarity of the face in each box to each reference; `margin` = diag - off:

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.211 | 0.195 | 0.132 | 0.370 | 0.291 | 0.163 | 0.127 | 0 | 2,2,2,2 |
| stacked | 0.755 | 0.382 | 0.649 | 0.369 | 0.562 | 0.516 | 0.046 | 0 | 2,2,2,2 |
| regional | 0.697 | 0.334 | 0.283 | 0.660 | 0.679 | 0.309 | 0.370 | 0 | 2,2,2,2 |

## Krea 2 Cyberrealistic v4.0 int8 (finetune)

Contact sheet: `cyber_sheet.jpg`

Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):

| config | vs joni | vs sansa | missing faces | faces |
|---|---|---|---|---|
| ref_joni | 0.935 | 0.121 | 0 | 1,1,1,1 |
| ref_sansa | 0.113 | 0.872 | 0 | 1,1,1,1 |
| solo_regional_joni | 0.894 | 0.163 | 0 | 1,1,1,1 |
| solo_regional_sansa | 0.191 | 0.790 | 0 | 1,1,1,1 |

Two people, similarity of the face in each box to each reference; `margin` = diag - off:

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.246 | 0.305 | 0.112 | 0.470 | 0.358 | 0.209 | 0.149 | 0 | 2,2,2,2 |
| stacked | 0.584 | 0.285 | 0.545 | 0.280 | 0.432 | 0.415 | 0.017 | 0 | 2,2,2,2 |
| regional | 0.726 | 0.192 | 0.202 | 0.671 | 0.698 | 0.197 | 0.501 | 0 | 2,2,2,2 |
| regional_text | 0.777 | 0.158 | 0.268 | 0.642 | 0.710 | 0.213 | 0.497 | 0 | 2,2,2,2 |

## Krea 2 SickOllie int8_convrot (finetune)

Contact sheet: `ollie_sheet.jpg`

Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):

| config | vs joni | vs sansa | missing faces | faces |
|---|---|---|---|---|
| ref_joni | 0.926 | 0.109 | 0 | 1,1,1,1 |
| ref_sansa | 0.106 | 0.901 | 0 | 1,1,1,1 |
| solo_regional_joni | 0.882 | 0.167 | 0 | 1,1,1,1 |
| solo_regional_sansa | 0.112 | 0.840 | 0 | 1,1,1,1 |

Two people, similarity of the face in each box to each reference; `margin` = diag - off:

| config | box 1 vs joni | box 1 vs sansa | box 2 vs joni | box 2 vs sansa | diag | off | margin | missing faces | faces |
|---|---|---|---|---|---|---|---|---|---|
| no_lora | 0.209 | 0.400 | 0.088 | 0.507 | 0.358 | 0.244 | 0.114 | 0 | 2,2,2,2 |
| stacked | 0.497 | 0.234 | 0.502 | 0.269 | 0.383 | 0.368 | 0.015 | 0 | 2,2,2,2 |
| regional | 0.710 | 0.214 | 0.131 | 0.694 | 0.702 | 0.173 | 0.529 | 0 | 2,2,2,2 |
| regional_text | 0.795 | 0.186 | 0.225 | 0.650 | 0.722 | 0.205 | 0.517 | 0 | 2,2,2,2 |
