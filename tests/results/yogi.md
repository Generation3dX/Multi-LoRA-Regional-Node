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
