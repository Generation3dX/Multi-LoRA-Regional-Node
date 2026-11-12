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
