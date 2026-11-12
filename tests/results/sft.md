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
