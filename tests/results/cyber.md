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
