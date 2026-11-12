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
