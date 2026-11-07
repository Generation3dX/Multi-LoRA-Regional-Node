# MOSS-Image Regional LoRA

A ComfyUI node that applies several character LoRAs to MOSS-Image, each confined to its own bounding box.

Each LoRA's contribution is added at forward time and multiplied by a mask built from its box. Outside the box the mask is 0. Model weights are never patched, so bf16 and int8_convrot checkpoints behave the same.

Requires ComfyUI ≥ 0.42 with native MOSS-Image support.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. There are no extra dependencies.

## Usage

1. **Create Bounding Boxes**: draw one box per character. Put each character's trigger word in its box description.
2. **MOSS-Image Layout Prompt**: wire in `elements`, and list the character boxes in `character_boxes`. Its output goes to `CLIPTextEncode`.
3. **MOSS-Image Region LoRA**: add one per character, with these settings:
   - `box`: the box number from the Create Bounding Boxes preview.
   - `lora_name`: picked from `models/loras` and its subfolders.
   - `strength`.
4. **MOSS-Image Regional LoRA**: wire in `model`, the **same** `elements`, and each Region LoRA into `regions`. A new `region` socket appears every time you connect one. The output `MODEL` goes to the sampler. No separate LoRA loader is needed.

Both nodes read the same boxes, so the layout placement and the LoRA scope cannot drift apart.

| Input | Meaning |
|---|---|
| `seam_feather` | Inward fade at the box edges, as a fraction of the image. 0 is a hard edge. The LoRA never acts outside its box. |
| `lora_on_text` | Also applies every region's LoRA to all prompt tokens. Off by default, because the prompt tokens are shared by all characters. |

When boxes overlap, the region with the stronger weighted mask owns each token.

The log reports how many layers each LoRA matched. A LoRA that matches nothing raises an error.

Recommended strengths: to be measured in Phase 2.

## Credits

The masked activation-delta approach comes from the Krea 2 regional multi-LoRA node by **Fedor** (CliffNodes), MIT licence: https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit
