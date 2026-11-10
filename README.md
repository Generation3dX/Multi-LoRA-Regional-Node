# MOSS-Image Regional LoRA

A ComfyUI node that applies several character LoRAs to MOSS-Image, each confined to its own bounding box.

Each LoRA's contribution is added at forward time and multiplied by a mask built from its box. Outside the box the mask is 0. Model weights are never patched, so bf16 and int8_convrot checkpoints behave the same.

Requires ComfyUI ≥ 0.42 with native MOSS-Image support.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. There are no extra dependencies.

## Usage

1. **Create Bounding Boxes**: draw one box per character. Put each character's trigger word in its box description.
2. **MOSS-Image Layout Prompt**: wire in `elements`, and list the character boxes in `character_boxes`. Its output goes to `CLIPTextEncode`.
3. **MOSS-Image Regional LoRA Loader**: add one per character, with these settings:
   - `box`: the box number from the Create Bounding Boxes preview.
   - `lora_name`: picked from `models/loras` and its subfolders.
   - `strength`.
4. **MOSS-Image Apply Regional LoRAs**: wire in `model`, the **same** `elements`, and each Regional LoRA Loader into `regions`. A new `region` socket appears every time you connect one. The output `MODEL` goes to the sampler. The character LoRAs need no core LoRA loader.

Both nodes read the same boxes, so the layout placement and the LoRA scope cannot drift apart.

### Resolution Selector

Link a **Resolution Selector**'s `width`/`height` to Create Bounding Boxes (and to the Layout Prompt and the latent). This pack's frontend script then keeps the canvas at the selector's exact output and rescales the drawn boxes with it. The canvas updates live when you change the aspect ratio or megapixels, and again before every run.

Without the script, Create Bounding Boxes would silently shrink or stretch the boxes. It saves them in pixels of its own size widgets, but the backend divides them by the linked size, which the canvas never sees.

If `width`/`height` are linked to anything other than one Resolution Selector, the run is refused with an error toast. Leaving both unlinked is fine. The browser console logs one `[MOSS bbox sync]` line per bbox node on every run.

| Input | Meaning |
|---|---|
| `seam_feather` | Inward fade at the box edges, as a fraction of the image. 0 is a hard edge. The LoRA never acts outside its box. |
| `lora_on_text` | Advanced input, hidden by default. Also applies every region's LoRA to all prompt tokens. Off by default, because the prompt tokens are shared by all characters; Phase 2 showed it does not help. |

When boxes overlap, the region with the stronger weighted mask owns each token.

The log reports how many layers each LoRA matched. A LoRA that matches nothing raises an error.

Recommended strength: 1.0, the same as for a global LoRA. On Yogi, Base and SFT, 1.0 keeps each identity in its box (see `DESIGN.md` §10). Leave `lora_on_text` off.

## Credits

The masked activation-delta approach comes from the Krea 2 regional multi-LoRA node by **Fedor** (CliffNodes), MIT licence: https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit
