# Multi LoRA Regional Node

ComfyUI nodes that apply several character LoRAs to **MOSS-Image** or **Krea 2**, each confined to its own bounding box.

Each LoRA's contribution is added at forward time and multiplied by a mask built from its box. Outside the box the mask is 0. Model weights are never patched, so bf16, fp8 and int8 checkpoints behave the same.

The model family is detected from the loaded model and text encoder; there is no selector. To switch between MOSS-Image and Krea 2, change the three loaders (UNet, CLIP with type `moss_image` or `krea2`, VAE). The prompt and LoRA nodes stay the same.

Requires ComfyUI ≥ 0.42 with native MOSS-Image and Krea 2 support.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. There are no extra dependencies.

## Usage

1. **Create Bounding Boxes**: draw one box per character and describe the character in the box description.
2. **Regional Layout Encode**: wire in `clip`, `elements` and the image `width`/`height`, and write the overall prompt. Its `CONDITIONING` goes to the sampler's positive input. It writes the layout in the form the loaded text encoder's model understands:
   - MOSS-Image: the layout JSON of core's MOSS-Image Layout Prompt. List the character boxes in `character_boxes`.
   - Krea 2: the box descriptions as position and size words, listed left to right ("On the far left, smaller and further back: …"). Krea 2 ignores box coordinates in a caption but follows these words, so placement is coarse: left, centre, right, and further back for clearly smaller boxes.
3. **Regional LoRA Loader**: add one per character, with these settings:
   - `box`: the box number from the Create Bounding Boxes preview.
   - `lora_name`: picked from `models/loras` and its subfolders.
   - `strength`.
4. **Apply Regional LoRAs**: wire in `model`, the **same** `elements`, and each Regional LoRA Loader into `regions`. A new `region` socket appears every time you connect one. The output `MODEL` goes to the sampler. The character LoRAs need no core LoRA loader.

Both nodes read the same boxes, so the layout placement and the LoRA scope cannot drift apart. On MOSS-Image, core's MOSS-Image Layout Prompt with `CLIPTextEncode` still works in place of Regional Layout Encode.

### Resolution Selector

Link a **Resolution Selector**'s `width`/`height` to Create Bounding Boxes (and to Regional Layout Encode and the latent). This pack's frontend script then keeps the canvas at the selector's exact output and rescales the drawn boxes with it. The canvas updates live when you change the aspect ratio or megapixels, and again before every run.

Without the script, Create Bounding Boxes would silently shrink or stretch the boxes. It saves them in pixels of its own size widgets, but the backend divides them by the linked size, which the canvas never sees.

If `width`/`height` are linked to anything other than one Resolution Selector, the run is refused with an error toast. Leaving both unlinked is fine. The browser console logs one `[MOSS bbox sync]` line per bbox node on every run.

### Apply Regional LoRAs inputs

| Input | Meaning |
|---|---|
| `seam_feather` | Inward fade at the box edges, as a fraction of the image. 0 is a hard edge. The LoRA never acts outside its box. |
| `lora_on_text` | Advanced input, hidden by default. Also applies every region's LoRA to all prompt tokens. Off by default, because the prompt tokens are shared by all characters; Phase 2 showed it does not help. |

When boxes overlap, the region with the stronger weighted mask owns each token.

The log reports how many layers each LoRA matched. A LoRA that matches nothing raises an error. On Krea 2, a LoRA's text-only `txtfusion` layers are not applied (logged), because every character shares the prompt.

Recommended strength: 1.0, the same as for a global LoRA, on both families (see `DESIGN.md` §10 and §11.5). Leave `lora_on_text` off.

## Tests

Run from the ComfyUI root with its venv. The test files name the models and LoRAs they use; edit them to match yours.

- `venv\Scripts\python.exe custom_nodes\MOSS-Image-Multi-Character\tests\forward_check.py <diffusion model> <lora> [--fp32]`: single-forward checks against core LoRA patching.
- `venv\Scripts\python.exe custom_nodes\MOSS-Image-Multi-Character\tests\matrix.py [--server URL] [--seeds 4] [--only yogi,cyber]`: identity-separation matrix through a running ComfyUI. Results go to `tests/results/`.
- `node custom_nodes\MOSS-Image-Multi-Character\tests\bbox_sync_e2e.mjs [--server URL] [--vue]`: end-to-end check of the Resolution Selector sync in headless Chrome, through a running ComfyUI.

## Credits

The masked activation-delta approach comes from the Krea 2 regional multi-LoRA node by **Fedor** (CliffNodes), MIT licence: https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit
