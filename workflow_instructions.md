# Multi-Character Regional LoRA: how to use

Works with **MOSS-Image** and **Krea 2**. The nodes detect the model family by themselves, so there's no switch to set.

## 1. Loaders

Set all three loaders to the same family:

| Loader | MOSS-Image | Krea 2 |
|---|---|---|
| Load Diffusion Model | a `moss` checkpoint | a `krea2` checkpoint (Cyberrealistic, SickOllie) |
| Load CLIP | Qwen 3.5 text encoder, type **`moss_image`** | Qwen3-VL text encoder, type **`krea2`** |
| Load VAE | `flux2-vae` | `qwen_image_vae` |

**Sampler:** for turbo Krea checkpoints use 8 steps, cfg 1, euler / simple. For MOSS, use your usual settings.

## 2. Resolution

Pick the aspect ratio and megapixels on the **Resolution Selector**. Its width and height feed Create Bounding Boxes, Regional Layout Encode and the empty latent. The box canvas resizes itself to match, and boxes you've already drawn are rescaled with it.

## 3. Bounding boxes and character text

In **Create Bounding Boxes**:

1. Draw one box per character, covering the area that character should fill (head to feet if you want a full body).
2. Give each box a description of that character only, e.g. *"a woman with long red hair, green eyes, wearing a black leather jacket"*.
   - Put appearance and clothing here, not in the main prompt.
   - Trigger words aren't needed.
3. Note the **box numbers** shown in the preview. The LoRA loaders refer to them.

Tips:
- Boxes may overlap a little. Where they do, the stronger box wins.
- On **Krea 2**, placement is coarse: far left / left / centre / right / far right, plus "further back" for a box clearly smaller than the tallest one. Spread the boxes across the width for reliable placement.

## 4. Overall prompt

In **Regional Layout Encode**, write the scene in `prompt`: poses, interaction, setting, lighting, camera and style. Describe what the characters are *doing*, not what they look like. Their appearance comes from the box descriptions.

> *Two friends sitting at a café table by a rain-streaked window, laughing, warm evening light, shallow depth of field, 35mm photo.*

- **`character_boxes`** (MOSS only): the numbers of the boxes that are people, e.g. `1, 2`. Krea 2 ignores this field.
- Its `CONDITIONING` output goes to the sampler's **positive** input.

## 5. Regional LoRAs

1. Add one **Regional LoRA Loader** per character:
   - `box`: that character's box number from step 3.
   - `lora_name`: a LoRA for the **same family** as the checkpoint (a MOSS LoRA on MOSS, a Krea 2 LoRA on Krea 2).
   - `strength`: **1.0** is the recommended starting point.
2. Connect each one to a `region` socket on **Apply Regional LoRAs**. A new socket appears each time you connect one.
3. Connect **Apply Regional LoRAs** to the model and to the **same** `elements` output from Create Bounding Boxes.
   - `seam_feather` (default 0.05) softens the box edges. Set it to 0 for a hard edge.

The character LoRAs don't need a normal LoRA loader. Each one only acts inside its own box.

## 6. Global LoRAs (optional)

You can add a normal **LoraLoaderModelOnly** *after* Apply Regional LoRAs for a style or content LoRA that should affect the whole image. Its output then goes to the sampler.

It must match the checkpoint family. A Krea LoRA on a MOSS model, or the reverse, silently does nothing. When that happens, the console shows `lora key not loaded` lines.

## Switching between MOSS and Krea 2

Change all three loaders (step 1), then switch every LoRA to its version for the other family: each Regional LoRA Loader and any global LoRA. Boxes and prompts stay as they are.

## Troubleshooting

- **"matched no layers of this … model" error:** the LoRA is for the other model family.
- **"box N does not exist" error:** a LoRA loader's `box` number is higher than the number of boxes drawn.
- **"supports MOSS-Image and Krea 2 …" error:** a loader is set to the wrong family, or the CLIP type is wrong.
- **Character in the wrong place:** check that the box number in the LoRA loader matches the box number in the preview.
- **Character likeness weak:** raise the strength in steps of 0.1, or make the box bigger. Small boxes give the model fewer pixels to work with.
