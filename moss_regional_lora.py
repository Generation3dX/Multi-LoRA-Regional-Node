"""Regional multi-LoRA for MOSS-Image and Krea 2.

Each region's LoRA delta is added at forward time, multiplied by a token mask
built from its Create Bounding Boxes box, so a LoRA cannot act outside its box.
Model weights are never patched, so bf16 and quantized checkpoints behave the same.
"""

import logging

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.patcher_extension
import comfy.text_encoders.krea2
import comfy.text_encoders.moss_image
import comfy.utils
import comfy.weight_adapter
import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_moss_image import MOSSImageLayoutPrompt

WRAPPER_KEY = "moss_regional_lora"
RegionLoRA = io.Custom("MOSS_REGION_LORA")
# adapters with a purely additive forward delta h(x); OFT/BOFT transform the output instead
ADDITIVE_ADAPTERS = (comfy.weight_adapter.LoRAAdapter, comfy.weight_adapter.LoHaAdapter, comfy.weight_adapter.LoKrAdapter)
# per model: latent pixels per token side, whether the text precedes the image in the sequence, and the transformer
# blocks a LoRA is applied to regionally. Layers outside them see no image tokens (Krea 2 txtfusion) and are skipped.
FAMILIES = {
    comfy.model_base.MOSSImage: (1, False, "layers."),  # [image | text]
    comfy.model_base.Krea2: (2, True, "blocks."),  # [text | image | reference images]
}


def box_token_mask(bbox, rows, cols, h, w, feather, device):
    # bbox is Create Bounding Boxes' [y0, x0, y1, x1] on a 0-1000 scale of the h x w token extent; the grid of
    # rows x cols tokens can be larger when the model pads the latent up to whole patches
    y0, x0, y1, x1 = (v / 1000 for v in bbox)

    def ramp(n, extent, a, b):
        # smoothstep that only feathers inward: 0 at and outside the edge, so the LoRA never reaches past its box.
        # An edge on the canvas border has no neighbour to fade towards; it covers the padded tokens past it too.
        width = max(1e-3, min(feather * extent, 0.5 * (b - a) * extent))
        t = torch.arange(n, device=device, dtype=torch.float32) + 0.5
        lo = -n if a <= 0 else a * extent
        hi = 2 * n if b >= 1 else b * extent
        t = (torch.minimum(t - lo, hi - t) / width).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    return (ramp(rows, h, y0, y1)[:, None] * ramp(cols, w, x0, x1)[None, :]).flatten()


def make_hook(entries, masks, start):
    length = masks[0].shape[-2]

    def hook(module, args, out):
        # only the masked token span is computed; tokens outside it get no LoRA
        x = args[0].narrow(-2, start, length)
        span = out.narrow(-2, start, length)
        for region, adapter, part in entries:
            delta = adapter.h(x, span) * masks[region]
            if part is None:
                span.add_(delta)
            else:
                span.narrow(-1, part[0], part[1]).add_(delta)
    return hook


class RegionalLoRAWrapper:
    def __init__(self, boxes, strengths, targets, feather, lora_on_text, patch, text_first, device, dtype):
        self.boxes = boxes
        self.strengths = strengths
        self.targets = targets  # {module name: [(region, adapter, (offset, size) or None)]}
        self.feather = feather
        self.lora_on_text = lora_on_text
        self.patch = patch
        self.text_first = text_first
        self.device = device
        self.dtype = dtype

    def outer_sample(self, executor, *args, **kwargs):
        # LoRA weights live on the GPU only for one sampling call. Moving them before the sampler loads the model lets
        # the model loader see the memory as used, and dropping them afterwards frees it for the VAE decode.
        adapters = [adapter for entries in self.targets.values() for _, adapter, _ in entries]
        stored = [adapter.weights for adapter in adapters]
        try:
            for adapter in adapters:
                adapter.weights = tuple(w.to(device=self.device, dtype=self.dtype) if torch.is_tensor(w) else w for w in adapter.weights)
            return executor(*args, **kwargs)
        finally:
            for adapter, weights in zip(adapters, stored):
                adapter.weights = weights

    def masks(self, height, width, text_len, device, dtype):
        """Returns the first masked sequence position and one mask per region over the masked span."""
        h, w = height / self.patch, width / self.patch
        rows, cols = -(-height // self.patch), -(-width // self.patch)
        masks = torch.stack([box_token_mask(b, rows, cols, h, w, self.feather, device) for b in self.boxes])
        # where boxes overlap, the region with the stronger weighted mask owns the token instead of the identities summing
        weighted = torch.stack([m * abs(s) for m, s in zip(masks, self.strengths)])
        masks = masks * (weighted >= weighted.amax(dim=0))
        start = text_len if self.text_first else 0
        if self.lora_on_text:
            text = masks.new_ones((masks.shape[0], text_len))
            masks = torch.cat((text, masks) if self.text_first else (masks, text), dim=1)
            start = 0
        return start, masks.to(dtype).unsqueeze(-1).unsqueeze(1).unbind(0)

    def __call__(self, executor, x, timestep, context, *args, **kwargs):
        start, masks = self.masks(x.shape[-2], x.shape[-1], context.shape[1], x.device, x.dtype)
        dm = executor.class_obj
        handles = []
        forced = []
        try:
            for name, entries in self.targets.items():
                module = dm.get_submodule(name)
                handles.append(module.register_forward_hook(make_hook(entries, masks, start)))
                # linear_input_act fuses the input activation into quantized GEMMs without calling the module (MOSS ffn_down on int8)
                forced.append((module, module.comfy_force_forward))
                module.comfy_force_forward = True
            return executor(x, timestep, context, *args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            for module, value in forced:
                module.comfy_force_forward = value


def load_region_lora(model, key_map, lora_name, strength, region, targets, blocks):
    lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
    patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    matched = 0
    outside = 0
    skipped = []
    for target, patch in patches.items():
        if not isinstance(patch, ADDITIVE_ADAPTERS):
            skipped.append(target)
            continue
        weight_key, part = (target[0], target[1]) if isinstance(target, tuple) else (target, None)
        if part is not None:
            dim, offset, size = part[:3]
            if dim != 0:
                skipped.append(target)
                continue
            part = (offset, size)
        name = weight_key[len("diffusion_model."):-len(".weight")]
        if not name.startswith(blocks):
            outside += 1
            continue
        model.model.diffusion_model.get_submodule(name)
        patch.multiplier = strength
        targets.setdefault(name, []).append((region, patch, part))
        matched += 1
    logging.info("Regional LoRA: region %d '%s' matched %d modules", region, lora_name, matched)
    if outside:
        logging.info("Regional LoRA: '%s' has %d layers outside the image blocks (text-only or global), not applied", lora_name, outside)
    if skipped:
        logging.warning("Regional LoRA: '%s' has %d patches that cannot be applied regionally and are ignored: %s", lora_name, len(skipped), skipped[:5])
    if matched == 0:
        raise ValueError("Regional LoRA: '{}' matched no layers of this {} model.".format(lora_name, type(model.model).__name__))


def krea2_layout_caption(prompt, elements):
    # Krea 2 ignores box coordinates in the caption but follows position and size words (DESIGN.md §11.2). Subjects are
    # listed left to right, the order Krea places them in when nothing else decides.
    elements = sorted(elements, key=lambda el: el["bbox"][1] + el["bbox"][3])
    tallest = max((el["bbox"][2] - el["bbox"][0] for el in elements), default=0)
    lines = []
    for el in elements:
        y0, x0, y1, x1 = el["bbox"]
        cx = (x0 + x1) / 2000
        where = "On the far left" if cx < 0.2 else "On the left" if cx < 0.4 else "In the centre" if cx < 0.6 else "On the right" if cx < 0.8 else "On the far right"
        depth = ", smaller and further back" if y1 - y0 < 0.6 * tallest else ""
        desc = el.get("desc", "").strip()
        if el.get("type") == "text":
            desc = '{}, the text "{}"'.format(desc, el.get("text", ""))
        lines.append("{}{}: {}.".format(where, depth, desc))
    return "\n\n".join([prompt.strip(), "\n".join(lines)]) if lines else prompt


class MOSSImageRegionLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageRegionLoRA",
            display_name="Regional LoRA Loader",
            category="model/conditioning/regional_lora",
            description="One LoRA and the box it is confined to, for Apply Regional LoRAs.",
            inputs=[
                io.Int.Input("box", default=1, min=1, max=100, tooltip="Number of the box, as tagged in the Create Bounding Boxes preview."),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[RegionLoRA.Output(display_name="region")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, box, lora_name, strength) -> io.NodeOutput:
        return io.NodeOutput({"box": box, "lora_name": lora_name, "strength": strength})


class MOSSImageRegionalLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageRegionalLoRA",
            display_name="Apply Regional LoRAs",
            category="model/conditioning/regional_lora",
            description="Applies each LoRA only inside its bounding box, on MOSS-Image and Krea 2. Use the same Create Bounding Boxes elements for Regional Layout Encode.",
            inputs=[
                io.Model.Input("model"),
                io.Array.Input("elements", tooltip="Prompt elements from the node Create Bounding Boxes."),
                io.Autogrow.Input("regions", template=io.Autogrow.TemplatePrefix(
                    RegionLoRA.Input("region", tooltip="A Regional LoRA Loader: one LoRA applied inside one box."), prefix="region", min=1, max=20)),
                io.Float.Input("seam_feather", default=0.05, min=0.0, max=0.5, step=0.01, tooltip="Inward fade at the box edges as a fraction of the image. 0 is a hard edge."),
                io.Boolean.Input("lora_on_text", default=False, tooltip="Also apply every region's LoRA to all text tokens. Off keeps the LoRAs off the shared prompt tokens.", advanced=True),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, elements, regions, seam_feather, lora_on_text) -> io.NodeOutput:
        family = next((f for c, f in FAMILIES.items() if isinstance(model.model, c)), None)
        if family is None:
            raise ValueError("Apply Regional LoRAs supports MOSS-Image and Krea 2 models, not {}.".format(type(model.model).__name__))
        patch, text_first, blocks = family
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        boxes, strengths, targets = [], [], {}
        for name in sorted(regions, key=lambda name: int(name.removeprefix("region"))):
            row = regions[name]
            if row is None or row["strength"] == 0:
                continue
            box = row["box"]
            if not 1 <= box <= len(elements):
                raise ValueError("Regional LoRA: box {} does not exist, there are {} boxes.".format(box, len(elements)))
            load_region_lora(model, key_map, row["lora_name"], row["strength"], len(boxes), targets, blocks)
            boxes.append(elements[box - 1]["bbox"])
            strengths.append(row["strength"])
        if not boxes:
            return io.NodeOutput(model)
        wrapper = RegionalLoRAWrapper(boxes, strengths, targets, seam_feather, lora_on_text, patch, text_first,
                                      model.load_device, model.model.get_dtype_inference())
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, wrapper.outer_sample)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, wrapper)
        return io.NodeOutput(model)


class RegionalLayoutEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegionalLayoutEncode",
            display_name="Regional Layout Encode",
            category="model/conditioning/regional_lora",
            description="Encodes the prompt with the Create Bounding Boxes layout in the form the loaded text encoder's model understands: "
                        "the MOSS-Image layout JSON, or position words for Krea 2.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, tooltip="Description of the whole image."),
                io.Array.Input("elements", tooltip="Prompt elements from the node Create Bounding Boxes."),
                io.Int.Input("width", default=1024, min=16, max=16384, step=8, tooltip="Width of the generated image. Used by MOSS-Image, which places the boxes in pixels."),
                io.Int.Input("height", default=1024, min=16, max=16384, step=8, tooltip="Height of the generated image. Used by MOSS-Image, which places the boxes in pixels."),
                io.String.Input("character_boxes", default="", tooltip="MOSS-Image only: numbers of the boxes to label as characters instead of objects, e.g. \"1, 3\"."),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, prompt, elements, width, height, character_boxes) -> io.NodeOutput:
        te = clip.cond_stage_model
        if isinstance(te, comfy.text_encoders.moss_image.MOSSImageTEModel):
            text = MOSSImageLayoutPrompt.execute(prompt, elements, width, height, character_boxes).args[0]
        elif isinstance(te, comfy.text_encoders.krea2.Krea2TEModel):
            text = krea2_layout_caption(prompt, elements)
        else:
            raise ValueError("Regional Layout Encode supports the MOSS-Image and Krea 2 text encoders, not {}.".format(type(te).__name__))
        logging.info("Regional Layout Encode prompt:\n%s", text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(clip.tokenize(text)))
