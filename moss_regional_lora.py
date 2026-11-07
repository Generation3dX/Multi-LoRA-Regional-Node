"""Regional multi-LoRA for MOSS-Image.

Each region's LoRA delta is added at forward time, multiplied by a token mask
built from its Create Bounding Boxes box, so a LoRA cannot act outside its box.
Model weights are never patched, so bf16 and int8 checkpoints behave the same.
"""

import logging

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter
import folder_paths
from comfy_api.latest import io
from comfy_api.latest._io import _DynamicGroup

WRAPPER_KEY = "moss_regional_lora"
# adapters with a purely additive forward delta h(x); OFT/BOFT transform the output instead
ADDITIVE_ADAPTERS = (comfy.weight_adapter.LoRAAdapter, comfy.weight_adapter.LoHaAdapter, comfy.weight_adapter.LoKrAdapter)


def box_token_mask(bbox, rows, cols, feather, device):
    # bbox is Create Bounding Boxes' [y0, x0, y1, x1] on a 0-1000 scale; a MOSS token is one latent pixel
    y0, x0, y1, x1 = (v / 1000 for v in bbox)

    def ramp(n, a, b):
        # smoothstep that only feathers inward: 0 at and outside the edge, so the LoRA never reaches past its box
        width = max(1e-3, min(feather * n, 0.5 * (b - a)))
        t = torch.arange(n, device=device, dtype=torch.float32) + 0.5
        t = (torch.minimum(t - a, b - t) / width).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    return (ramp(rows, y0 * rows, y1 * rows)[:, None] * ramp(cols, x0 * cols, x1 * cols)[None, :]).flatten()


def make_hook(entries, masks):
    def hook(module, args, out):
        x = args[0]
        for region, adapter, part in entries:
            delta = adapter.h(x, out) * masks[region]
            if part is None:
                out.add_(delta)
            else:
                out.narrow(-1, part[0], part[1]).add_(delta)
    return hook


class RegionalLoRAWrapper:
    def __init__(self, boxes, strengths, targets, feather, lora_on_text):
        self.boxes = boxes
        self.strengths = strengths
        self.targets = targets  # {module name: [(region, adapter, (offset, size) or None)]}
        self.feather = feather
        self.lora_on_text = lora_on_text
        self.device = None

    def to_device(self, device, dtype):
        for entries in self.targets.values():
            for _, adapter, _ in entries:
                adapter.weights = tuple(w.to(device=device, dtype=dtype) if torch.is_tensor(w) else w for w in adapter.weights)
        self.device = device

    def masks(self, rows, cols, text_len, device, dtype):
        masks = torch.stack([box_token_mask(b, rows, cols, self.feather, device) for b in self.boxes])
        # where boxes overlap, the region with the stronger weighted mask owns the token instead of the identities summing
        weighted = torch.stack([m * abs(s) for m, s in zip(masks, self.strengths)])
        masks = masks * (weighted >= weighted.amax(dim=0))
        # MOSS sequence is [image (rows * cols) | text]
        text = masks.new_full((masks.shape[0], text_len), 1.0 if self.lora_on_text else 0.0)
        return torch.cat((masks, text), dim=1).to(dtype).unsqueeze(-1).unsqueeze(1).unbind(0)

    def __call__(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        if self.device != x.device:
            self.to_device(x.device, x.dtype)
        masks = self.masks(x.shape[-2], x.shape[-1], context.shape[1], x.device, x.dtype)
        dm = executor.class_obj
        handles = []
        forced = []
        try:
            for name, entries in self.targets.items():
                module = dm.get_submodule(name)
                handles.append(module.register_forward_hook(make_hook(entries, masks)))
                # linear_input_act fuses the input activation into quantized GEMMs without calling the module (ffn_down on int8)
                forced.append((module, module.comfy_force_forward))
                module.comfy_force_forward = True
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            for h in handles:
                h.remove()
            for module, value in forced:
                module.comfy_force_forward = value


def load_region_lora(model, key_map, lora_name, strength, region, targets):
    lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
    patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    matched = 0
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
        model.model.diffusion_model.get_submodule(name)
        patch.multiplier = strength
        targets.setdefault(name, []).append((region, patch, part))
        matched += 1
    logging.info("MOSS regional LoRA: region %d '%s' matched %d modules", region, lora_name, matched)
    if skipped:
        logging.warning("MOSS regional LoRA: '%s' has %d patches that cannot be applied regionally and are ignored: %s", lora_name, len(skipped), skipped[:5])
    if matched == 0:
        raise ValueError("MOSS regional LoRA: '{}' matched no MOSS-Image layers.".format(lora_name))


class MOSSImageRegionalLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageRegionalLoRA",
            display_name="MOSS-Image Regional LoRA",
            category="model/conditioning/moss_image",
            description="Applies each LoRA only inside its bounding box. Use the same Create Bounding Boxes elements for the MOSS-Image Layout Prompt.",
            inputs=[
                io.Model.Input("model"),
                io.Array.Input("elements", tooltip="Prompt elements from the node Create Bounding Boxes."),
                _DynamicGroup.Input(
                    "regions",
                    template=[
                        io.Int.Input("box", default=1, min=1, max=100, tooltip="Number of the box, as tagged in the Create Bounding Boxes preview."),
                        io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                        io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                    ],
                    min=1,
                    max=20,
                    tooltip="Each row applies one LoRA inside one box.",
                    group_name="Region",
                ),
                io.Float.Input("seam_feather", default=0.05, min=0.0, max=0.5, step=0.01, tooltip="Inward fade at the box edges as a fraction of the image. 0 is a hard edge."),
                io.Boolean.Input("lora_on_text", default=False, tooltip="Also apply every region's LoRA to all text tokens. Off keeps the LoRAs off the shared prompt tokens."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, elements, regions, seam_feather, lora_on_text) -> io.NodeOutput:
        if not isinstance(model.model, comfy.model_base.MOSSImage):
            raise ValueError("MOSS-Image Regional LoRA only supports MOSS-Image models.")
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        boxes, strengths, targets = [], [], {}
        for row in regions:
            if not row.get("lora_name") or row["strength"] == 0:
                continue
            box = row["box"]
            if not 1 <= box <= len(elements):
                raise ValueError("MOSS regional LoRA: box {} does not exist, there are {} boxes.".format(box, len(elements)))
            load_region_lora(model, key_map, row["lora_name"], row["strength"], len(boxes), targets)
            boxes.append(elements[box - 1]["bbox"])
            strengths.append(row["strength"])
        if not boxes:
            return io.NodeOutput(model)
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                                   RegionalLoRAWrapper(boxes, strengths, targets, seam_feather, lora_on_text))
        return io.NodeOutput(model)
