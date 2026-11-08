"""Single-forward checks of the regional LoRA node against ComfyUI's own LoRA patching.

Run from the ComfyUI root with its venv:
    venv\\Scripts\\python.exe custom_nodes\\MOSS-Image-Multi-Character\\tests\\forward_check.py <diffusion model> <lora>

1. Full-image box + LoRA on text must reproduce the globally weight-patched LoRA.
2. Every hooked layer kind (qkv, o, ffn_up_gate, ffn_down) must fire, also on int8.
3. A left-half box must change the left half much more than the right half.
"""
import collections
import importlib.util
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1] + ["--cuda-device", "0"] + sys.argv[1:]
model_name, lora_name = sys.argv[3], sys.argv[4]
fp32 = "--fp32" in sys.argv[5:]
sys.argv = sys.argv[:3]

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402,F401
import comfy.model_management  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

spec = importlib.util.spec_from_file_location("moss_regional_lora", pathlib.Path(__file__).resolve().parents[1] / "moss_regional_lora.py")
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)

calls = collections.Counter()
make_hook = node.make_hook


def counting_make_hook(entries, masks):
    hook = make_hook(entries, masks)

    def wrapped(module, args, out):
        calls[type(module).__name__ + ":" + str(tuple(module.weight.shape))] += 1
        return hook(module, args, out)
    return wrapped


node.make_hook = counting_make_hook


def run(patcher, x, sigma, ctx):
    comfy.model_management.load_models_gpu([patcher])
    # the sampler merges the patcher's wrappers into transformer_options (comfy/sampler_helpers.py)
    transformer_options = {"wrappers": patcher.wrappers}

    def forward():
        return patcher.model.apply_model(x, sigma, c_crossattn=ctx, transformer_options=transformer_options).float()
    # the sampler runs everything inside the OUTER_SAMPLE wrappers, which move the regional LoRA weights to the GPU
    outer = patcher.wrappers.get(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, {}).get(node.WRAPPER_KEY)
    if not outer:
        return forward()
    out = outer[0](forward)
    devices = {w.device.type for entries in outer[0].__self__.targets.values() for _, a, _ in entries for w in a.weights if torch.is_tensor(w)}
    print("regional LoRA weights after sampling are on:", devices)
    return out


def regional(patcher, bbox, lora_on_text):
    regions = {"region0": {"box": 1, "lora_name": lora_name, "strength": 1.0}}
    return node.MOSSImageRegionalLoRA.execute(patcher, [{"bbox": bbox}], regions, 0.0, lora_on_text).args[0]


dtype = torch.float32 if fp32 else torch.bfloat16
model_options = {"dtype": dtype} if dtype == torch.float32 else {}
base = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", model_name), model_options=model_options)
dev = comfy.model_management.get_torch_device()
g = torch.Generator(device="cpu").manual_seed(0)
x = torch.randn(1, 128, 64, 64, generator=g).to(dev, dtype)
ctx = torch.randn(1, 77, 4096, generator=g).to(dev, dtype)

wrapper = node.RegionalLoRAWrapper([[0, 0, 1000, 500]], [1.0], {}, 0.0, False, dev, dtype)
m = wrapper.masks(64, 64, 77, dev, dtype)[0].reshape(-1)
print("mask: left %.0f / %d tokens on, right %.0f, text %.0f" % (m[:4096].view(64, 64)[:, :32].sum(), 64 * 32, m[:4096].view(64, 64)[:, 32:].sum(), m[4096:].sum()))
sigma = torch.tensor([0.6], device=dev)

out_base = run(base, x, sigma, ctx)
lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
out_global = run(comfy.sd.load_lora_for_models(base, None, lora, 1.0, 0)[0], x, sigma, ctx)
out_full = run(regional(base, [0, 0, 1000, 1000], True), x, sigma, ctx)
print("hook calls per forward:", dict(calls))
calls.clear()
out_left = run(regional(base, [0, 0, 1000, 500], False), x, sigma, ctx)

lora_effect = (out_global - out_base).norm()
print("1. |regional_full - global| / |global - base| = %.4f  (|global - base| %.2f, |regional_full - base| %.2f)"
      % ((out_full - out_global).norm() / lora_effect, lora_effect, (out_full - out_base).norm()))
d = (out_left - out_base).abs().mean(dim=(0, 1))
print("3. mean |change| left half %.5f, right half %.5f" % (d[:, :32].mean(), d[:, 32:].mean()))
