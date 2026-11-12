"""Identity test matrix: queue API workflows on a running ComfyUI and score identity separation.

Run from the ComfyUI root with its venv, with ComfyUI already running (run_5090_no_sage.bat):
    venv\\Scripts\\python.exe custom_nodes\\MOSS-Image-Multi-Character\\tests\\matrix.py [--server http://127.0.0.1:5090] [--seeds 4] [--only yogi,cyber]

Images already generated are reused, so a rerun only queues what is missing and rescores.
Results go to tests/results/: images/, one contact sheet and one table per checkpoint, and results.md with all tables.

Score: insightface buffalo_l (ArcFace) on CPU. Each character's reference is the mean face embedding of
its solo, globally applied LoRA images on the same checkpoint. For every two-person image, the largest face
inside each box is compared to every reference: a strong diagonal and a weak off-diagonal mean the identities
stayed in their boxes. solo_regional applies one LoRA regionally on a full-canvas box, which shows how much
likeness the regional path keeps against the global LoRA.
"""
import argparse
import json
import pathlib
import shutil
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
COMFY = HERE.parents[2]
RESULTS = HERE / "results"
SIZE = 1024

FAMILIES = {
    "moss": dict(clip="qwen3.5_9b_bf16.safetensors", clip_type="moss_image", vae="flux2-vae.safetensors", latent="EmptyFlux2LatentImage", characters=[
        ("joni", "moss\\moss_joni_v1.0-3000.safetensors", "gen3dx_joni, a woman with ginger hair and green eyes, wearing a plain white t-shirt"),
        ("sansa", "moss\\moss_sansa_v1.0_2500.safetensors", "gen3dx_sansa, a woman with dark brown hair and dark brown eyes, wearing a plain white t-shirt"),
    ]),
    "krea2": dict(clip="qwen3vl_4b_bf16.safetensors", clip_type="krea2", vae="qwen_image_vae.safetensors", latent="EmptySD3LatentImage", characters=[
        ("joni", "krea2\\krea2_joni_v1.2-3000.safetensors", "a woman with ginger hair and green eyes, wearing a plain white t-shirt"),
        ("sansa", "krea2\\krea2_sansa_v1.0-5000.safetensors", "a woman with dark brown hair and dark brown eyes, wearing a plain white t-shirt"),
    ]),
}
# adjacent, non-overlapping halves, pixels on the SIZE canvas
BOXES = [(0, 0, SIZE // 2, SIZE), (SIZE // 2, 0, SIZE // 2, SIZE)]
SCENE = "two women standing side by side against a plain grey studio backdrop, waist-up, both facing the camera.\n\nrealism, realistic, photography, soft studio lighting, sharp focus"
SOLO = "{}, standing against a plain grey studio backdrop, waist-up, facing the camera.\n\nrealism, realistic, photography, soft studio lighting, sharp focus"
NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"

# sampler settings: Yogi from the user's production workflow, Base/SFT from the MOSS Base t2i template, Krea 2 turbo finetunes at 8 steps
CHECKPOINTS = {
    "yogi": dict(family="moss", unet="moss\\moss_RealismByStableYogi Moss V1.5 Civ INT8-ConvRot C88.safetensors", steps=9, cfg=1.0, sampler="euler", scheduler="beta",
                 label="MOSS Yogi V1.5 int8_convrot (transfer: base model unknown)"),
    "base": dict(family="moss", unet="moss\\moss_image_base_bf16.safetensors", steps=25, cfg=2.5, sampler="er_sde", scheduler="simple", label="MOSS Base bf16"),
    "sft": dict(family="moss", unet="moss\\moss_image_sft_bf16.safetensors", steps=25, cfg=2.5, sampler="er_sde", scheduler="simple",
                label="MOSS SFT bf16 (cross-checkpoint transfer experiment)"),
    "cyber": dict(family="krea2", unet="krea\\krea2_Cyberrealistic_v40_int8.safetensors", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                  label="Krea 2 Cyberrealistic v4.0 int8 (finetune)"),
    "ollie": dict(family="krea2", unet="krea\\krea2_sickOllie_int8_convrot.safetensors", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                  label="Krea 2 SickOllie int8_convrot (finetune)"),
}
# solo: index of the one character in a single-person image, else a two-person layout
CONFIGS = {
    "ref_joni": dict(solo=0, globals=[(0, 1.0)]),
    "ref_sansa": dict(solo=1, globals=[(1, 1.0)]),
    "solo_regional_joni": dict(solo=0, regional=1.0),
    "solo_regional_sansa": dict(solo=1, regional=1.0),
    "no_lora": dict(),
    "stacked": dict(globals=[(0, 1.0), (1, 1.0)]),
    "regional": dict(regional=1.0),
    "regional_text": dict(regional=1.0, text=True),
}
ROWS = {ck: list(CONFIGS) for ck in CHECKPOINTS}
ROWS["base"] = ["ref_joni", "ref_sansa", "no_lora", "stacked", "regional", "regional_text"]
ROWS["sft"] = ["ref_joni", "ref_sansa", "no_lora", "stacked", "regional"]


def build_prompt(ck, config, seed, prefix):
    family = FAMILIES[ck["family"]]
    characters = family["characters"]
    cfg = CONFIGS[config]
    solo = cfg.get("solo")
    p = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": ck["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": family["clip"], "type": family["clip_type"], "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": family["vae"]}},
        "latent": {"class_type": family["latent"], "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
    }
    model = ["unet", 0]
    for i, (c, strength) in enumerate(cfg.get("globals", [])):
        p[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": characters[c][1], "strength_model": strength}}
        model = [f"lora{i}", 0]
    meta = lambda desc: {"type": "obj", "text": "", "desc": desc, "palette": []}
    if solo is None:
        boxes = list(zip(BOXES, range(len(characters))))
        editor = [{"x": x, "y": y, "width": w, "height": h, "metadata": meta(characters[c][2])} for (x, y, w, h), c in boxes]
    else:
        boxes = [((0, 0, SIZE, SIZE), solo)]
        editor = [{"x": 0, "y": 0, "width": SIZE, "height": SIZE, "metadata": meta(characters[solo][2])}]
    p["boxes"] = {"class_type": "CreateBoundingBoxes", "inputs": {"width": SIZE, "height": SIZE, "editor_state": {"__value__": editor}}}
    if solo is None:
        p["pos"] = {"class_type": "RegionalLayoutEncode", "inputs": {"clip": ["clip", 0], "prompt": SCENE, "elements": ["boxes", 2], "width": SIZE, "height": SIZE,
                                                                     "character_boxes": ", ".join(str(i + 1) for i in range(len(BOXES)))}}
    else:
        p["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": SOLO.format(characters[solo][2])}}
    if "regional" in cfg:
        p["regional"] = {"class_type": "MOSSImageRegionalLoRA", "inputs": {"model": model, "elements": ["boxes", 2], "seam_feather": 0.05, "lora_on_text": cfg.get("text", False)}}
        for i, (_, c) in enumerate(boxes):
            p[f"region{i}"] = {"class_type": "MOSSImageRegionLoRA", "inputs": {"box": i + 1, "lora_name": characters[c][1], "strength": cfg["regional"]}}
            p["regional"]["inputs"][f"regions.region{i}"] = [f"region{i}", 0]
        model = ["regional", 0]
    p["sampler"] = {"class_type": "KSampler", "inputs": {"model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                                                         "seed": seed, "steps": ck["steps"], "cfg": ck["cfg"], "sampler_name": ck["sampler"],
                                                         "scheduler": ck["scheduler"], "denoise": 1.0}}
    p["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    p["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return p


def request(server, path, data=None):
    req = urllib.request.Request(server + path, data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("{} {}: {}".format(path, e.code, e.read().decode()[:3000])) from None


def generate(server, prompt, dest):
    prompt_id = request(server, "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        entry = request(server, "/history/" + prompt_id).get(prompt_id)
        if entry and entry["status"].get("completed"):
            break
        if entry and entry["status"].get("status_str") == "error":
            raise RuntimeError("ComfyUI failed on {}: {}".format(dest.name, json.dumps(entry["status"].get("messages"))[-2000:]))
        time.sleep(1)
    image = entry["outputs"]["save"]["images"][0]
    src = COMFY / "output" / image["subfolder"] / image["filename"]
    shutil.copyfile(src, dest)


def largest_face(app, image):
    faces = app.get(np.ascontiguousarray(np.asarray(image)[:, :, ::-1]))
    if not faces:
        return None, 0
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding, len(faces)


def score(app, ck_name, seeds):
    """Per config: (similarity matrix of box or solo face x reference, missing faces, faces per image)."""
    images = RESULTS / "images"
    load = lambda config, s: Image.open(images / f"{ck_name}_{config}_{s}.png").convert("RGB")
    refs = []
    for name, _, _ in FAMILIES[CHECKPOINTS[ck_name]["family"]]["characters"]:
        embs = [e for s in seeds if (e := largest_face(app, load("ref_" + name, s))[0]) is not None]
        ref = np.mean(embs, axis=0)
        refs.append(ref / np.linalg.norm(ref))
    rows = {}
    for config in ROWS[ck_name]:
        sims, missing, faces = [], 0, []
        solo = "solo" in CONFIGS[config]
        for s in seeds:
            image = load(config, s)
            emb, count = largest_face(app, image)
            faces.append(count)
            crops = [emb] if solo else [largest_face(app, image.crop((x, y, x + w, y + h)))[0] for x, y, w, h in BOXES]
            matrix = np.full((len(crops), len(refs)), np.nan)
            for b, e in enumerate(crops):
                if e is None:
                    missing += 1
                else:
                    matrix[b] = [float(e @ r) for r in refs]
            sims.append(matrix)
        rows[config] = (np.nanmean(np.stack(sims), axis=0), missing, faces)
    return rows


def contact_sheet(ck_name, seeds, path):
    configs = ROWS[ck_name]
    tile = 256
    sheet = Image.new("RGB", (tile * len(seeds) + 150, tile * len(configs)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, config in enumerate(configs):
        draw.text((6, r * tile + tile // 2), config, fill="black")
        for c, s in enumerate(seeds):
            sheet.paste(Image.open(RESULTS / "images" / f"{ck_name}_{config}_{s}.png").convert("RGB").resize((tile, tile)), (150 + c * tile, r * tile))
    sheet.save(path, quality=90)


def table(ck_name, rows):
    names = [n for n, _, _ in FAMILIES[CHECKPOINTS[ck_name]["family"]]["characters"]]
    lines = ["## " + CHECKPOINTS[ck_name]["label"], "", f"Contact sheet: `{ck_name}_sheet.jpg`", "",
             "Single person, similarity to each reference (`ref_*` against their own mean shows the ceiling):", "",
             "| config | vs " + " | vs ".join(names) + " | missing faces | faces |", "|---" * (len(names) + 3) + "|"]
    for config, (m, missing, faces) in rows.items():
        if "solo" in CONFIGS[config]:
            lines.append("| {} | {} | {} | {} |".format(config, " | ".join("{:.3f}".format(v) for v in m[0]), missing, ",".join(map(str, faces))))
    lines += ["", "Two people, similarity of the face in each box to each reference; `margin` = diag - off:", "",
              "| config | " + " | ".join(f"box {i + 1} vs {n}" for i in range(len(BOXES)) for n in names) + " | diag | off | margin | missing faces | faces |",
              "|---" * (len(BOXES) * len(names) + 6) + "|"]
    for config, (m, missing, faces) in rows.items():
        if "solo" in CONFIGS[config]:
            continue
        diag = np.nanmean(np.diag(m))
        off = np.nanmean(m[~np.eye(len(BOXES), dtype=bool)])
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            config, " | ".join("{:.3f}".format(v) for v in m.flatten()), diag, off, diag - off, missing, ",".join(map(str, faces))))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:5090")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--only", default=",".join(CHECKPOINTS))
    args = parser.parse_args()
    seeds = [1000 + i for i in range(args.seeds)]
    ck_names = args.only.split(",")
    (RESULTS / "images").mkdir(parents=True, exist_ok=True)

    for ck_name in ck_names:
        for config in ROWS[ck_name]:
            for s in seeds:
                dest = RESULTS / "images" / f"{ck_name}_{config}_{s}.png"
                if dest.exists():
                    continue
                t = time.time()
                generate(args.server, build_prompt(CHECKPOINTS[ck_name], config, s, "moss_regional_test/" + dest.stem), dest)
                print("%s %.1fs" % (dest.name, time.time() - t), flush=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    for ck_name in ck_names:
        contact_sheet(ck_name, seeds, RESULTS / f"{ck_name}_sheet.jpg")
        (RESULTS / f"{ck_name}.md").write_text(table(ck_name, score(app, ck_name, seeds)), encoding="utf-8")
        print((RESULTS / f"{ck_name}.md").read_text(encoding="utf-8"))
    header = ["# Identity matrix", "",
              "ArcFace cosine similarity to each character's solo global-LoRA reference, mean over seeds {}.".format(seeds), ""]
    sections = [(RESULTS / f"{ck}.md").read_text(encoding="utf-8") for ck in CHECKPOINTS if (RESULTS / f"{ck}.md").exists()]
    (RESULTS / "results.md").write_text("\n".join(header + sections), encoding="utf-8")


if __name__ == "__main__":
    main()
