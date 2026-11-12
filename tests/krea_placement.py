"""Research probe: does Krea 2 put subjects inside boxes from the text alone?

No LoRAs. Two layouts x three prompt styles x 4 seeds on Krea 2 turbo int8_convrot.
Run from the ComfyUI root with ComfyUI running: venv\Scripts\python.exe custom_nodes\MOSS-Image-Multi-Character\tests\krea_placement.py [server URL]
Score: insightface face detection; for each box, is there a face whose centre lies inside it,
and the face height relative to the box height.
"""
import json
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5090"
COMFY = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent / "results" / "krea_placement"
SIZE = 1024
SEEDS = [11, 12, 13, 14]

# boxes as Create Bounding Boxes' [ymin, xmin, ymax, xmax] on 0-1000
LAYOUTS = {
    "halves": [("a woman with bright red hair, wearing a white t-shirt", [0, 0, 1000, 500]),
               ("a woman with long black hair, wearing a white t-shirt", [0, 500, 1000, 1000])],
    # a small distant woman at the back left, a large close woman on the right
    "uneven": [("a woman with bright red hair, wearing a white t-shirt", [380, 80, 760, 300]),
               ("a woman with long black hair, wearing a white t-shirt", [0, 450, 1000, 950])],
}
SCENE = "two women in a plain grey photo studio, realistic photograph, soft studio lighting"


def prose(layout, positional):
    (a, ba), (b, bb) = LAYOUTS[layout]
    if not positional:
        return f"{SCENE}. One is {a}. The other is {b}."
    if layout == "halves":
        return f"{SCENE}. On the left is {a}. On the right is {b}."
    return f"{SCENE}. In the background on the left, small and far away, stands {a}. In the foreground on the right, close to the camera, stands {b}."


def ideogram(layout):
    elements = [{"type": "obj", "bbox": box, "desc": desc} for desc, box in LAYOUTS[layout]]
    return json.dumps({"high_level_description": SCENE,
                       "compositional_deconstruction": {"background": "plain grey photo studio", "elements": elements}})


STYLES = {"prose_noposition": lambda l: prose(l, False), "prose_positional": lambda l: prose(l, True), "ideogram_json": ideogram}


def build(text, seed, prefix):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea\\krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": text}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                                                         "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


def request(path, data=None):
    req = urllib.request.Request(SERVER + path, data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()[:2000]) from None


def generate(prompt, dest):
    pid = request("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        entry = request("/history/" + pid).get(pid)
        if entry and entry["status"].get("completed"):
            break
        if entry and entry["status"].get("status_str") == "error":
            raise RuntimeError(json.dumps(entry["status"].get("messages"))[-2000:])
        time.sleep(0.5)
    img = entry["outputs"]["save"]["images"][0]
    shutil.copyfile(COMFY / "output" / img["subfolder"] / img["filename"], dest)


def main():
    OUT.mkdir(exist_ok=True)
    for layout in LAYOUTS:
        for style, fn in STYLES.items():
            for s in SEEDS:
                dest = OUT / f"{layout}_{style}_{s}.png"
                if not dest.exists():
                    t = time.time()
                    generate(build(fn(layout), s, "krea_placement/" + dest.stem), dest)
                    print(dest.name, "%.1fs" % (time.time() - t), flush=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("\n| layout | prompt | box 1 hit | box 2 hit | face/box height b1 | b2 | faces per image |")
    print("|---|---|---|---|---|---|---|")
    for layout, boxes in LAYOUTS.items():
        sheet = Image.new("RGB", (256 * len(SEEDS) + 150, 256 * len(STYLES)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, style in enumerate(STYLES):
            hits = np.zeros(2)
            ratios = [[], []]
            counts = []
            draw.text((4, r * 256 + 120), style, fill="black")
            for c, s in enumerate(SEEDS):
                im = Image.open(OUT / f"{layout}_{style}_{s}.png").convert("RGB")
                faces = app.get(np.asarray(im)[:, :, ::-1])
                counts.append(len(faces))
                for b, (_, (y0, x0, y1, x1)) in enumerate(boxes):
                    X0, Y0, X1, Y1 = x0 * SIZE / 1000, y0 * SIZE / 1000, x1 * SIZE / 1000, y1 * SIZE / 1000
                    inside = [f for f in faces if X0 <= (f.bbox[0] + f.bbox[2]) / 2 <= X1 and Y0 <= (f.bbox[1] + f.bbox[3]) / 2 <= Y1]
                    if inside:
                        hits[b] += 1
                        f = max(inside, key=lambda f: f.bbox[3] - f.bbox[1])
                        ratios[b].append((f.bbox[3] - f.bbox[1]) / (Y1 - Y0))
                thumb = im.copy()
                d = ImageDraw.Draw(thumb)
                for _, (y0, x0, y1, x1) in boxes:
                    d.rectangle([x0 * SIZE / 1000, y0 * SIZE / 1000, x1 * SIZE / 1000, y1 * SIZE / 1000], outline=(0, 255, 0), width=6)
                for f in faces:
                    d.rectangle(list(f.bbox), outline=(255, 0, 0), width=6)
                sheet.paste(thumb.resize((256, 256)), (150 + c * 256, r * 256))
            fmt = lambda v: "%.2f" % np.mean(v) if v else "-"
            print(f"| {layout} | {style} | {int(hits[0])}/{len(SEEDS)} | {int(hits[1])}/{len(SEEDS)} | {fmt(ratios[0])} | {fmt(ratios[1])} | {','.join(map(str, counts))} |")
        sheet.save(OUT / f"sheet_{layout}.jpg", quality=88)


if __name__ == "__main__":
    sys.exit(main())
