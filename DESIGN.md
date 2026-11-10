# DESIGN — MOSS-Image Regional Multi-LoRA (Phase 0)

Everything below was read from source on ComfyUI v0.42.5 (`483dae43`) or measured from the files on disk. File and line references are from that checkout.

## 1. TL;DR

- **Recommendation:** use masked activation deltas, as in Krea v1, re-targeted to MOSS. Let MOSS's native layout JSON handle placement. Take both the layout JSON and the LoRA masks from the same `Create Bounding Boxes` output. Port none of the V12 machinery.
- **A direct port of Krea v1 would silently fail in four ways on MOSS:**
  - It would match only 3 of the 5 LoRA'd projections per layer, because the checkpoints fuse q, k and v.
  - It would drop `ffn_down` on int8, because no module forward runs there.
  - It would place the mask over the wrong tokens, because MOSS puts the image first, not the text.
  - It would build the mask on a grid half the true resolution, because MOSS has no 2×2 patchify.

  §5 explains each one.
- **Decisions needed from you** are listed in §8.

## 2. Krea v1: how it actually works

Read from `krea2_regional_multilora.py`, which is a copy of the reference repo.

| Aspect | What the code does |
|---|---|
| Entry point | A `DIFFUSION_MODEL` wrapper (`add_wrapper_with_key`, l.835-843). On each call it registers `register_forward_hook` on every matched Linear and removes the hooks in `finally` (l.632-654). |
| Delta | `out += mask_i * (x @ down.T @ up.T)`. The scale (`alpha/rank * strength`) is folded into `up`. The delta is computed in bf16 (l.448-471). LoKr uses a Kronecker product without materializing the full weight (l.437-445). |
| Matching | Both LoRA keys and live module names are normalized by stripping trainer prefixes and removing `.` and `_` (l.167-196). A Krea-only alias table maps diffusers names to native names (l.199-253). Each region logs `matched X/Y` and warns only when X = 0 (l.527-537). |
| Mask | A rectangle with an **inward-only** smoothstep feather on the token grid, so the value is exactly 0 at the box edge and outside it (l.400-431). The grid is `latent_H//2 × latent_W//2`, Krea's patch size 2 (l.549-558). |
| Text tokens | The mask is 0 on text tokens. The image block is placed at `[txtlen : txtlen+n_img]`, where `txtlen` comes from `context.shape[1]` (l.596-630), because Krea's sequence is `[text \| image]`. |
| Overlap | **Deltas sum.** v1 has no "stronger wins" logic, so that README claim does not describe v1. |
| Loader | Loads raw A/B (or LoKr factors) into fp32 on CPU, then casts to bf16 on the device once (l.268-364, l.574-594). It never touches model weights, so it is quantization-safe. |
| Strength | The default is 1.1 per region, and `base_strength` multiplies it. |

## 3. MOSS-Image forward pass

Source: `comfy/ldm/moss_image/model.py`

- **Single-stream DiT.** `MOSSImage._forward` (l.157-190) does the following:
  - The image goes through `video_embedder` (a Linear from 128 to 3072) **with no patchify**.
  - Text goes through `text_embedder`: a projection plus 2 `TextBlock`s that are not LoRA'd.
  - The sequence is concatenated as **`[image (H·W) | text (T)]`** (l.169), and each of the 42 `Block`s processes it as one sequence (l.177-185).
  - The output head reads `x[:, :H*W]` (l.189).
- **Token order inside the image block.** It is row-major: token `r*W + c`. The RoPE ids confirm this (l.141-148): rows use `repeat_interleave(w)`, columns use `repeat(h)`. The text RoPE ids are offset after the image.
- **Box-to-grid math.** The latent format is `Flux2`: 128 channels, `spacial_downscale_ratio = 16` (`comfy/latent_formats.py:197-199`). The model consumes the latent `H×W` directly, so **1 token = 16×16 px**. 1024² gives a 64×64 grid of 4096 image tokens. A pixel box `[x0,x1)` maps to columns `x0/16 … x1/16`.
- **Batching.** Cond and uncond run as **separate** forward calls because their text lengths differ, and RoPE follows the true length (`model_base.py:2803-2809`). Each call has its own `T`, but the image span is always `[0, H·W)`. That span **does not depend on the text length**, which removes Krea's `txtlen` guesswork.
- **Block layout** (l.95-122). Each block has these projections:
  - `proj.qkv` (fused, 3072→9216) **or** `proj.q`, `proj.k`, `proj.v`. `fused_qkv` is set when `layers.0.proj.qkv.weight` exists (`model_detection.py:1032`).
  - `proj.o` (3072→3072).
  - `proj.ffn_up_gate` (3072→16384).
  - `proj.ffn_down` (8192→3072).

  `ffn_down` is called through `comfy.ops.linear_input_act(ffn_down, ffn_up_gate(y), "situ")` (l.122). Both residuals go through a gate: `x.addcmul_(y, gate)` (l.28-32), with `gate_a`/`gate_f` from the shared modulation `mod`.
- **Checkpoints on disk** (measured):
  - `moss_image_base_bf16` and `moss_image_sft_bf16` are both **fused qkv**: `layers.N.proj.qkv.weight` with shape 9216×3072. Neither has split q/k/v.
  - Both RealismByStableYogi int8_convrot files are fused qkv with `int8_tensorwise` + `convrot` on `qkv`, `o`, `ffn_up_gate` and `ffn_down`.
- **Core LoRA mapping.** ComfyUI maps LoRA `…proj.q/k/v` onto slices of the fused `qkv` weight (`comfy/lora.py:359-365`). A normal `LoraLoader` therefore works on these files, which matters for the global reference rows of the test matrix.

### int8_convrot: do module hooks see the right tensors?

- **`qkv`, `o` and `ffn_up_gate`: yes.** They are called as `module(x)`, so `register_forward_hook` fires. The hook sees the real bf16 input activation and the dequantized bf16 output, because quantization is internal to the op.
- **`ffn_down`: no.** `linear_input_act` (`comfy/ops.py:1039-1062`) never calls the module when the weight is a `QuantizedTensor`. It calls `weight.linear_input_act(x, …, input_act="situ")` directly (l.1049-1058, via l.1028-1036), which fuses situ into activation quantization. **A forward hook on `ffn_down` never fires on int8.**

  On bf16 it does fire (l.1044-1046, `linear(_eager_input_act(...))`), so the same code would apply the LoRA on bf16 and drop it on int8. That is exactly the kind of silent split this project must avoid.

### Memory compiler

The block loop runs inside `malloc_graph_begin/end` (l.176-187). MOSS does not pass `enable_graph=True`, so no CUDA graph is captured and Python hooks run every step. Extra allocations in hooks can only cause allocator "graph breaks", which are logged at debug level. That is a performance cost, not a correctness problem.

### Layer 22

**Both LoRAs cover layers 0-41 except layer 22.** Layer 22 is also the SLG uncond-skip layer recommended by the guide. ai-toolkit appears to exclude it on purpose. Separately, `SkipLayerGuidanceDiTSimple` skips layers by installing `patches_replace["dit"][("single_block", i)]` on the uncond pass (`comfy_extras/nodes_slg.py`). That makes it a conflict if this node also used block replacement.

## 4. Layout Prompt node

Source: `comfy_extras/nodes_moss_image.py:13-51`

- **Inputs:**
  - `prompt`.
  - `element`, an `io.Array` from `Create Bounding Boxes`. Its `bbox` is `[y0,x0,y1,x1]` on a 0-1000 scale.
  - `width` and `height`.
  - `character_boxes`, the 1-based box numbers to label `"character"` instead of `"object"`.
- **Output:** a JSON **string**: `{"summary":{"caption":…},"items":[{label,caption,bbox},…],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`. Each bbox is converted to centre-origin pixels, with `floor` on the low edge and `ceil` on the high edge (l.43-46). This string goes into a normal `CLIPTextEncode`, so the layout is plain text conditioning. Nothing reaches the DiT except the text.
- **Not the `BOUNDING_BOX` type.** `Create Bounding Boxes` outputs both a `BOUNDING_BOX` (`bboxes`) and an `Array` (`elements`) (`comfy_extras/nodes_bounding_boxes.py:345-349`). The layout node consumes **`elements`**, not `BOUNDING_BOX`. The CLAUDE.md assumption that they share the `BOUNDING_BOX` type is wrong. See decision D2.
- **Base or SFT.** The upstream USERGUIDE documents layout prompts ("Experimental") only in the Base guide. It shows only the `text` and `object` labels and says nothing about SFT. Layout on SFT is **unverified** and has to be tested empirically (matrix row 5).
- **Guide sampler settings for Base:**
  - ≥ 50 steps, Euler, CFG 3, shift 5.
  - SLG on layer 22, uncond skip.
  - Total pixels of 1M, 1.5M or 2M, in multiples of 16.

  The guide also says to keep the text encoder frozen when post-training, so the LoRAs touch only the DiT.

## 5. Krea-specific workarounds MOSS does not need, and Krea code that would break

**Not needed:**

| Krea feature | Why MOSS doesn't need it |
|---|---|
| Krea diffusers↔native alias table | ai-toolkit MOSS keys already use native names (`diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`). Only the q/k/v → fused-qkv slice mapping is needed. |
| `txtlen` / trailing-block mask placement | The MOSS image span is always `[0, H·W)`. |
| V12 exact Qwen token-span resolution for vision-grounding tokens | MOSS uses no vision tokens in plain T2I. The Qwen chat template is fixed (`comfy/text_encoders/moss_image.py`). |
| V12 attraction field and box-authoritative framing | Placement comes from the native layout JSON. Only add these if testing shows subjects drifting out of their boxes. |
| V12 FlexAttention block-sparse ownership | Phase 3 only, if leakage survives masking. |
| Scene/outfit transfer, edit patch, detailer, reference lock | Out of scope. |
| `split_mode` auto strips, `blend_override`, `KREA2_MASKS` / `KREA2_DATA` outputs | Not needed: the boxes always come from the layout. Under AGENTS.md, nodes should not emit unused pass-through or debug outputs. |

**Would break if ported as-is:**

1. **Name matching misses q/k/v.** Live modules are `layers.N.proj.qkv`, while the LoRA has `proj.q/k/v`. Krea would match 3 of 5 per layer, logging `123/205`. It only warns at 0, so the miss would be silent.
2. **`ffn_down` is dropped on int8 only** (§3).
3. **Mask placement.** With `txtlen = T`, Krea writes the mask at `[T : T+H·W]`, which spans the image tail and the text. The image starts at 0.
4. **Grid.** Krea uses `latent // 2`, but MOSS needs `latent // 1`.

## 6. Candidate designs

### A. Masked activation deltas re-targeted to MOSS (recommended)

The same principle as Krea v1, wired to MOSS's real structure:

- **Wrapper:** a `DIFFUSION_MODEL` wrapper installs hooks per call and removes them in `finally`, as in Krea. The image span is `[0, H·W)` from `x.shape`.
- **`qkv` hook:** computes q, k and v deltas from the hooked input with each region's `q`, `k` and `v` LoRA, then adds them into the matching third of the output. For split-qkv checkpoints, it hooks `q`, `k` and `v` directly.
- **`o` and `ffn_up_gate` hooks:** plain output hooks.
- **`ffn_down`:** the `ffn_up_gate` hook keeps its (patched) output for that block call. A **forward hook on the `Block` itself** then adds `gate_f * mask * (situ(h) @ A.T @ B.T)` to the block output, with `gate_f = mod[5]` from the block's inputs. This is algebraically identical to adding the delta inside `gated_residual` (l.122). It works on bf16 and int8 alike, and it needs no `patches_replace`, so it doesn't conflict with SLG. If SLG skips a block, the block doesn't run and no delta is added, which is correct.

  The cost is one eager `situ` on the 16384-wide pre-activation per block, about 128 MB transient at 1024². It is freed when the block returns.
- **Masks:** inward smoothstep feather, as in Krea, on the exact MOSS grid. Box edges use the same `floor`/`ceil` rounding as the layout node, so the LoRA scope and the layout box cover the same pixels.

  On overlap, the region with the larger `mask·strength` takes each token, instead of summing. Summing two identities in the overlap is the bleed we want to avoid. This is a small, explicit change from v1.
- **Text tokens:** default **off** (mask 0 on `[H·W, H·W+T)`), with a toggle for test row 6. "On" applies every region's LoRA to all text tokens, which is a deliberate leak and only useful as an experiment.

  A per-caption token-span mask is *not* proposed, for two reasons. Qwen is causal, so every token after `gen3dx_joni` in the JSON already carries it. And text tokens attend bidirectionally over the whole image in the DiT. Span masking would cost complexity without isolating anything.
- **Loader:** reuse Krea's A/B/alpha + LoKr parsing, but drop the Krea alias table. Match `layers.N.proj.X` exactly, plus the q/k/v → qkv slice. Fail loudly when matched < expected: log per-region `matched/expected` and warn on **any** shortfall, not just 0. As built (§9), both test LoRAs report 246 adapters: 41 layers × 6, with q, k and v counted separately and layer 22 absent.

**Pros:** works identically on bf16 and int8_convrot. Never touches weights. LoRA effect is exactly zero outside its box. Small (one file, about 300-400 lines). The design is already proven on Krea.

**Cons:**
- Hook overhead: 4 hooks per block × 42 blocks, plus rank-32 matmuls per region. Expect roughly 5-15 % slower sampling, to be measured.
- Identity may be weaker than a global LoRA, because the text tokens never see the LoRA and only the box sees it. Expect strength > 1, as Krea needed 1.5.
- Attention still mixes information across boxes, since unmasked image tokens attend to masked ones. That is residual leakage which Design D would address.

### B. Same deltas, implemented by replacing each block via `patches_replace`

This means re-implementing `Block.forward` with deltas inline, which removes the extra `situ`.

**Rejected.** It duplicates core MOSS code, so it breaks whenever upstream changes the block (and the block just changed for the memory compiler and fused int8). It also collides with `SkipLayerGuidanceDiTSimple`, which owns `("single_block", 22)` and would drop our deltas or be dropped itself.

### C. Native layout only + stacked global LoRAs (no custom node)

`LoraLoader` ×2 + Layout Prompt with `character` labels and trigger words in each caption.

This is the cheapest option, and it is test row 2. Layout controls *where* the characters go, but both LoRAs act on every token, so identities **will** blend. It's the baseline to beat, not a solution.

### D. Attention-level ownership (Phase 3 only)

Block each box's image tokens from attending to the other boxes' image tokens or caption spans. It is precise, but `optimized_attention` here has no mask path. It would need FlexAttention/Triton, which is fragile on Windows, or a dense-mask SDPA fallback, which is slow and memory-heavy at 4k+ tokens.

Only worth it if the Phase 2 matrices show off-diagonal similarity that Design A cannot reduce.

**Recommendation: A**, with C kept as the measured baseline. Revisit D only on evidence.

## 7. Phase 1 node sketch (Design A)

- **`MOSSRegionalMultiLoRA`:**
  - Inputs: `model`, `elements` (Array from Create Bounding Boxes), `width`, `height`, `regions` (autogrow of `{lora, strength}`; region i ↔ element i), `seam_feather`, `text_tokens` (bool, default off).
  - Output: `MODEL`.
- **Wiring:** the same `elements` also feed the core **MOSS-Image Layout Prompt**. The trigger words go in each box's caption, and the boxes are marked as `character_boxes`. No new prompt node is needed.
- **Logging:** each region logs `matched/expected` once, and the session logs its grid once.

## 8. Decisions (resolved 2026-11-07)

- **D1:** Design A.
- **D2:** `elements`.
- **D3:** clean slate. Krea files and history were removed; the MIT LICENSE and the credit were kept.
- **D4:** int8 testing uses `moss_RealismByStableYogi Moss V1.5 Civ INT8-ConvRot C88` because it is the production target. Its base checkpoint is unknown, so the int8 rows are labelled as a transfer experiment against Base-trained LoRAs.

## 9. Phase 1 as built (differs from §6/§7 in two places)

- **Loader:** Krea's loader was not ported. The node uses core instead:
  - `comfy.lora.model_lora_keys_unet` + `comfy.lora.load_lora`. These already map ai-toolkit `proj.q/k/v` onto `(qkv.weight, (0, offset, size))` slices (`comfy/lora.py:359-365`).
  - The adapter's own forward delta `h(x)`, from `comfy/weight_adapter/lora.py:288` and the LoKr and LoHa equivalents.

  So every key format and adapter type that core supports works without a local copy. The two test LoRAs report **246 matched adapters**: 41 layers × q, k, v, o, ffn_up_gate, ffn_down. Non-additive patches (OFT/BOFT, diff, set) are logged and ignored, and a LoRA with 0 matches raises an error.
- **int8 `ffn_down`:** no block-level hook. The wrapper sets `comfy_force_forward = True` on each hooked module for the duration of the forward call, and restores it in `finally`. `linear_input_act` then takes its eager path, `linear(situ(x))` (`comfy/ops.py:1044-1046`), so the module forward and its hook run, with the int8 GEMM still doing the matmul. Core bypass LoRA uses the same flag (`comfy/weight_adapter/bypass.py:202-203`).
- **Nodes:**
  - `MOSSImageRegionLoRA`: `box`, `lora_name`, `strength`. It outputs a `MOSS_REGION_LORA`.
  - `MOSSImageRegionalLoRA`: `model`, `elements`, `regions` (Autogrow of `MOSS_REGION_LORA`), `seam_feather`, `lora_on_text`.

  `box` is the 1-based box number, the same numbering as the Layout Prompt's `character_boxes`. The grid comes from the latent at call time, so there are no width/height inputs.

  A first version used core's `_DynamicGroup` rows, as `LoadLoraModel` does. The installed frontend (1.55.18) has no renderer for `COMFY_DYNAMICGROUP_V3` and showed `regions` as a dead socket. Autogrow is rendered, which is why there is a separate Regional LoRA Loader node (node id `MOSSImageRegionLoRA`).

### Single-forward checks

`tests/forward_check.py`, 64×64 latent, Joni LoRA, strength 1:

| Check | Base bf16 | Base, fp32 compute | Yogi int8 C88 |
|---|---|---|---|
| Hook calls per forward (qkv / o / up_gate / down) | 41 / 41 / 41 / 41 | 41 / 41 / 41 / 41 | 41 / 41 / 41 / 41 |
| Full box + text vs. core `LoraLoader`, rel. diff | 0.111 | **0.005** | 0.360 |
| LoRA effect size, core vs. regional | 182.7 / 182.6 | — | 247.8 / 246.2 |
| Left-half box: mean change left / right | 0.147 / 0.113 | 0.145 / 0.109 | 0.212 / 0.164 |

How to read these:
- At fp32 the node reproduces core LoRA patching, so the bf16 and int8 gaps come from rounding `W + ΔW` back into the stored dtype in the *reference*. The effect sizes agree to within 1 %.
- The masks are exactly 0 outside the box and on the text tokens. The right half still changes through attention: unmasked tokens attend to LoRA-modified ones. That is the residual leakage Phase 2 must measure on real images.

## 8a. Decisions needed (original Phase 0 list)

- **D1 Design:** approve A, or choose another.
- **D2 Box input:** `elements` (Array, which is what the Layout node consumes, so both read identical coordinates) or `BOUNDING_BOX` (the `bboxes` output; same pixels, but a second wire that could get out of sync). **Recommend `elements`.**
- **D3 Project folder:** the folder is a full copy of the Krea repo, including its git history. The same repo is **also installed** as `custom_nodes/Krea2-Multi-Character-…`, so both register identical node IDs. Options:
  - (a) Clean slate: delete the Krea files, keep LICENSE and credit, start a fresh git history. **Recommended.**
  - (b) Keep the Krea files on a branch.
- **D4 int8 Base:** there is **no int8_convrot Base or SFT** on disk. The only int8 files are RealismByStableYogi finetunes, whose base is unknown.
  - Rows 4-5 need the official Comfy-Org int8_convrot Base, which you would download.
  - Otherwise the Yogi int8 file becomes a labelled transfer experiment, and the int8 hook path is checked by matched counts plus a bf16-vs-int8 delta comparison instead.

## 10. Phase 2 results

Harness: `tests/matrix.py` queues API workflows on a running ComfyUI through `/prompt`, then scores the images with insightface `buffalo_l` (ArcFace, CPU). Setup: 1024×1024, Joni boxed on the left half and Sansa on the right, 4 fixed seeds, LoRA strength 1.0, neutral clothed portrait prompts. Each character's reference is the mean face embedding of its solo, globally applied LoRA images on the same checkpoint. Full tables are in `tests/results/results.md`, contact sheets in `tests/results/*_sheet.jpg`.

Mean margin (similarity to the right character minus the wrong one):

| config | Yogi V1.5 int8_convrot (transfer) | Base bf16 | SFT bf16 (transfer) |
|---|---|---|---|
| no LoRA (control) | 0.13 | 0.07 | 0.13 |
| both LoRAs stacked globally | 0.02 | 0.06 | 0.05 |
| regional | **0.42** | 0.31 | 0.36 |
| regional + `lora_on_text` | 0.35 | **0.34** | — |

- Stacking globally makes both women look like Joni: box 2 scores 0.58–0.69 against Joni. The regional node gives each box its own identity on every checkpoint. Box 2's similarity to Joni drops to 0.15–0.35, close to the no-LoRA control.
- The int8_convrot path performs at least as well as bf16, so the `comfy_force_forward` hook path holds up on real sampling.
- Layout also works on SFT.
- `lora_on_text` hurts on Yogi (0.35 vs 0.42) and helps slightly on Base (0.34 vs 0.31). Keep it off by default.
- No third person and no missing faces in any image.
- Residual leakage through attention is small. On Yogi, box 2's similarity to Joni is 0.35 with the node vs 0.16 with no LoRA. Visually it shows up as occasional borrowed details, such as Joni's fringe on Sansa. Masking alone meets the goal, so Phase 3 (attention ownership) is not justified by these numbers.
