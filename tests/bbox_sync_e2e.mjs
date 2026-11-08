// End-to-end check of web/bbox_resolution_sync.js in the real frontend, driven through headless Chrome.
//
// Needs ComfyUI running (no models are loaded) and Chrome. From the ComfyUI root:
//     node custom_nodes\MOSS-Image-Multi-Character\tests\bbox_sync_e2e.mjs [--server http://127.0.0.1:5090] [--vue]
//
// Graph: Resolution Selector -> width/height of Create Bounding Boxes and MOSS-Image Layout Prompt, two boxes splitting the
// canvas into left and right halves, Layout Prompt -> Preview Any. Each case changes the selector, queues through the real
// Run command and checks on the server that the layout boxes still split the full canvas at the size ResolutionSelector
// computes in Python. A width link from anything else must be refused without queueing.
import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const server = args.includes("--server") ? args[args.indexOf("--server") + 1] : "http://127.0.0.1:5090";
const vue = args.includes("--vue");
const COMFY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"];
const PORT = 9333;

// [aspect_ratio, megapixels, multiple, fire widget callbacks]; false = values change without callbacks, so only the queue-time sync can catch it
const CASES = [
    ["1:1 (Square)", 1.0, 8, true],
    ["4:3 (Standard)", 2.0, 32, true],
    ["9:16 (Portrait Widescreen)", 1.5, 16, true],
    ["21:9 (Ultrawide)", 1.0, 8, false],
    ["3:4 (Portrait Standard)", 2.0, 32, true],
    ["16:9 (Widescreen)", 0.5, 64, false],
    ["2:3 (Portrait Photo)", 1.2, 8, true],
    // heights that are multiples of 8 but not of 16, the step of the canvas size widgets
    ["4:3 (Standard)", 1.0, 8, true],
    ["4:3 (Standard)", 1.0, 8, false],
    ["16:9 (Widescreen)", 1.0, 8, false],
];

const expected = JSON.parse(execFileSync(path.join(COMFY, "venv/Scripts/python.exe"), ["-c",
    "import json, sys\nfrom comfy_extras.nodes_resolution import ResolutionSelector as R\n" +
    "print(json.dumps([list(R.execute(a, m, k).result) for a, m, k, _ in json.loads(sys.argv[1])]))", JSON.stringify(CASES)], { cwd: COMFY }).toString());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const getJSON = async (url) => (await fetch(url)).json();

const profile = mkdtempSync(path.join(tmpdir(), "moss-bbox-e2e-"));
const chromePath = CHROME.find((p) => { try { execFileSync(p, ["--version"], { stdio: "ignore", timeout: 10000 }); return true; } catch { return false; } }) ?? CHROME[0];
const chrome = spawn(chromePath, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, "--no-first-run", "--window-size=1600,1000", "about:blank"], { stdio: "ignore" });

let ws;
let nextId = 1;
const pending = new Map();
const consoleLines = [];

function send(method, params = {}) {
    const id = nextId++;
    ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression) {
    const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
}

async function connect() {
    for (let i = 0; i < 50; i++) {
        try {
            const page = (await getJSON(`http://127.0.0.1:${PORT}/json/list`)).find((t) => t.type === "page");
            if (page) {
                ws = new WebSocket(page.webSocketDebuggerUrl);
                await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
                ws.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    if (msg.id && pending.has(msg.id)) {
                        const { resolve, reject } = pending.get(msg.id);
                        pending.delete(msg.id);
                        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
                    } else if (msg.method === "Runtime.consoleAPICalled") {
                        const text = msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
                        if (text.startsWith("[MOSS bbox sync]")) consoleLines.push(`${msg.params.type}: ${text}`);
                    }
                };
                return;
            }
        } catch { /* chrome still starting */ }
        await sleep(200);
    }
    throw new Error("could not reach headless Chrome");
}

async function waitFor(expression, what, ms = 60000) {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(250)) {
        if (await evaluate(expression).catch(() => false)) return;
    }
    throw new Error("timed out waiting for " + what);
}

const APP = "(window.comfyAPI?.app?.app ?? window.app)";

async function queueAndWait() {
    const before = new Set(Object.keys(await getJSON(`${server}/history`)));
    await evaluate(`${APP}.extensionManager.command.execute("Comfy.QueuePrompt")`);
    for (const end = Date.now() + 30000; Date.now() < end; await sleep(250)) {
        const history = await getJSON(`${server}/history`);
        const id = Object.keys(history).find((k) => !before.has(k) && history[k].status?.completed);
        if (id) return history[id];
        const failed = Object.keys(history).find((k) => !before.has(k) && history[k].status?.status_str === "error");
        if (failed) throw new Error("prompt failed: " + JSON.stringify(history[failed].status.messages).slice(-1500));
    }
    return null;
}

const results = [];
let failures = 0;
function check(name, ok, detail) {
    results.push([ok ? "PASS" : "FAIL", name, detail]);
    if (!ok) failures++;
}

try {
    await connect();
    await send("Runtime.enable");
    await send("Page.enable");
    await send("Page.navigate", { url: server });
    await waitFor(`!!${APP}?.rootGraph && ${APP}.extensions.some((e) => e.name === "MOSS.BoundingBoxResolutionSync") && !!LiteGraph.registered_node_types["CreateBoundingBoxes"]`, "the frontend, the sync extension and the node definitions");
    if (vue) {
        await evaluate(`${APP}.extensionManager.setting.set("Comfy.VueNodes.Enabled", true)`);
        await sleep(500);
    }
    // the frontend restores its startup workflow asynchronously; build the test graph after that
    await sleep(5000);

    const ids = await evaluate(`(() => {
        const app = ${APP};
        const g = app.rootGraph;
        g.clear();
        const add = (type, pos) => { const n = LiteGraph.createNode(type); n.pos = pos; g.add(n); return n; };
        const w = (n, name) => n.widgets.find((x) => x.name === name);
        const sel = add("ResolutionSelector", [0, 0]);
        const bb = add("CreateBoundingBoxes", [320, 0]);
        const lp = add("MOSSImageLayoutPrompt", [760, 0]);
        const pa = add("PreviewAny", [1100, 0]);
        w(bb, "width").value = 1024;
        w(bb, "height").value = 1024;
        const meta = (desc) => ({ type: "obj", text: "", desc, palette: [] });
        w(bb, "editor_state").value = [{ x: 0, y: 0, width: 512, height: 1024, metadata: meta("left") }, { x: 512, y: 0, width: 512, height: 1024, metadata: meta("right") }];
        w(lp, "prompt").value = "bbox sync test";
        w(lp, "character_boxes").value = "1, 2";
        sel.connect(0, bb, bb.findInputSlot("width"));
        sel.connect(1, bb, bb.findInputSlot("height"));
        sel.connect(0, lp, lp.findInputSlot("width"));
        sel.connect(1, lp, lp.findInputSlot("height"));
        bb.connect(2, lp, lp.findInputSlot("element"));
        lp.connect(0, pa, 0);
        return { sel: sel.id, bb: bb.id, pa: pa.id };
    })()`);
    await sleep(500);
    // canvas width, height, and the right and bottom edges of the saved boxes, in the pixels the backend divides by
    const CANVAS = `(() => {
        const n = ${APP}.rootGraph.getNodeById(${ids.bb});
        const w = (k) => n.widgets.find((x) => x.name === k).value;
        const boxes = w("editor_state");
        return [w("width"), w("height"), Math.max(...boxes.map((b) => b.x + b.width)), Math.max(...boxes.map((b) => b.y + b.height))];
    })()`;

    for (const [i, [ratio, mp, multiple, live]] of CASES.entries()) {
        const [W, H] = expected[i];
        const name = `${ratio}, ${mp} MP, x${multiple}, ${live ? "live" : "queue-time only"}`;
        await evaluate(`(() => {
            const n = ${APP}.rootGraph.getNodeById(${ids.sel});
            for (const [k, v] of [["aspect_ratio", ${JSON.stringify(ratio)}], ["megapixels", ${mp}], ["multiple", ${multiple}]]) {
                const w = n.widgets.find((x) => x.name === k);
                w.value = v;
                if (${live}) w.callback?.(v);
            }
        })()`);
        await sleep(300);
        const canvas = await evaluate(CANVAS);
        const entry = await queueAndWait();
        if (!entry) { check(name, false, "nothing was queued"); continue; }
        const layout = JSON.parse(entry.outputs[String(ids.pa)].text[0]);
        const [a, b] = layout.items.map((it) => it.bbox);
        const tol = Math.ceil(Math.max(W, H) / 1000) + 1;
        const near = (v, t) => Math.abs(v - t) <= tol;
        const ok = a[0] === -Math.floor(H / 2) && a[1] === -Math.floor(W / 2) && a[2] === H - Math.floor(H / 2) && near(a[3], 0) &&
            b[0] === -Math.floor(H / 2) && near(b[1], 0) && b[2] === H - Math.floor(H / 2) && b[3] === W - Math.floor(W / 2);
        const liveOk = !live || (canvas[0] === W && canvas[1] === H);
        const after = await evaluate(CANVAS);
        const savedOk = after[0] === W && after[1] === H && after[2] === W && after[3] === H;
        check(name, ok && liveOk && savedOk, `python ${W}x${H}, canvas before queue ${canvas[0]}x${canvas[1]}, after ${after[0]}x${after[1]} ` +
            `with saved boxes reaching ${after[2]}x${after[3]}, layout boxes ${JSON.stringify(a)} ${JSON.stringify(b)}`);
    }

    if (vue) {
        // the mounted canvas must follow a size set without the widget callback, or its next box edit saves against a stale size
        const before = await evaluate(CANVAS);
        await evaluate(`(() => { const n = ${APP}.rootGraph.getNodeById(${ids.bb}); n.widgets.find((x) => x.name === "width").value = ${before[0] * 2}; })()`);
        await sleep(500);
        const after = await evaluate(CANVAS);
        check("Vue canvas re-saves its boxes when the size widget is set directly", after[2] === before[2] * 2,
            `width ${before[0]} -> ${after[0]}, saved boxes reach ${before[2]} -> ${after[2]}`);
    }

    // width linked from a Primitive Int: the canvas cannot know the value, so the run must be refused
    await evaluate(`(() => {
        const app = ${APP};
        const bb = app.rootGraph.getNodeById(${ids.bb});
        const p = LiteGraph.createNode("PrimitiveInt");
        p.pos = [0, 400];
        app.rootGraph.add(p);
        p.connect(0, bb, bb.findInputSlot("width"));
    })()`);
    await sleep(300);
    const entry = await queueAndWait().catch((e) => ({ error: e.message }));
    const refused = consoleLines.some((l) => l.startsWith("error:") && l.includes("link width and height"));
    check("width linked to Primitive Int is refused", !entry && refused, entry ? "a prompt was queued" : "not queued, error shown");
} catch (e) {
    check("harness", false, e.stack);
} finally {
    ws?.close();
    chrome.kill();
    await sleep(500);
    rmSync(profile, { recursive: true, force: true, maxRetries: 5 });
}

console.log(`Frontend mode: ${vue ? "Vue nodes" : "LiteGraph"}\n`);
for (const [status, name, detail] of results) console.log(`${status}  ${name}\n      ${detail}`);
console.log("\nExtension console output:\n" + consoleLines.map((l) => "  " + l).join("\n"));
console.log(failures ? `\n${failures} FAILED` : "\nall passed");
process.exit(failures ? 1 : 0);
