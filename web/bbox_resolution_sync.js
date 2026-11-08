import { app } from "../../scripts/app.js";

// Create Bounding Boxes saves its boxes in pixels of its own width/height widgets, but the backend divides them by the
// width/height it receives. When those inputs are linked, the canvas never sees the linked values and the boxes silently
// shrink or stretch. For links from a Resolution Selector this keeps the widgets equal to the selector's output and
// rescales the saved boxes with them; any other link is refused at queue time.

// comfy_extras/nodes_resolution.py ASPECT_RATIOS
const RATIOS = {
    "1:1 (Square)": [1, 1],
    "2:3 (Portrait Photo)": [2, 3],
    "3:2 (Photo)": [3, 2],
    "3:4 (Portrait Standard)": [3, 4],
    "4:3 (Standard)": [4, 3],
    "9:16 (Portrait Widescreen)": [9, 16],
    "16:9 (Widescreen)": [16, 9],
    "21:9 (Ultrawide)": [21, 9],
};
const TAG = "[MOSS bbox sync]";

// Python's round(): halves go to the even neighbour
function pyRound(v) {
    const r = Math.round(v);
    return v % 1 === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// ResolutionSelector.execute, step for step so the floats match
function selectorSize(node) {
    const label = widget(node, "aspect_ratio")?.value;
    const ratio = RATIOS[label];
    if (!ratio) throw new Error(`Resolution Selector #${node.id}: unknown aspect ratio "${label}", update ${TAG} RATIOS.`);
    const multiple = widget(node, "multiple").value;
    const scale = Math.sqrt((widget(node, "megapixels").value * 1024 * 1024) / (ratio[0] * ratio[1]));
    return [pyRound((ratio[0] * scale) / multiple) * multiple, pyRound((ratio[1] * scale) / multiple) * multiple];
}

function linkSource(node, name) {
    const slot = node.findInputSlot(name);
    const link = slot >= 0 ? node.getInputLink(slot) : null;
    return link ? { node: node.graph.getNodeById(link.origin_id), slot: link.origin_slot } : null;
}

function* graphNodes(graph) {
    for (const node of graph.nodes ?? graph._nodes ?? []) {
        yield node;
        if (node.subgraph) yield* graphNodes(node.subgraph);
    }
}

// returns a log line, or null when width/height are not linked (then the widgets are what the backend receives)
function syncBoxes(bbox) {
    const w = linkSource(bbox, "width");
    const h = linkSource(bbox, "height");
    if (!w && !h) return null;
    if (!w || !h || w.node !== h.node || w.node?.comfyClass !== "ResolutionSelector" || w.slot !== 0 || h.slot !== 1) {
        throw new Error(`Create Bounding Boxes #${bbox.id}: link width and height both to the width and height outputs of one Resolution Selector, or unlink both. ` +
            "The canvas cannot see any other linked value, so its boxes would be scaled to the wrong size.");
    }
    const [W, H] = selectorSize(w.node);
    const ww = widget(bbox, "width");
    const hw = widget(bbox, "height");
    const editor = widget(bbox, "editor_state");
    const W0 = ww.value;
    const H0 = hw.value;
    const boxes = Array.isArray(editor.value) ? editor.value : [];
    const line = `#${bbox.id} <- Resolution Selector #${w.node.id}: ${W}x${H}, ${boxes.length} boxes`;
    if (W0 === W && H0 === H) return { line, rescaled: false };
    editor.value = boxes.map((b) => ({
        ...b, x: Math.round((b.x * W) / W0), y: Math.round((b.y * H) / H0),
        width: Math.round((b.width * W) / W0), height: Math.round((b.height * H) / H0),
    }));
    editor.callback?.(editor.value);
    // no widget callback: it snaps to the widget's step of 16, while the selector can output any multiple of 8
    ww.value = W;
    hw.value = H;
    return { line: `${line}, rescaled from ${W0}x${H0}`, rescaled: true };
}

function syncAll() {
    const lines = [];
    for (const node of graphNodes(app.rootGraph ?? app.graph)) {
        if (node.comfyClass !== "CreateBoundingBoxes") continue;
        const result = syncBoxes(node);
        if (!result) continue;
        lines.push(result.line);
        if (result.rescaled) {
            app.extensionManager?.toast?.add({ severity: "info", summary: "Bounding boxes resized", detail: result.line, life: 3000 });
        }
    }
    return lines;
}

function syncLive() {
    try {
        for (const line of syncAll()) console.info(TAG, line);
    } catch (e) {
        console.warn(TAG, e.message);
    }
}

app.registerExtension({
    name: "MOSS.BoundingBoxResolutionSync",
    setup() {
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            let lines;
            try {
                lines = syncAll();
            } catch (e) {
                console.error(TAG, e.message);
                app.extensionManager?.toast?.add({ severity: "error", summary: "Run not queued", detail: e.message });
                return false;
            }
            for (const line of lines) console.info(TAG, "queue", line);
            // let the canvas widget re-save its boxes at the new size before the prompt is built
            await new Promise((resolve) => setTimeout(resolve, 0));
            return queuePrompt.apply(this, args);
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === "ResolutionSelector") {
            for (const name of ["aspect_ratio", "megapixels", "multiple"]) {
                const w = widget(node, name);
                if (!w) continue;
                const callback = w.callback;
                w.callback = function (...args) {
                    const r = callback?.apply(this, args);
                    syncLive();
                    return r;
                };
            }
        } else if (node.comfyClass === "CreateBoundingBoxes") {
            const onConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                const r = onConnectionsChange?.apply(this, args);
                queueMicrotask(syncLive);
                return r;
            };
        }
    },
    afterConfigureGraph() {
        syncLive();
    },
});
