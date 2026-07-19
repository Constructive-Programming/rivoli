#!/usr/bin/env python3
"""Range-request extract GLM-5.2 layer 78 (MTP) from HF and convert to the
engine's snapshot format. The zai-org/GLM-5.2 checkpoint is BF16 (no fp8), so:
  - weight matrices (experts, shared, attn projections, eh_proj) -> int4
    (colibri scheme: per-row s=max(|w|.max/7,1e-8), q=clip(rint(w/s),-8,7),
     nibble+8, low=col2j high=col2j+1); written as `<name>.weight`(U8) +
     `<name>.weight.qs`(F32).
  - norms / router gate / e_score_correction_bias -> F32 (widened).
  - indexer.* -> BF16 verbatim.
Resumable: each tensor cached under .parts; assembled into out-mtp-00000.safetensors.
"""
import json, os, struct, sys, time, urllib.request
import numpy as np

REPO = "https://huggingface.co/zai-org/GLM-5.2/resolve/main"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "out-mtp-00000.safetensors")
STATE = OUT + ".state.json"
PARTS = OUT + ".parts"
LAYER = 78

def ranged_get(url, start, end, tries=6):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
                assert len(data) == end - start + 1, f"short {len(data)}!={end-start+1}"
                return data
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"  retry {a+1} {e}", flush=True); time.sleep(2 ** a)

def shard_header(url):
    n = struct.unpack("<Q", ranged_get(url, 0, 7))[0]
    return json.loads(ranged_get(url, 8, 8 + n - 1)), 8 + n

def bf16_to_f32(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)

def quant_int4(w):  # w: [O,I] f32 -> (U8 [O*ceil(I/2)], f32 scale [O])  (colibri math)
    O, I = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 7.0, 1e-8)
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8); out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8); out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def kind(name):
    if ".indexer" in name:
        return "skip"  # layer-78 indexer already lives in out-idx (dedup)
    if name.endswith(("norm.weight", "layernorm.weight", "enorm.weight",
                      "hnorm.weight", "e_score_correction_bias")) or name.endswith("mlp.gate.weight"):
        return "f32"
    return "int4"  # experts, shared_experts, attn projections, eh_proj

def main():
    idx = json.load(open(os.path.join(SCRATCH, "glm52-index.json")))
    wm = idx["weight_map"]
    names = sorted(k for k in wm if f".layers.{LAYER}." in k)
    by_shard = {}
    for n in names:
        by_shard.setdefault(wm[n], []).append(n)
    print(f"{len(names)} tensors, {len(by_shard)} shards", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}

    for shard in sorted(by_shard):
        todo = [n for n in by_shard[shard] if n not in state]
        if not todo:
            continue
        url = f"{REPO}/{shard}"
        hdr, base = shard_header(url)
        for name in todo:
            m = hdr[name]; s, e = m["data_offsets"]
            dt = m["dtype"]
            assert dt in ("BF16", "F32"), f"{name} is {dt} (expected BF16 or F32)"
            raw = ranged_get(url, base + s, base + e - 1)
            # widen bf16→f32; f32 passes through
            f32 = (lambda r: np.frombuffer(r, np.float32) if dt == "F32" else bf16_to_f32(r))
            k = kind(name)
            if k == "skip":
                continue
            if k == "bf16":
                # indexer weights are bf16; if the source is f32, narrow back to bf16
                if dt == "BF16":
                    _emit(state, name, "BF16", m["shape"], raw)
                else:
                    v = np.frombuffer(raw, np.float32)
                    _emit(state, name, "BF16", m["shape"],
                          (v.view(np.uint32) >> 16).astype(np.uint16).tobytes())
            elif k == "f32":
                _emit(state, name, "F32", m["shape"], f32(raw).astype(np.float32).tobytes())
            else:
                w = f32(raw).astype(np.float32).reshape(m["shape"])
                packed, scale = quant_int4(w)
                _emit(state, name, "U8", [int(packed.size)], packed.tobytes())
                _emit(state, name + ".qs", "F32", [int(scale.size)], scale.tobytes())
            json.dump(state, open(STATE, "w"))
            print(f"  {name} {k} {m['shape']}", flush=True)
    _assemble(state)

def _emit(state, name, dtype, shape, data_bytes):
    part = os.path.join(PARTS, name.replace("/", "_"))
    with open(part, "wb") as f:
        f.write(data_bytes)
    state[name] = {"dtype": dtype, "shape": shape, "part": part}

def _assemble(state):
    order = sorted(state)
    off, entries = 0, {}
    for n in order:
        sz = os.path.getsize(state[n]["part"])
        entries[n] = {"dtype": state[n]["dtype"], "shape": state[n]["shape"],
                      "data_offsets": [off, off + sz]}
        off += sz
    hb = json.dumps(entries).encode()
    with open(OUT, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for n in order:
            f.write(open(state[n]["part"], "rb").read())
    print(f"wrote {OUT} ({(8+len(hb)+off)/1e9:.2f} GB, {len(order)} tensors)", flush=True)

if __name__ == "__main__":
    main()
