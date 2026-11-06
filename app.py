# AuroraIMG2-50M image API. GET /generate?prompt=... with "Authorization: Bearer $AURORA_API_KEY" returns an image.
# Runs the ONNX export of https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview (int8 weights), decoded
# with TAEF2 (https://huggingface.co/madebyollin/taef2), on CPU with onnxruntime.
import os, io, json, hmac, time, zlib, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

KEY = os.environ.get("AURORA_API_KEY", "").encode()
PORT = int(os.environ.get("PORT", 8080))
THREADS = int(os.environ.get("THREADS", 2))          # onnxruntime threads per generation
CONCURRENCY = int(os.environ.get("CONCURRENCY", 1))  # generations at once; the rest queue
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 3))      # queued requests beyond this get 503
FORMATS = {"webp": ("WEBP", "image/webp", {"quality": 90}), "png": ("PNG", "image/png", {}), "jpeg": ("JPEG", "image/jpeg", {"quality": 90})}

M = Path(__file__).parent / "model"
opts = ort.SessionOptions()
opts.intra_op_num_threads, opts.inter_op_num_threads = THREADS, 1
opts.enable_cpu_mem_arena = False  # ~90 MB less peak RAM (260 vs 350 MB) at the same speed
S = {n: ort.InferenceSession(M / f"{n}.onnx", opts, providers=["CPUExecutionProvider"]) for n in ("text_encoder", "denoiser", "decoder")}
run = lambda n, **x: S[n].run(None, x)[0]
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
C = json.loads((M / "config.json").read_text())

# Same layout as the reference pipeline: <bos>, up to 62 tokens of the prompt with a leading space, <eos>, padded to 64
def encode(p):
    ids = tok.encode(" " + " ".join(p.split()), add_special_tokens=False).ids[:C["max_tokens"] - 2]
    t = np.array([[C["bos_id"], *ids, C["eos_id"]] + [C["pad_id"]] * (C["max_tokens"] - 2 - len(ids))], np.int64)
    return run("text_encoder", ids=t), t != C["pad_id"]

# The learned empty prompt that guidance pushes away from: <bos> <null> <eos>
NULL = np.full((1, C["max_tokens"]), C["pad_id"], np.int64)
NULL[0, :3] = C["bos_id"], C["null_id"], C["eos_id"]
NULL_CTX, NULL_MASK = run("text_encoder", ids=NULL), NULL != C["pad_id"]

# Euler on the rectified-flow ODE from t=1 (noise) to 0, with classifier-free guidance, as in the reference pipeline
def generate(prompt, steps, cfg, seed):
    ctx, m = encode(prompt)
    z = np.random.default_rng(seed).standard_normal((1, 128, 16, 16), dtype=np.float32)
    ts = np.linspace(1, 0, steps + 1, dtype=np.float32)
    for a, b in zip(ts[:-1], ts[1:]):
        if cfg > 1:
            v_u, v_c = run("denoiser", z=np.concatenate([z, z]), t=np.full(2, a, np.float32),
                           ctx=np.concatenate([NULL_CTX, ctx]), mask=np.concatenate([NULL_MASK, m]))
            v = v_u + cfg * (v_c - v_u)
        else:
            v = run("denoiser", z=z, t=np.full(1, a, np.float32), ctx=ctx, mask=m)[0]
        z = z + (b - a) * v[None]
    img = run("decoder", z=z)[0].transpose(1, 2, 0)
    return Image.fromarray((img * 255).round().astype(np.uint8))

def params(q):
    def num(k, cast, lo, hi, default):
        try: v = cast(q[k]) if k in q else default
        except ValueError: raise ValueError(f"{k} must be a number")
        if not lo <= v <= hi: raise ValueError(f"{k} must be {lo}-{hi}")
        return v
    prompt = q.get("prompt", "").strip()
    if not 1 <= len(prompt) <= 200: raise ValueError("prompt must be 1-200 characters")
    fmt = q.get("format", "webp").lower()
    if fmt not in FORMATS: raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    return dict(prompt=prompt, steps=num("steps", int, 1, 50, 12), cfg=num("cfg", float, 0, 20, 3.0),
                seed=num("seed", int, 0, 2**32 - 1, zlib.crc32(prompt.lower().encode())), size=num("size", int, 64, 1024, 512), fmt=fmt)

gen_slots, queue_lock, queued = threading.BoundedSemaphore(CONCURRENCY), threading.Lock(), 0

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(s, code, body, ctype="application/json", **headers):
        s.send_response(code)
        for k, v in {"Content-Type": ctype, "Content-Length": len(body), "Cache-Control": "no-store", **headers}.items(): s.send_header(k, str(v))
        s.end_headers(); s.wfile.write(body)

    err = lambda s, code, msg, **h: s.reply(code, json.dumps({"error": msg}).encode(), **h)

    def do_GET(s):
        global queued
        u = urlparse(s.path)
        if u.path == "/health": return s.reply(200, b"ok", "text/plain")
        if u.path != "/generate": return s.err(404, "not found")
        if not KEY or not hmac.compare_digest(s.headers.get("Authorization", "").encode(), b"Bearer " + KEY): return s.err(401, "unauthorized")
        try: p = params({k: v[-1] for k, v in parse_qs(u.query).items()})
        except ValueError as e: return s.err(400, str(e))
        with queue_lock:
            if queued >= CONCURRENCY + MAX_QUEUE: return s.err(503, "busy, retry shortly", **{"Retry-After": 5})
            queued += 1
        try:
            t0 = time.perf_counter()
            with gen_slots: img = generate(p["prompt"], p["steps"], p["cfg"], p["seed"])
            if p["size"] != 256: img = img.resize((p["size"],) * 2, Image.LANCZOS)
            fmt, ctype, save = FORMATS[p["fmt"]]
            buf = io.BytesIO(); img.save(buf, fmt, **save)
            s.reply(200, buf.getvalue(), ctype, **{"X-Seed": p["seed"], "X-Generation-Ms": round((time.perf_counter() - t0) * 1000)})
        finally:
            with queue_lock: queued -= 1

if __name__ == "__main__":
    if not KEY: print("warning: AURORA_API_KEY is not set, so /generate rejects every request", flush=True)
    generate("warmup", 2, 3.0, 0)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"listening on :{PORT} (threads={THREADS}, concurrency={CONCURRENCY})", flush=True)
    server.serve_forever()
