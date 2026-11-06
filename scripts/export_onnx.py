# Converts AuroraIMG2-50M-Preview (https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview, Apache-2.0) and the
# TAEF2 decoder (https://huggingface.co/madebyollin/taef2, MIT) to the ONNX files in model/.
# Only needed to regenerate them. Needs: torch diffusers onnx onnxruntime safetensors numpy
#   python scripts/export_onnx.py
# TAEF2 replaces the model's bundled FLUX.2 small decoder: it decodes the same latents ~10x faster (0.4s vs 3.6s on 2
# CPU threads) at 5 MB instead of 112 MB, and looks the same at 256px. Weights are read from .safetensors files only.
import sys, json, shutil, urllib.request, importlib.util
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from safetensors.torch import load_file
from onnxruntime.quantization import quantize_dynamic, QuantType
torch.set_grad_enabled(False)

REPO, REV = "AuroraAI-Research/AuroraIMG2-50M-Preview", "ce244e5d0277b2964fe26b34aacac45540b674b6"
TAEF2, TAEF2_REV, TAESD_REV = "madebyollin/taef2", "bd244ebfe4398c84fbf312a2f1c868c676b500e3", "fdbae2df989db95761f482efb8977e1182fb68d2"
ROOT = Path(__file__).resolve().parent.parent
HF, OUT = ROOT / ".hf", ROOT / "model"
FILES = {f: f"https://huggingface.co/{REPO}/resolve/{REV}/{f}" for f in
         ["pipeline.py", "tokenizer/tokenizer.json"] + [f"{m}/{n}" for m in ("text_encoder", "transformer") for n in ("config.json", "diffusion_pytorch_model.safetensors")]}
FILES |= {"taef2.safetensors": f"https://huggingface.co/{TAEF2}/resolve/{TAEF2_REV}/taef2.safetensors",
          "taesd.py": f"https://raw.githubusercontent.com/madebyollin/taesd/{TAESD_REV}/taesd.py"}
for f, url in FILES.items():
    p = HF / f
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, p)

def module(name):
    spec = importlib.util.spec_from_file_location(name, HF / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
P, T = module("pipeline"), module("taesd")

def load(cls, sub):
    cfg = {k: v for k, v in json.loads((HF / sub / "config.json").read_text()).items() if not k.startswith("_")}
    m = cls(**cfg); m.load_state_dict(load_file(HF / sub / "diffusion_pytorch_model.safetensors"), strict=True); return m.eval(), cfg
te, te_cfg = load(P.AuroraTextEncoder, "text_encoder")
den, den_cfg = load(P.AuroraDenoiser, "transformer")
assert den_cfg["latent_layout"] == "flux2_pipeline", den_cfg["latent_layout"]
tae = T.TAESD(encoder_path=None, decoder_path=None, latent_channels=32, arch_variant="flux_2")
# The safetensors keys are diffusers-style (decoder.layers.N); taesd.py's decoder starts with an extra layer
tae.load_state_dict({(lambda e, _l, i, *s: ".".join([e, str(int(i) + (e == "decoder")), *s]))(*k.split(".")): v
                     for k, v in load_file(HF / "taef2.safetensors").items()})

class TextEncoder(torch.nn.Module):
    def __init__(s): super().__init__(); s.te = te
    def forward(s, ids): return s.te(ids)[0]
class Denoiser(torch.nn.Module):
    def __init__(s): super().__init__(); s.den = den
    def forward(s, z, t, ctx, mask): return s.den(z, t, ctx, mask)
class Decoder(torch.nn.Module):
    """Unpacks the model's 128x16x16 latents to FLUX.2's 32x32x32 and decodes them to 0..1 RGB with TAEF2. Unlike the
    FLUX.2 decoder, TAEF2 takes the latents still batch-normalized, so the pipeline's BatchNorm un-normalization is skipped."""
    def __init__(s): super().__init__(); s.dec = tae.decoder
    def forward(s, z): return s.dec(P._unpatchify_flux2(z)).clamp(0, 1)

L = te_cfg["max_tokens"]
ids = torch.zeros(2, L, dtype=torch.long); ids[:, :5] = torch.tensor([te_cfg["bos_id"], 300, 301, 302, te_cfg["eos_id"]])
mask = ids != te_cfg["pad_id"]
ctx, z, t = te(ids)[0], torch.randn(2, 128, 16, 16), torch.tensor([0.7, 0.7])
OUT.mkdir(exist_ok=True)
graphs = {
    "text_encoder": (TextEncoder(), (ids,), ["ids"], ["ctx"]),
    "denoiser": (Denoiser(), (z, t, ctx, mask), ["z", "t", "ctx", "mask"], ["v"]),
    "decoder": (Decoder(), (z,), ["z"], ["image"]),
}
for name, (m, args, ins, outs) in graphs.items():
    f = OUT / f"{name}.onnx"
    torch.onnx.export(m.eval(), args, f, input_names=ins, output_names=outs, dynamic_axes={k: {0: "batch"} for k in ins + outs}, dynamo=False, opset_version=18)
    feed = {k: a.numpy() for k, a in zip(ins, args)}
    want = m(*args).numpy()
    got = ort.InferenceSession(f, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    err = float(np.abs(got - want).max())
    print(f"{name}.onnx: max abs diff vs torch {err:.2e}")
    assert err < 1e-3, name
    # int8 weights for the transformers' MatMuls: about 4x less RAM and disk, same images. The decoder is mostly convolutions and already small.
    if name != "decoder":
        q = f.with_suffix(".int8.onnx")
        quantize_dynamic(f, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        got = ort.InferenceSession(q, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        print(f"  int8: max abs diff vs fp32 {float(np.abs(got - want).max()):.2e}, mean {float(np.abs(got - want).mean()):.2e}")
        q.replace(f)
    print(f"  {f.stat().st_size / 1e6:.1f} MB")
shutil.copyfile(HF / "tokenizer/tokenizer.json", OUT / "tokenizer.json")
(OUT / "config.json").write_text(json.dumps({k: te_cfg[k] for k in ("max_tokens", "pad_id", "bos_id", "eos_id", "null_id")}, indent=2) + "\n")
