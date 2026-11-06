# AuroraIMG2-50M API

A small, authenticated text-to-image API for direct-img. It runs [AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview), a 50M-parameter diffusion transformer with a 12.7M-parameter text encoder, on CPU with onnxruntime, and decodes with [TAEF2](https://huggingface.co/madebyollin/taef2) instead of the bundled FLUX.2 small decoder. On 2 threads of a 4-core Haswell VPS, an image takes about 6s and the container peaks at about 260 MB of RAM.

The output is 256×256 and upscaled. Subjects usually come out recognizable (cats, cars, rooms, landscapes), but prompts with several objects, counts, positions or actions are often wrong, and faces, hands and text are weak.

It replaces [AuroraIMG-6M](https://github.com/direct-img/AuroraIMG-6M) (64×64, ~0.4s, ~150 MB), which gets the colors and mood right but rarely the object.

## API

```
GET /generate?prompt=frog+wizard+casting+a+spell
Authorization: Bearer $AURORA_API_KEY
```

| Param | Default | Range |
|---|---|---|
| `prompt` | required | 1-200 characters (only the first 62 tokens are used) |
| `size` | `512` | 64-1024 (output is square, upscaled from 256×256) |
| `format` | `webp` | `webp`, `png`, `jpeg` |
| `steps` | `12` | 1-50 (12 looks close to 20, at 60% of the time; each step is ~0.5s) |
| `cfg` | `3` | 0-20 (guidance; ≤1 disables it and halves the time) |
| `seed` | CRC32 of the lowercased prompt | 0-4294967295 |

The same prompt and params always give the same image. Responses include `X-Seed` and `X-Generation-Ms`.

Errors are JSON `{"error": "..."}`: 400 bad params, 401 missing or wrong key, 503 when the queue is full (`Retry-After: 5`).

`GET /health` returns `ok` without auth.

## Config

| Env var | Default | |
|---|---|---|
| `AURORA_API_KEY` | none | Required. Without it `/generate` rejects every request |
| `PORT` | `8080` | |
| `THREADS` | `2` | onnxruntime threads per generation (1 thread: ~10.5s an image) |
| `CONCURRENCY` | `1` | Generations at once; more requests queue |
| `MAX_QUEUE` | `3` | Queued requests beyond this get 503, so the last in line waits ~25s at most |

## Deploy

Pushing to `main` builds the image with GitHub Actions, pushes it to `ghcr.io/direct-img/auroraimg2-50m:main`, then calls the Dokploy deploy webhook (repo secret `DOKPLOY_WEBHOOK_URL`). It runs as the AuroraIMG2-50M app in the direct-img Dokploy project, at `aurora2.direct-img.link`.

Run locally:

```sh
pip install -r requirements.txt
AURORA_API_KEY=dev python app.py
curl -H "Authorization: Bearer dev" "localhost:8080/generate?prompt=orange+cat" -o cat.webp
```

## Model and license

`model/` holds an ONNX conversion of [AuroraAI-Research/AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview) at revision `ce244e5`, licensed under the Apache License 2.0 ([`model/LICENSE`](model/LICENSE)), and of the [TAEF2](https://huggingface.co/madebyollin/taef2) decoder at revision `bd244eb`, licensed under the MIT License ([`model/LICENSE-TAEF2`](model/LICENSE-TAEF2)). Attribution and changes are in [`model/NOTICE`](model/NOTICE); all three ship in the Docker image.

The text encoder and denoiser weights are quantized to int8, which takes them from 255 MB to 76 MB. In fp32 the ONNX pipeline matches the original one exactly (same pixels from the same starting noise). int8 changes images by about 9-24/255 per pixel on average, which looks like a different sample of the same prompt rather than a worse one. TAEF2 takes the latents still batch-normalized, unlike the FLUX.2 decoder, so the pipeline's un-normalization step is skipped.

The model was trained on [FLUX-Reason-6M](https://huggingface.co/datasets/QLNI/FLUX-Reason-6M-flux2-latents) latents; the dataset has its own terms.

`scripts/export_onnx.py` regenerates `model/` from the original repos (needs torch, diffusers, onnx, onnxruntime, safetensors).
