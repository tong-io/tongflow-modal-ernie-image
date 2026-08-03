"""Modal deploy entry for ernie-image.

Deploy:
  modal deploy deploy.py

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.

ERNIE-Image uses Hugging Face diffusers ``ErnieImagePipeline``. Support landed
after PyPI 0.36.x; we pin a diffusers git revision that includes the pipeline.
Default weights: ``baidu/ERNIE-Image-Turbo`` (8 steps, guidance 1.0). Swap to
(50 steps, guidance 4.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset
from tongflow.slots import node_slot


# Diffusers commit adding ERNIE-Image pipeline (see huggingface/diffusers#13432).
DIFFUSERS_GIT = (
    "git+https://github.com/huggingface/diffusers.git@"
    "dc8d9032171c83741fd37ed2b12bc9d8274464f3"
)


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "baidu/ERNIE-Image-Turbo")
MODEL_DIR = f"/models/{REPO_ID}"

_TURBOISH = "turbo" in REPO_ID.lower()
DEFAULT_STEPS = 8 if _TURBOISH else 50
DEFAULT_GUIDANCE = 1.0 if _TURBOISH else 4.0
DEFAULT_USE_PE = True

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        DIFFUSERS_GIT,
        "tongflow==0.2.21", "fastapi[standard]",
        "transformers==5.4.0",
        "safetensors==0.7.0",
        "loguru==0.7.3",
        "pillow==12.1.1",
        "accelerate==1.13.0",
        "huggingface_hub==1.6.0",
        "tqdm==4.67.3",
        "sentencepiece==0.2.1",
    )
)

with image.imports():
    import torch
    from diffusers import ErnieImagePipeline


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
)
class Inference:
    @modal.enter()
    def load(self):
        self.pipe = ErnieImagePipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def _png_bytes(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        use_pe: bool | None = None,
        seed: int = 42,
    ) -> bytes:
        import io

        steps = DEFAULT_STEPS if num_inference_steps is None else num_inference_steps
        gs = DEFAULT_GUIDANCE if guidance_scale is None else guidance_scale
        pe = DEFAULT_USE_PE if use_pe is None else use_pe

        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=int(steps),
            guidance_scale=float(gs),
            use_pe=pe,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        use_pe: bool | None = None,
        seed: int = 42,
    ) -> bytes:
        return self._png_bytes(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_pe=use_pe,
            seed=seed,
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")

        # `use_pe` is plugin-specific and not part of the ABI; keep the default.
        raw = self._png_bytes(
            text,
            height=input.height if input.height is not None else 1024,
            width=input.width if input.width is not None else 1024,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            use_pe=None,
            seed=int(input.seed) if input.seed is not None else 42,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

