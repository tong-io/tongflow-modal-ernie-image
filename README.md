# tongflow-modal-ernie-image

Official TongFlow plugin. Text-to-image generation with **ERNIE-Image** (`baidu/ERNIE-Image-Turbo` by default, `baidu/ERNIE-Image` for the full model), running on a GPU via [Modal](https://modal.com). An alternative to `tongflow-modal-z-image` on the same `image-gen` slot.

## Capabilities

- **Image generation** (`image-gen`) — generate an image from a text prompt.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The ERNIE-Image weights are public; an `HF_TOKEN` (via Modal secret `huggingface`) is optional and only helps avoid Hugging Face rate limits.
