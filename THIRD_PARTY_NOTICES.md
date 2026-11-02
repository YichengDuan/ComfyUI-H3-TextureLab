# Provenance and third-party notices

## Akatz AI / Akatz Labs — original workflow lineage

This project began by adapting **Character Swap V1.5 · H3 Relay 2.0**, rather than designing the original character-swap workflow or training its LoRA.

- H3 Relay: https://github.com/akatz-ai/h3-relay
- Character Swap V1.5 and original workflow: https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
- Exact starting workflow: https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/examples/H3-Character-Swap-Relay-V2.json
- Toolkit description: https://github.com/akatz-ai/h3-relay/blob/main/docs/toolkit-v2.md

H3 Relay source is GPL-3.0. Its original notices and authorship remain upstream. The initial local experiments modified a runtime copy of Relay's workflow/helpers. The standalone release implements its own disk AV timeline and exposes new example workflows; it does not bundle the user's private Relay installation or original private workflow files. The Character Swap model weights are not redistributed, and the Ref2V texture experiments do not use the Character Swap adapter.

## NikoDemon80 — H3 Motion Context

https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Reviewed experimental dependency revision: `518c083d11d80fa5920222ad89889924ee4401c9`.

Native latent tail conditioning, temporal layout checks, Refine Blend, and Join Match belong to this upstream GPL-3.0 project. They are called through its registered nodes. Install the dependency separately. It is not included as an uncredited copy in this repository.

## ComfyUI / Comfy Kitchen / MiniMax

ComfyUI provides native H3 conditioning, sampling, VAE management, loop execution and model-patcher interfaces. Comfy Kitchen provides the existing quantized/fused operators used by the optional performance experiments. MiniMax and the checkpoint/LoRA publishers provide the model weights.

- https://github.com/Comfy-Org/ComfyUI
- https://github.com/Comfy-Org/comfy-kitchen
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/lightx2v/Minimax-h3-Turbo

All weights and dependencies retain their separate licenses. This repository includes no checkpoints.

## Test media

The example reference images were generated for local evaluation. Published test images are re-encoded without embedded workflow/EXIF metadata. Only explicitly selected neutral, fully clothed synthetic-adult examples are included. Private user images, private workflow prompts and the user's phone photograph are excluded.
