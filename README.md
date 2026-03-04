# thesis

notes
maybe thats usefull for mamba:
https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies

dependencies = ["flash-attn", "torch"]
_[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]

[tool.uv.extra-build-variables]
flash-attn = { FLASH_ATTENTION_SKIP_CUDA_BUILD = "TRUE" }_


[project]
name = "project"
version = "0.1.0"
description = "..."
readme = "README.md"
requires-python = ">=3.12"
dependencies = ["mamba", "torch", "torchvision", "torchaudio"]

[tool.uv.extra-build-dependencies]
mamba = [{ requirement = "torch", match-runtime = true }]
torchvision = [{ requirement = "torch", match-runtime = true }]
torchaudio = [{ requirement = "torch", match-runtime = true }]