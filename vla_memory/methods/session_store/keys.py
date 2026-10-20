"""The keys of a stored moment. Each maps the pictures of one moment (exterior, wrist; uint8 H x W x 3) to a unit vector.

    colour   the colour-blob fingerprint of both pictures (fingerprint_from_task): no model, a few milliseconds
    dinov2   DINOv2 ViT-S/14 (facebook/dinov2-small), the exterior picture only: the whole picture resized to 224 x 294
             (H x W, the 3:4 aspect kept, a multiple of the 14-pixel patch; no crop), ImageNet normalisation, the CLS
             token after the final layer norm (pooler_output), L2-normalised; 384 values
    siglip   SigLIP base/16 (google/siglip-base-patch16-224), exterior and wrist: each picture through the model's own
             processor (resize to 224 x 224), the pooled image embedding, L2-normalised, the two joined and divided by
             sqrt(2); 1,536 values

The learned keys need torch and transformers, imported when the first picture is encoded (require_encoders checks for them
earlier), and the weights of their models in the Hugging Face cache (HF_HOME), downloaded on first use. The colour key needs
neither. The models are frozen and loaded on first use; batches run without gradients, and torch uses `threads` CPU threads
(default 8) so that two proxies on one host do not compete for all cores.
"""
from __future__ import annotations

import importlib.util
from typing import Sequence

import numpy as np
from PIL import Image

from ...fingerprint import fingerprint_from_task
from .options import KEY_NAMES

DINO_MODEL = "facebook/dinov2-small"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
DINO_SIZE = (224, 294)                          # H x W
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
LEARNED_DIMENSIONS = {"dinov2": 384, "siglip": 1536}   # the learned keys' vector lengths (the colour key's depends on the task)
ENCODER_PACKAGES = ("torch", "transformers")    # what the learned keys need


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Every row scaled to length 1 (an all-zero row stays zero)."""
    matrix = np.asarray(matrix, np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def require_encoders(names: Sequence[str]) -> None:
    """Raise a ValueError (a usage error) when a learned key among `names` is wanted but torch or transformers is not
    installed. The colour key needs neither."""
    learned = [name for name in names if name in LEARNED_DIMENSIONS]
    missing = [package for package in ENCODER_PACKAGES if importlib.util.find_spec(package) is None]
    if learned and missing:
        raise ValueError(f"the keys {learned} need {' and '.join(missing)}: install them with the extra "
                         f"(pip install -e '.[session_store_encoders]' in the framework checkout)")


class KeyEncoder:
    """Encodes the pictures of moments by name: encode(name, moments) -> (n, D) float32 unit rows, where each moment is a list
    of pictures [exterior, wrist]. The task gives the colour key its fingerprint."""

    def __init__(self, task, device: str = "cpu", threads: int = 8, batch: int = 32):
        self.task = task
        self.fingerprint = fingerprint_from_task(task)
        self.device = device
        self.threads = threads
        self.batch = batch
        self._dino = None          # the DINOv2 model, loaded when first used
        self._siglip = None        # the SigLIP model and its image processor, loaded when first used

    # ------------------------------------------------------------------ public
    def encode(self, name: str, moments: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """The key `name` of every moment."""
        if name not in KEY_NAMES:
            raise ValueError(f"unknown key {name!r} (known: {KEY_NAMES})")
        if not len(moments):
            return np.zeros((0, self.dim(name)), np.float32)
        if name == "colour":
            return np.stack([self.fingerprint(list(pictures)) for pictures in moments]).astype(np.float32)
        if name == "dinov2":
            return self._in_batches(self._dino_rows, [pictures[0] for pictures in moments])
        exterior = self._in_batches(self._siglip_rows, [pictures[0] for pictures in moments])
        wrist = self._in_batches(self._siglip_rows, [pictures[1] for pictures in moments])
        return (np.concatenate([exterior, wrist], axis=1) / np.sqrt(2.0)).astype(np.float32)

    def dim(self, name: str) -> int:
        """The length of the key's vectors."""
        if name in LEARNED_DIMENSIONS:
            return LEARNED_DIMENSIONS[name]
        return len(self.fingerprint([np.zeros((240, 320, 3), np.uint8)] * 2))

    # ------------------------------------------------------------------ the learned keys
    def _torch(self):
        """torch, with the encoders' number of CPU threads set."""
        import torch
        torch.set_num_threads(self.threads)
        return torch

    def _in_batches(self, rows_of, pictures: list) -> np.ndarray:
        """rows_of(pictures) for `batch` pictures at a time, stacked and scaled to unit rows."""
        rows = [rows_of(pictures[start:start + self.batch]) for start in range(0, len(pictures), self.batch)]
        return unit_rows(np.concatenate(rows))

    def _dino_rows(self, pictures: list) -> np.ndarray:
        """The DINOv2 embeddings (n, 384) of the pictures, not yet normalised."""
        torch = self._torch()
        if self._dino is None:
            from transformers import AutoModel
            self._dino = AutoModel.from_pretrained(DINO_MODEL).eval().to(self.device)
        height, width = DINO_SIZE
        resized = [Image.fromarray(np.asarray(picture)[..., :3]).resize((width, height), Image.BICUBIC) for picture in pictures]
        arrays = [np.asarray(picture, np.float32) / 255.0 for picture in resized]
        batch = (np.stack(arrays) - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).to(self.device)
        with torch.inference_mode():
            return self._dino(pixel_values=tensor).pooler_output.float().cpu().numpy()

    def _siglip_rows(self, pictures: list) -> np.ndarray:
        """The SigLIP embeddings (n, 768) of the pictures, not yet normalised."""
        torch = self._torch()
        if self._siglip is None:
            from transformers import AutoImageProcessor, SiglipVisionModel
            self._siglip = (SiglipVisionModel.from_pretrained(SIGLIP_MODEL).eval().to(self.device),
                            AutoImageProcessor.from_pretrained(SIGLIP_MODEL))
        model, processor = self._siglip
        inputs = processor(images=[np.asarray(picture)[..., :3] for picture in pictures], return_tensors="pt")
        with torch.inference_mode():
            return model(pixel_values=inputs["pixel_values"].to(self.device)).pooler_output.float().cpu().numpy()
