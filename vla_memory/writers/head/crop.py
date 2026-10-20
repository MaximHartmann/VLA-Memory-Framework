"""The centre crop of the exterior picture at inference, for a policy trained with openpi's random-crop augmentation.

While it trains, openpi crops every non-wrist picture to 95 % at a random offset and resizes it back, so the model only sees
pictures zoomed by about 5 %; at inference openpi does not crop. CenterCrop is the mean of that augmentation: the centre
`fraction` of the picture, resized back (bilinear) to its size. The head server applies it behind the policy's own input
transforms (server.create_policy, --center-crop), where the pictures already have the model's size.
"""
from __future__ import annotations

import dataclasses

import numpy as np

EXTERIOR_IMAGE = "base_0_rgb"       # the model's key of the exterior camera in the policy's input dict


@dataclasses.dataclass(frozen=True)
class CenterCrop:
    """A transform of the policy's input dict: the pictures under `keys` cropped to their centre `fraction` and resized back."""

    fraction: float = 0.95
    keys: tuple[str, ...] = (EXTERIOR_IMAGE,)

    def __call__(self, data: dict) -> dict:
        """The input dict with the cropped pictures (it is changed in place, as openpi's own transforms do)."""
        images = dict(data["image"])
        for key in self.keys:
            if key in images:
                images[key] = self.crop(images[key])
        data["image"] = images
        return data

    def crop(self, picture) -> np.ndarray:
        """One picture (..., height, width, channels): its centre `fraction`, resized back to its size (uint8 stays uint8)."""
        import jax.image
        picture = np.asarray(picture)
        height, width = picture.shape[-3:-1]
        crop_height, crop_width = int(height * self.fraction), int(width * self.fraction)
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        centre = picture[..., top:top + crop_height, left:left + crop_width, :].astype(np.float32)
        resized = np.asarray(jax.image.resize(centre, (*centre.shape[:-3], height, width, centre.shape[-1]), "bilinear"))
        if picture.dtype == np.uint8:
            return np.clip(np.round(resized), 0, 255).astype(np.uint8)
        return resized.astype(picture.dtype)
