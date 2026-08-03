from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Grad-CAM for a convolutional layer of a classification model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_indices is None:
            target_indices = logits.argmax(dim=1)
        target_indices = target_indices.to(logits.device)
        scores = logits.gather(1, target_indices.view(-1, 1)).sum()
        scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        flattened = cam.flatten(start_dim=1)
        minimum = flattened.min(dim=1).values[:, None, None]
        maximum = flattened.max(dim=1).values[:, None, None]
        cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
        return cam.detach(), logits.detach()

    def close(self) -> None:
        self._hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def denormalise_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    image = tensor * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


def overlay_cam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colour_map: str = "jet",
) -> np.ndarray:
    import matplotlib.pyplot as plt

    heatmap = plt.get_cmap(colour_map)(cam)[..., :3]
    overlay = (1 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0, 1)

