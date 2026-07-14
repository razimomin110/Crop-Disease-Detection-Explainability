"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability.

Hooks into the last convolutional layer of the backbone to produce a heatmap
showing which spatial regions of the input image most influenced the prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM visualisation for a given model and target convolutional layer.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    target_layer : torch.nn.Module
        The convolutional layer to hook (e.g., ``model.get_target_layer()``).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward & backward hooks
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ───────────────────────────────────────────────────
    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ── Generate heatmap ────────────────────────────────────────────────
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Pre-processed image tensor of shape (1, C, H, W).
        target_class : int, optional
            Class index to visualise. If ``None``, uses the predicted class.

        Returns
        -------
        heatmap : np.ndarray
            2-D array (H, W) with values in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients and backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only positive contributions

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    # ── Overlay on original image ───────────────────────────────────────
    @staticmethod
    def overlay_heatmap(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend the Grad-CAM heatmap onto the original image.

        Parameters
        ----------
        original_image : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.
        heatmap : np.ndarray
            2-D heatmap in [0, 1].
        alpha : float
            Blending weight for the heatmap.

        Returns
        -------
        blended : np.ndarray
            RGB image with overlay, uint8.
        """
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = np.uint8(alpha * heatmap_color + (1 - alpha) * original_image)
        return blended

    # ── Cleanup ─────────────────────────────────────────────────────────
    def remove_hooks(self):
        """Remove forward/backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()
