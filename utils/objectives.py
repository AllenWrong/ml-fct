#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

import torch
from torch import nn


class UncertaintyLoss(nn.Module):
    """Loss that takes a loss vector and a Sigma vector and applies uncertainty to it"""

    def __init__(
        self, mu_uncertainty: float = 0.5, sigma_dim: int = 1, **kwargs
    ) -> None:
        super(UncertaintyLoss, self).__init__()
        self.mu = mu_uncertainty
        self.sigma_dim = sigma_dim

    def forward(self, sigma: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        :param sigma: feature vector (N x sigma_dim) that includes the sigma vector
        :param loss:    loss vector (N x 1) where N is the batch size

        :return:    return either a Nx1 or a 1x1 dimensional loss vector
        """
        batch = sigma.size(0)
        reg = self.mu * sigma.view(batch, -1).mean(dim=1)
        loss_value = 0.5 * torch.exp(-sigma.squeeze()) * loss + reg

        return loss_value.mean()


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        """Construct LabelSmoothing module.

        :param smoothing: label smoothing factor. Default is 0.0.
        :param reduction: reduction method to use from ["mean", "none"]. Default is "mean".
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        assert reduction in ["mean", "none"]
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Logits tensor.
        :param target: Ground truth target classes.
        :return: Loss tensor.
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss
