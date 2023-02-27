#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

import models
from utils.objectives import UncertaintyLoss, LabelSmoothing


def get_model(arch_params: Dict, **kwargs) -> nn.Module:
    """Get a model given its configurations.

    :param arch_params: A dictionary containing all model parameters.
    :return: A torch model.
    """
    print("=> Creating model '{}'".format(arch_params.get("arch")))
    model = models.__dict__[arch_params.get("arch")](**arch_params)
    return model


def get_optimizer(
        model: nn.Module,
        algorithm: str,
        lr: float,
        weight_decay: float,
        momentum: Optional[float] = None,
        no_bn_decay: bool = False,
        nesterov: bool = False,
        **kwargs
) -> torch.optim.Optimizer:
    """Get an optimizer given its configurations.

    :param model: A torch model (with parameters to be trained).
    :param algorithm: String defining what optimization algorithm to use.
    :param lr: Learning rate.
    :param weight_decay: Weight decay coefficient.
    :param momentum: Momentum value.
    :param no_bn_decay: Whether to avoid weight decay for Batch Norm params.
    :param nesterov: Whether to use Nesterov update.
    :return: A torch optimizer objet.
    """
    if algorithm == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if no_bn_decay else weight_decay,
                },
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif algorithm == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    return optimizer


def get_criteria(similarity_loss: Optional[Dict] = None,
                 discriminative_loss: Optional[Dict] = None,
                 uncertainty_loss: Optional[Dict] = None,
                 ) -> Tuple[Dict[str, float], Dict[str, Callable]]:
    """Get training criteria.

    :param similarity_loss: Optional dictionary to determine similarity loss. It should have a name key with values from
    ["mse", "l1", "cosine"]. "mu_similarity" key determines coefficient of the similarity loss, with default value of 1.
    :param discriminative_loss: Optional dictionary to determine discriminative loss. Name key can take value from
    ["LabelSmoothing", "CE"]. If "LabelSmoothing" is picked, the "label_smoothing" key can be used to determine the
    value of label smoothing (from [0,1] interval).  "mu_disc" key determines coefficient of the discriminative loss.
    :param uncertainty_loss: Optional dictionary to determine whether to have uncertainty in loss. "mu_uncertainty" key
    determines coefficient of the regularization term in Bayesian uncertainty estimation formulation.

    :return: Two dictionaries. First one has coefficients: {"mu_similarity": mu_similarity, "mu_disc": mu_disc}
    second one has callable loss functions with "criterion_similarity", "criterion_disc", and "criterion_uncertainty"
    keys.
    """
    if uncertainty_loss is not None:
        criterion_uncertainty = UncertaintyLoss(**uncertainty_loss)
        reduction = "none"
    else:
        criterion_uncertainty = None
        reduction = "mean"

    if similarity_loss is not None:
        if similarity_loss.get("name") == "mse":
            criterion_similarity = nn.MSELoss(reduction=reduction)
        elif similarity_loss.get("name") == "l1":
            criterion_similarity = nn.L1Loss(reduction=reduction)
        elif similarity_loss.get("name") == "cosine":

            def similarity_cosine(x, y):
                return nn.CosineEmbeddingLoss(reduction=reduction)(
                    x.squeeze(),
                    y.squeeze(),
                    target=torch.ones(x.shape[0], device=y.device),
                )

            criterion_similarity = similarity_cosine
        else:
            raise NotImplementedError("Similarity loss not implemented!")
        mu_similarity = similarity_loss.get("mu_similarity", 1.0)
    else:
        criterion_similarity = None
        mu_similarity = None

    if discriminative_loss is not None:
        if discriminative_loss.get("name") == "LabelSmoothing":
            criterion_disc = LabelSmoothing(
                smoothing=discriminative_loss.get("label_smoothing"),
                reduction=reduction,
            )
        elif discriminative_loss.get("name") == "CE":
            criterion_disc = nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise NotImplementedError("discriminative loss not implemented")
        mu_disc = discriminative_loss.get("mu_disc")
    else:
        criterion_disc = None
        mu_disc = None

    mus_dict = {"mu_similarity": mu_similarity, "mu_disc": mu_disc}
    criteria_dict = {
        "criterion_similarity": criterion_similarity,
        "criterion_disc": criterion_disc,
        "criterion_uncertainty": criterion_uncertainty,
    }
    return mus_dict, criteria_dict
