#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

from typing import Optional, Union

import tqdm
import torch
import torch.nn as nn

from utils.logging_utils import AverageMeter


class TransformationTrainer:
    """Class to train and evaluate transformation models."""

    def __init__(
        self,
        old_model: Union[nn.Module, torch.jit.ScriptModule],
        new_model: Union[nn.Module, torch.jit.ScriptModule],
        side_info_model: Union[nn.Module, torch.jit.ScriptModule],
        mu_similarity: float = 1,
        mu_disc: Optional[float] = None,
        criterion_similarity: Optional[nn.Module] = None,
        criterion_disc: Optional[Union[torch.jit.ScriptModule, nn.Module]] = None,
        criterion_uncertainty: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        """Construct a TransformationTrainer module.

        :param old_model: A model that returns old embedding given x.
        :param new_model: A model that returns new embedding given x.
        :param side_info_model: A model that returns side-info given x.
        :param mu_similarity: hyperparameter for similarity loss
        :param mu_disc: hyperparameter for classification loss
        :param criterion_similarity: objective function computing the similarity between new and h(old) features.
        :param criterion_disc: objective function for the classification head for h(old).
        :param criterion_uncertainty: Uncertainty based Loss function.
        """

        self.old_model = old_model
        self.old_model.eval()
        self.new_model = new_model
        self.new_model.eval()
        self.side_info_model = side_info_model
        self.side_info_model.eval()

        self.mu_similarity = mu_similarity
        self.mu_disc = mu_disc

        self.criterion_similarity = criterion_similarity
        self.criterion_disc = criterion_disc
        self.criterion_uncertainty = criterion_uncertainty

    def compute_loss(
        self,
        new_feature: torch.Tensor,
        recycled_feature: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss for a batch.

        :param new_feature: Tensor of features computed by new model.
        :param recycled_feature: Tensor of features computed by transformation model.
        :param target: Labels tensor.
        :param sigma: Tensor of sigmas computed by transformation.
        :return: Total loss tensor.
        """
        loss = 0
        if self.criterion_similarity is not None:
            similarity_loss = self.criterion_similarity(
                new_feature.squeeze(), recycled_feature.squeeze()
            )
            if len(similarity_loss.shape) > 1:
                similarity_loss = similarity_loss.mean(dim=1)
            loss += self.mu_similarity * similarity_loss

        if self.criterion_disc is not None:
            if isinstance(self.new_model, torch.nn.DataParallel):
                fc_layer = self.new_model.module.model.fc
            else:
                fc_layer = self.new_model.model.fc
            logits = fc_layer(recycled_feature[:, : new_feature.size()[1]])
            loss_disc = self.criterion_disc(logits.squeeze(), target)
            loss += self.mu_disc * loss_disc

        if self.criterion_uncertainty:
            loss = self.criterion_uncertainty(sigma, loss)
        return loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        switch_mode_to_eval: bool,
    ) -> float:
        """Run one epoch of training.

        :param train_loader: Data loader to train the model.
        :param model: Model to be trained.
        :param optimizer: A torch optimizer object.
        :param device: Device the model is on.
        :param switch_mode_to_eval: If true model is train on eval mode!
        :return: Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")

        if switch_mode_to_eval:
            model.eval()
        else:
            model.train()

        for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)  # only needed by L_disc

            with torch.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

            recycled_feature = model(old_feature, side_info)
            sigma = recycled_feature[:, new_feature.size()[1] :]
            recycled_feature = recycled_feature[:, : new_feature.size()[1]]

            loss = self.compute_loss(new_feature, recycled_feature, target, sigma)
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses.avg

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device,
    ) -> float:
        """Run validation.

        :param val_loader: Data loader to evaluate the model.
        :param model: Model to be evaluated.
        :param device: Device the model is on.
        :return: Average loss on current epoch.
        """
        losses = AverageMeter("Loss", ":.3f")
        model.eval()

        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)  # only needed by L_disc

            with torch.no_grad():
                old_feature = self.old_model(images)
                new_feature = self.new_model(images)
                side_info = self.side_info_model(images)

                recycled_feature = model(old_feature, side_info)
                sigma = recycled_feature[:, new_feature.size()[1] :]
                recycled_feature = recycled_feature[:, : new_feature.size()[1]]

                loss = self.compute_loss(new_feature, recycled_feature, target, sigma)
                losses.update(loss.item(), images.size(0))

        return losses.avg
