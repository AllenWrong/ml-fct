#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

from typing import Union, Tuple, List, Optional, Callable
import copy

from sklearn.metrics import average_precision_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions.

    From https://github.com/YantaoShen/openBCT/blob/main/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_ap(
    distance_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Get pair-wise cosine distances.

    :param distance_matrix: pairwise distance matrix between embeddings of gallery and query sets, shape = (n, n)
    :param labels: labels for the query data (assuming the same as gallery), shape = (n,)

    :return: mean average precision (float)
    """
    distance_matrix = distance_matrix
    m, n = distance_matrix.shape
    assert m == n

    # Sort and find correct matches
    distance_matrix, gallery_matched_indices = torch.sort(distance_matrix, dim=1)
    distance_matrix = distance_matrix.cpu().numpy()
    gallery_matched_indices = gallery_matched_indices.cpu().numpy()

    truth_mask = labels[gallery_matched_indices] == labels[:, None]
    truth_mask = truth_mask.cpu().numpy()

    # Compute average precision for each query
    average_precisions = list()
    for query_index in range(n):

        valid_sorted_match_indices = (
            gallery_matched_indices[query_index, :] != query_index
        )
        y_true = truth_mask[query_index, valid_sorted_match_indices]
        y_score = -distance_matrix[query_index][valid_sorted_match_indices]
        if not np.any(y_true):
            continue  # if a query does not have any match, we exclude it from mAP calculation.
        average_precisions.append(average_precision_score(y_true, y_score))
    return np.mean(average_precisions)


def cosine_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, diag_only: bool = False
) -> torch.Tensor:
    """Get pair-wise cosine distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :param diag_only: if True, only diagonal of distance matrix is computed and returned.
    :return: Distance tensor between features x and y with shape (n, n) if diag_only is False. Otherwise, elementwise
    distance tensor with shape (n,).
    """
    x_norm = F.normalize(x, p=1, dim=-1)
    y_norm = F.normalize(y, p=1, dim=-1)
    if diag_only:
        return 1.0 - torch.sum(x_norm * y_norm, dim=1)
    return 1.0 - x_norm @ y_norm.T


def l2_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, diag_only: bool = False
) -> torch.Tensor:
    """Get pair-wise l2 distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :param diag_only: if True, only diagonal of distance matrix is computed and returned.
    :return: Distance tensor between features x and y with shape (n, n) if diag_only is False. Otherwise, elementwise
    distance tensor with shape (n,).
    """
    if diag_only:
        return torch.norm(x - y, dim=1, p=2)
    return torch.cdist(x, y, p=2)


def cmc_optimized(
    distmat: torch.Tensor,
    query_ids: Optional[torch.Tensor] = None,
    topk: int = 5,
) -> Tuple[float, float]:
    """Compute Cumulative Matching Characteristics metric.

    :param distmat: pairwise distance matrix between embeddings of gallery and query sets
    :param query_ids: labels for the query data. We're assuming query_ids and gallery_ids are the same.
    :param topk: parameter for top k retrieval
    :return: CMC top-1 and top-5 floats, as well as per-query top-1 and top-5 values.
    """
    distmat = copy.deepcopy(distmat)
    query_ids = copy.deepcopy(query_ids)

    distmat.fill_diagonal_(float("inf"))
    distmat_new_old_sorted, indices = torch.sort(distmat)
    labels = query_ids.unsqueeze(dim=0).repeat(query_ids.shape[0], 1)
    sorted_labels = torch.gather(labels, 1, indices)

    top1_retrieval = sorted_labels[:, 0] == query_ids
    top5_retrieval = (
        (sorted_labels[:, :topk] == query_ids.unsqueeze(1)).sum(dim=1).clamp(max=1)
    )

    top1 = top1_retrieval.sum() / query_ids.shape[0]
    top5 = top5_retrieval.sum() / query_ids.shape[0]

    return float(top1), float(top5)


def generate_feature_matrix(
    gallery_model: Union[nn.Module, torch.jit.ScriptModule],
    query_model: Union[nn.Module, torch.jit.ScriptModule],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Feature Matrix
    :param gallery_model: Model to compute gallery features.
    :param query_model: Model to compute query features.
    :param val_loader: Data loader to get gallery/query data.
    :param device: Device to use for computations.
    :param verbose: Whether to be verbose.
    :return: Three tensors gallery_features (n, d), query_features (n, d), labels (n,), where n is size of val dataset,
    and d is the embedding dimension.
    """

    gallery_model.eval()
    query_model.eval()

    gallery_model.to(device)
    query_model.to(device)

    gallery_features = []
    query_features = []
    labels = []

    iterator = tqdm.tqdm(val_loader) if verbose else val_loader

    with torch.no_grad():
        for data, label in iterator:
            data = data.to(device)
            label = label.to(device)
            gallery_feature = gallery_model(data)
            query_feature = query_model(data)

            gallery_features.append(gallery_feature.squeeze())
            query_features.append(query_feature.squeeze())

            labels.append(label)

    gallery_features = torch.cat(gallery_features)
    query_features = torch.cat(query_features)
    labels = torch.cat(labels)

    return gallery_features, query_features, labels


def get_backfilling_orders(
    backfilling_list: List[str],
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    distance_metric: Callable,
    sigma: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, str]]:
    """Compute backfilling ordering.

    :param backfilling_list: list of desired backfilling orders from ["random", "distance", "sigma']
    :param query_features: Tensor of query features with shape (n, d), where n is dataset size and d is embedding dim.
    :param gallery_features: Tensor of gallery features with shape (n, d).
    :param distance_metric: Callable to compute distance between features.
    :param sigma: Tensor of computed sigmas with shape (n,).

    :return: List of (ordering, ordering_name) tuples. ordering is a permutation of [0, 1, ..., n-1] determining the
    backfilling ordering, and ordering_name is the name of ordering. For example, if ordering=[3, 0, 2, 1] it means
    first backfill gallery data at index 3, followed by elements at indices 0, 2, and 1, respectively.
    """
    orderings_list = []
    n = query_features.shape[0]
    for ordering_name in backfilling_list:
        if ordering_name.lower() == "random":
            ordering = torch.randperm(n)
        elif ordering_name.lower() == "distance":
            distances = distance_metric(
                query_features.cpu(), gallery_features.cpu(), diag_only=True
            )
            ordering = torch.argsort(distances, descending=True)
        elif ordering_name.lower() == "sigma":
            assert sigma.numel() == n
            ordering = torch.argsort(sigma, dim=0, descending=True)
        else:
            print(f"{ordering_name} is not implemented for backfilling")
            continue

        # Sanity checks:
        assert torch.unique(ordering).shape == ordering.shape
        assert ordering.min() == 0
        assert ordering.max() == n - 1
        orderings_list.append((ordering, ordering_name))

    return orderings_list


def cmc_evaluate(
    gallery_model: Union[nn.Module, torch.jit.ScriptModule],
    query_model: Union[nn.Module, torch.jit.ScriptModule],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    distance_metric_name: str,
    verbose: bool = False,
    compute_map: bool = False,
    backfilling: Optional[int] = None,
    backfilling_list: List[str] = ["random"],
    backfilling_result_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Run CMC and mAP evaluations.

    :param gallery_model: Model to compute gallery features.
    :param query_model: Model to compute query features.
    :param val_loader: Data loader to get gallery/query data.
    :param device: Device to use for computations.
    :param distance_metric_name: Name of distance metric to use. Choose from ['l2', 'cosine'].
    :param verbose: Whether to be verbose.
    :param compute_map: Whether to compute mean average precision.
    :param backfilling: Number of intermediate backfilling steps. None means 0 intermediate backfilling. In this case
    only results for 0% backfilling will be computed.
    :param backfilling_list: list of desired backfilling orders from ["random", "distance", "sigma']. Default is
    ['random'].
    :param backfilling_result_path: path to save backfilling results.
    """
    distance_map = {"l2": l2_distance_matrix, "cosine": cosine_distance_matrix}
    distance_metric = distance_map.get(distance_metric_name.lower())

    print("Generating Feature Matrix")
    gallery_features, query_features, labels = generate_feature_matrix(
        gallery_model, query_model, val_loader, device, verbose
    )

    # (Possibly) Split gallery features and sigmas
    embedding_dim = query_features.shape[1]
    sigma = gallery_features[:, embedding_dim:].squeeze()  # (n,)
    sigma = None if sigma.numel() == 0 else sigma
    gallery_features = gallery_features[:, :embedding_dim]  # (n, d)

    n = query_features.shape[0]  # dataset size
    backfilling = backfilling if backfilling is not None else -1

    orderings_list = get_backfilling_orders(
        backfilling_list=backfilling_list,
        query_features=query_features,
        gallery_features=gallery_features,
        distance_metric=distance_metric,
        sigma=sigma,
    )

    backfilling_results = {}
    for ordering, ordering_name in orderings_list:
        print(f"\nBackfilling evaluation with {ordering_name} ordering.")
        gallery_features_reordered = copy.deepcopy(gallery_features[ordering]).cpu()
        query_features_reordered = copy.deepcopy(query_features[ordering]).cpu()
        labels_reordered = copy.deepcopy(labels[ordering]).cpu()

        # Lists to store top1, top5, and ,mAP
        outputs = {"CMC-top1": [], "CMC-top5": [], "mAP": []}

        iterator = (
            tqdm.tqdm(range(backfilling + 2)) if verbose else range(backfilling + 2)
        )

        for i in iterator:
            if backfilling >= 0:
                cutoff_index = (i * n) // (backfilling + 1)
            else:
                cutoff_index = 0
            backfilling_mask = torch.zeros((n, 1), dtype=torch.bool)
            backfilling_mask[torch.arange(n) < cutoff_index] = True
            backfilled_gallery = torch.where(
                backfilling_mask, query_features_reordered, gallery_features_reordered
            )

            distmat = distance_metric(query_features_reordered, backfilled_gallery)

            top1, top5 = cmc_optimized(
                distmat=distmat,
                query_ids=labels_reordered,
                topk=5,
            )

            if compute_map:
                mean_ap_out = mean_ap(distance_matrix=distmat, labels=labels_reordered)
            else:
                mean_ap_out = None

            outputs["CMC-top1"].append(top1)
            outputs["CMC-top5"].append(top5)
            outputs["mAP"].append(mean_ap_out)

        backfilling_results[ordering_name] = outputs

        for metric_name, metric_data in outputs.items():
            print_partial_backfilling(metric_data, metric_name)

    if backfilling_result_path is not None:
        np.save(backfilling_result_path, backfilling_results)


def print_partial_backfilling(data: List, metric_name: str) -> None:
    """Print partial backfilling results.

    :param data: list of floats for a given metric in [0,1] range.
    :param metric_name: name of the metric.
    """
    print(f"*** {metric_name} ***:")
    data_str_list = ["{:.2f}".format(100 * x) for x in data]
    print(" -> ".join(data_str_list))
    if len(data) > 1:
        print("AUC: {:.2f} %".format(100 * np.mean(data)))
    print("\n")
