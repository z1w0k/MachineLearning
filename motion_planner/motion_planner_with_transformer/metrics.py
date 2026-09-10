import torch


def average_displacement_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(predictions - targets, dim=-1)
    return distances.mean()


def final_displacement_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(predictions - targets, dim=-1)
    return distances[:, -1].mean()


def collision_rate(
    ego_predictions: torch.Tensor,
    other_futures: torch.Tensor,
    safety_distance: float = 2.0,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        ego_predictions[:, None] - other_futures,
        dim=-1,
    )
    collisions = (distances < safety_distance).any(dim=(1, 2))
    return collisions.float().mean()


def route_deviation(
    predictions: torch.Tensor,
    routes: torch.Tensor,
) -> torch.Tensor:
    distances_to_route = torch.cdist(predictions, routes)
    return distances_to_route.min(dim=-1).values.mean()

