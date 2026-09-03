import torch


def minimum_average_displacement_error(
    samples: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        samples - targets[:, None],
        dim=-1,
    )
    return distances.mean(dim=-1).min(dim=1).values.mean()


def minimum_final_displacement_error(
    samples: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        samples - targets[:, None],
        dim=-1,
    )
    return distances[:, :, -1].min(dim=1).values.mean()


def select_best_trajectory(
    samples: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        samples - targets[:, None],
        dim=-1,
    )
    best_indices = distances.mean(dim=-1).argmin(dim=1)
    batch_indices = torch.arange(samples.shape[0], device=samples.device)
    return samples[batch_indices, best_indices]


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


def comfort_metrics(
    predictions: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocities = (
        predictions[:, 1:] - predictions[:, :-1]
    ) / dt
    accelerations = (
        velocities[:, 1:] - velocities[:, :-1]
    ) / dt
    jerks = (
        accelerations[:, 1:] - accelerations[:, :-1]
    ) / dt

    mean_acceleration = torch.linalg.vector_norm(
        accelerations,
        dim=-1,
    ).mean()
    mean_jerk = torch.linalg.vector_norm(
        jerks,
        dim=-1,
    ).mean()

    return mean_acceleration, mean_jerk
