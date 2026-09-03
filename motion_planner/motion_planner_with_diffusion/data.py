import torch
from torch.utils.data import TensorDataset


SCENARIO_NAMES = (
    "straight",
    "acceleration_or_braking",
    "left_or_right_turn",
    "leader_following",
)


def create_multimodal_dataset(
    num_samples: int,
    num_agents: int,
    history_steps: int,
    future_steps: int,
    dt: float,
    coordinate_scale: float = 10.0,
    route_steps: int = 20,
) -> TensorDataset:
    total_steps = history_steps + future_steps
    times = torch.arange(total_steps, dtype=torch.float32) * dt

    scenario_types = torch.arange(num_samples) % len(SCENARIO_NAMES)
    scenario_types = scenario_types[torch.randperm(num_samples)]

    leader_x = torch.rand(num_samples, 1) * 20.0 - 10.0
    gaps = torch.rand(num_samples, 1) * 5.0 + 7.0
    agent_indices = torch.arange(num_agents).reshape(1, num_agents)
    initial_x = leader_x - gaps * agent_indices
    initial_y = torch.rand(num_samples, num_agents) * 2.0 - 1.0

    scene_speeds = torch.rand(num_samples, 1) * 7.0 + 5.0
    speed_changes = torch.rand(num_samples, num_agents) * 2.0 - 1.0
    speeds = scene_speeds + speed_changes

    x = initial_x.unsqueeze(-1) + speeds.unsqueeze(-1) * times
    y = initial_y.unsqueeze(-1).expand(-1, -1, total_steps).clone()

    future_start = (history_steps - 1) * dt
    future_time = torch.clamp(times - future_start, min=0.0)

    acceleration_signs = torch.randint(0, 2, (num_samples, 1, 1)) * 2 - 1
    acceleration_strengths = torch.rand(num_samples, 1, 1) * 1.5 + 0.5
    acceleration_mask = (scenario_types == 1).reshape(-1, 1, 1)
    x = x + (
        0.5
        * acceleration_signs
        * acceleration_strengths
        * future_time.reshape(1, 1, -1) ** 2
        * acceleration_mask
    )

    turn_directions = torch.randint(0, 2, (num_samples, 1, 1)) * 2 - 1
    turn_strengths = torch.rand(num_samples, 1, 1) * 1.0 + 0.5
    turn_mask = (scenario_types == 2).reshape(-1, 1, 1)
    y = y + (
        0.5
        * turn_directions
        * turn_strengths
        * future_time.reshape(1, 1, -1) ** 2
        * turn_mask
    )

    leader_brake_start = (history_steps - 2) * dt
    reaction_delays = torch.arange(num_agents).reshape(1, num_agents, 1) * 2 * dt
    braking_time = torch.clamp(
        times.reshape(1, 1, -1)
        - leader_brake_start
        - reaction_delays,
        min=0.0,
    )
    braking_strengths = -(torch.rand(num_samples, 1, 1) * 1.5 + 1.0)
    following_mask = (scenario_types == 3).reshape(-1, 1, 1)
    x = x + (
        0.5
        * braking_strengths
        * braking_time**2
        * following_mask
    )

    trajectories = torch.stack((x, y), dim=-1)

    route_fractions = torch.linspace(0.0, 1.0, route_steps).reshape(1, -1)
    route_lengths = scene_speeds * future_steps * dt
    route_x = route_lengths * route_fractions
    route_time = route_x / scene_speeds
    route_y = (
        0.5
        * turn_directions.squeeze(-1)
        * turn_strengths.squeeze(-1)
        * route_time**2
        * (scenario_types == 2).reshape(-1, 1)
    )
    routes = torch.stack((route_x, route_y), dim=-1)

    angles = torch.rand(num_samples) * 2.0 * torch.pi
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    rotation_matrices = torch.stack(
        (
            torch.stack((cosines, -sines), dim=-1),
            torch.stack((sines, cosines), dim=-1),
        ),
        dim=1,
    )
    trajectories = torch.matmul(
        trajectories,
        rotation_matrices[:, None].transpose(-2, -1),
    )
    routes = torch.matmul(
        routes,
        rotation_matrices.transpose(-2, -1),
    )

    scene_origins = torch.rand(num_samples, 1, 1, 2) * 20.0 - 10.0
    trajectories = trajectories + scene_origins

    histories = trajectories[:, :, :history_steps]
    futures = trajectories[:, :, history_steps:]
    origin = histories[:, :1, -1:, :]

    base_difficulty = torch.tensor([0.1, 0.5, 0.7, 1.0])
    difficulty_scores = base_difficulty[scenario_types]
    difficulty_scores = difficulty_scores + (12.0 - gaps.squeeze(1)) / 20.0

    return TensorDataset(
        (histories - origin) / coordinate_scale,
        (futures - origin) / coordinate_scale,
        routes / coordinate_scale,
        scenario_types,
        difficulty_scores,
    )
