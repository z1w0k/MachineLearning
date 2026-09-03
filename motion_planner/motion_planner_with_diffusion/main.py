from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from data import SCENARIO_NAMES, create_multimodal_dataset
from metrics import (
    collision_rate,
    comfort_metrics,
    minimum_average_displacement_error,
    minimum_final_displacement_error,
    route_deviation,
    select_best_trajectory,
)
from model import TrajectoryDiffusion


torch.manual_seed(42)

history_steps = 10
future_steps = 10
route_steps = 20
num_agents = 4
diffusion_steps = 30
num_samples = 6
coordinate_scale = 10.0
dt = 0.2
num_epochs = 30

project_directory = Path(__file__).resolve().parent
checkpoint_path = project_directory / "best_diffusion.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", device)

dataset = create_multimodal_dataset(
    num_samples=4_000,
    num_agents=num_agents,
    history_steps=history_steps,
    future_steps=future_steps,
    dt=dt,
    coordinate_scale=coordinate_scale,
    route_steps=route_steps,
)

scenario_types = dataset.tensors[3]
for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
    count = (scenario_types == scenario_index).sum().item()
    print(f"{scenario_name}: {count}")

split_generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = random_split(
    dataset,
    [3_200, 800],
    generator=split_generator,
)

train_indices = torch.tensor(train_dataset.indices)
difficulty_scores = dataset.tensors[4][train_indices]
sample_weights = 1.0 + 3.0 * difficulty_scores
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
)

model = TrajectoryDiffusion(
    history_steps=history_steps,
    future_steps=future_steps,
    route_steps=route_steps,
    diffusion_steps=diffusion_steps,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
best_validation_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for histories, futures, routes, _, _ in train_loader:
        histories = histories.to(device)
        futures = futures.to(device)
        routes = routes.to(device)
        ego_futures = futures[:, 0]

        timesteps = torch.randint(
            0,
            diffusion_steps,
            (histories.shape[0],),
            device=device,
        )
        noise = torch.randn_like(ego_futures)
        noisy_futures = model.add_noise(ego_futures, noise, timesteps)
        predicted_noise = model(
            noisy_futures,
            histories,
            routes,
            timesteps,
        )
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += loss.item() * histories.shape[0]

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for histories, futures, routes, _, _ in validation_loader:
            histories = histories.to(device)
            futures = futures.to(device)
            routes = routes.to(device)
            ego_futures = futures[:, 0]

            timesteps = torch.randint(
                0,
                diffusion_steps,
                (histories.shape[0],),
                device=device,
            )
            noise = torch.randn_like(ego_futures)
            noisy_futures = model.add_noise(ego_futures, noise, timesteps)
            predicted_noise = model(
                noisy_futures,
                histories,
                routes,
                timesteps,
            )
            loss = criterion(predicted_noise, noise)
            validation_loss += loss.item() * histories.shape[0]

    train_loss = train_loss / len(train_dataset)
    validation_loss = validation_loss / len(validation_dataset)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), checkpoint_path)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}"
        )

model.load_state_dict(
    torch.load(checkpoint_path, weights_only=True, map_location=device)
)
model.eval()

total_min_ade = 0.0
total_min_fde = 0.0
total_collision_rate = 0.0
total_route_deviation = 0.0
total_acceleration = 0.0
total_jerk = 0.0
total_endpoint_spread = 0.0
scenario_ade = torch.zeros(len(SCENARIO_NAMES))
scenario_fde = torch.zeros(len(SCENARIO_NAMES))
scenario_counts = torch.zeros(len(SCENARIO_NAMES))

with torch.no_grad():
    for histories, futures, routes, scenarios, _ in validation_loader:
        histories = histories.to(device)
        futures = futures.to(device) * coordinate_scale
        routes = routes.to(device)

        samples = model.sample(
            histories,
            routes,
            num_samples=num_samples,
        ) * coordinate_scale
        routes_meters = routes * coordinate_scale
        ego_targets = futures[:, 0]
        selected = select_best_trajectory(samples, ego_targets)
        acceleration, jerk = comfort_metrics(selected, dt)

        batch_size = histories.shape[0]
        total_min_ade += (
            minimum_average_displacement_error(samples, ego_targets).item()
            * batch_size
        )
        total_min_fde += (
            minimum_final_displacement_error(samples, ego_targets).item()
            * batch_size
        )
        total_collision_rate += (
            collision_rate(selected, futures[:, 1:]).item()
            * batch_size
        )
        total_route_deviation += (
            route_deviation(selected, routes_meters).item()
            * batch_size
        )
        total_acceleration += acceleration.item() * batch_size
        total_jerk += jerk.item() * batch_size

        endpoints = samples[:, :, -1]
        endpoint_center = endpoints.mean(dim=1, keepdim=True)
        endpoint_spread = torch.linalg.vector_norm(
            endpoints - endpoint_center,
            dim=-1,
        ).mean()
        total_endpoint_spread += endpoint_spread.item() * batch_size

        distances = torch.linalg.vector_norm(
            samples - ego_targets[:, None],
            dim=-1,
        ).cpu()
        min_ade_per_scene = distances.mean(dim=-1).min(dim=1).values
        min_fde_per_scene = distances[:, :, -1].min(dim=1).values

        for scenario_index in range(len(SCENARIO_NAMES)):
            scenario_mask = scenarios == scenario_index
            count = scenario_mask.sum().item()
            if count > 0:
                scenario_ade[scenario_index] += (
                    min_ade_per_scene[scenario_mask].sum()
                )
                scenario_fde[scenario_index] += (
                    min_fde_per_scene[scenario_mask].sum()
                )
                scenario_counts[scenario_index] += count

validation_size = len(validation_dataset)
print("-" * 60)
print(f"Diffusion samples per scene: {num_samples}")
print(f"minADE={total_min_ade / validation_size:.4f}")
print(f"minFDE={total_min_fde / validation_size:.4f}")
print(f"collision_rate={total_collision_rate / validation_size:.4f}")
print(f"route_deviation={total_route_deviation / validation_size:.4f}")
print(f"acceleration={total_acceleration / validation_size:.4f}")
print(f"jerk={total_jerk / validation_size:.4f}")
print(f"endpoint_spread={total_endpoint_spread / validation_size:.4f}")

print("-" * 60)
print("Metrics by scenario")
for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
    print(
        f"{scenario_name}: "
        f"minADE={scenario_ade[scenario_index] / scenario_counts[scenario_index]:.4f}, "
        f"minFDE={scenario_fde[scenario_index] / scenario_counts[scenario_index]:.4f}"
    )

print("Best model saved to", checkpoint_path)
