import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from data import SCENARIO_NAMES, create_dataset
from metrics import (
    average_displacement_error,
    collision_rate,
    comfort_metrics,
    final_displacement_error,
    route_deviation,
)
from model import TrajectoryTransformer


torch.manual_seed(42)

history_steps = 10
future_steps = 10
route_steps = 20
dt = 0.2
num_epochs = 25

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", device)

dataset = create_dataset(
    num_samples=2_000,
    num_agents=4,
    history_steps=history_steps,
    future_steps=future_steps,
    dt=dt,
    route_steps=route_steps,
)

scenario_types = dataset.tensors[3]
for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
    count = (scenario_types == scenario_index).sum().item()
    print(f"{scenario_name}: {count}")

split_generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = random_split(
    dataset,
    [1_600, 400],
    generator=split_generator,
)
validation_loader = DataLoader(validation_dataset, batch_size=64)


def create_train_loader(hard_mining: bool) -> DataLoader:
    if not hard_mining:
        return DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
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
    return DataLoader(
        train_dataset,
        batch_size=64,
        sampler=sampler,
    )


def evaluate_model(model: TrajectoryTransformer) -> dict:
    model.eval()
    totals = {
        "ade": 0.0,
        "fde": 0.0,
        "collision_rate": 0.0,
        "route_deviation": 0.0,
        "acceleration": 0.0,
        "jerk": 0.0,
    }
    scenario_ade = torch.zeros(len(SCENARIO_NAMES))
    scenario_fde = torch.zeros(len(SCENARIO_NAMES))
    scenario_counts = torch.zeros(len(SCENARIO_NAMES))

    with torch.no_grad():
        for histories, futures, routes, scenarios, _ in validation_loader:
            histories = histories.to(device)
            futures = futures.to(device)
            routes = routes.to(device)
            ego_targets = futures[:, 0]
            predictions = model(histories, routes)

            acceleration, jerk = comfort_metrics(predictions, dt)
            batch_size = histories.shape[0]

            totals["ade"] += (
                average_displacement_error(predictions, ego_targets).item()
                * batch_size
            )
            totals["fde"] += (
                final_displacement_error(predictions, ego_targets).item()
                * batch_size
            )
            totals["collision_rate"] += (
                collision_rate(predictions, futures[:, 1:]).item()
                * batch_size
            )
            totals["route_deviation"] += (
                route_deviation(predictions, routes).item()
                * batch_size
            )
            totals["acceleration"] += acceleration.item() * batch_size
            totals["jerk"] += jerk.item() * batch_size

            distances = torch.linalg.vector_norm(
                predictions - ego_targets,
                dim=-1,
            ).cpu()
            for scenario_index in range(len(SCENARIO_NAMES)):
                scenario_mask = scenarios == scenario_index
                count = scenario_mask.sum().item()
                if count > 0:
                    scenario_ade[scenario_index] += (
                        distances[scenario_mask].mean(dim=-1).sum()
                    )
                    scenario_fde[scenario_index] += (
                        distances[scenario_mask, -1].sum()
                    )
                    scenario_counts[scenario_index] += count

    for metric_name in totals:
        totals[metric_name] /= len(validation_dataset)

    totals["scenario_ade"] = scenario_ade / scenario_counts
    totals["scenario_fde"] = scenario_fde / scenario_counts
    return totals


def train_experiment(
    experiment_name: str,
    hard_mining: bool,
) -> tuple[TrajectoryTransformer, dict]:
    torch.manual_seed(42)
    train_loader = create_train_loader(hard_mining)
    model = TrajectoryTransformer(
        history_steps=history_steps,
        future_steps=future_steps,
        route_steps=route_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print("-" * 60)
    print(experiment_name)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for histories, futures, routes, _, _ in train_loader:
            histories = histories.to(device)
            futures = futures.to(device)
            routes = routes.to(device)

            predictions = model(histories, routes)
            loss = criterion(predictions, futures[:, 0])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * histories.shape[0]

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={total_loss / len(train_dataset):.4f}"
            )

    results = evaluate_model(model)
    return model, results


random_model, random_results = train_experiment(
    "Random sampling",
    hard_mining=False,
)
hard_model, hard_results = train_experiment(
    "Hard scenario mining",
    hard_mining=True,
)

print("-" * 60)
print("Comparison")
for name, results in (
    ("Random", random_results),
    ("Hard mining", hard_results),
):
    print(
        f"{name}: "
        f"ADE={results['ade']:.4f}, "
        f"FDE={results['fde']:.4f}, "
        f"collision_rate={results['collision_rate']:.4f}, "
        f"route_deviation={results['route_deviation']:.4f}, "
        f"acceleration={results['acceleration']:.4f}, "
        f"jerk={results['jerk']:.4f}"
    )

print("-" * 60)
print("Hard-mining metrics by scenario")
for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
    print(
        f"{scenario_name}: "
        f"ADE={hard_results['scenario_ade'][scenario_index]:.4f}, "
        f"FDE={hard_results['scenario_fde'][scenario_index]:.4f}"
    )
