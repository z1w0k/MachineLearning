import torch
import torch.nn as nn


class TrajectoryDiffusion(nn.Module):
    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        route_steps: int = 20,
        diffusion_steps: int = 30,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.future_steps = future_steps
        self.diffusion_steps = diffusion_steps

        self.history_projection = nn.Linear(2, hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, history_steps, hidden_size) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )
        self.agent_norm = nn.LayerNorm(hidden_size)
        self.route_projection = nn.Linear(2, hidden_size)
        self.route_positional_embedding = nn.Parameter(
            torch.randn(1, route_steps, hidden_size) * 0.02
        )
        route_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.route_encoder = nn.TransformerEncoder(
            route_encoder_layer,
            num_layers=1,
        )
        self.time_embedding = nn.Embedding(
            diffusion_steps,
            hidden_size,
        )

        network_input_size = future_steps * 2 + hidden_size * 3
        self.noise_network = nn.Sequential(
            nn.Linear(network_input_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, future_steps * 2),
        )

        schedule_steps = torch.linspace(0, diffusion_steps, diffusion_steps + 1)
        alpha_bars = torch.cos(
            (
                schedule_steps / diffusion_steps + 0.008
            ) / 1.008 * torch.pi / 2
        ) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = betas.clamp(max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_condition(
        self,
        histories: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = histories.shape[0]
        num_agents = histories.shape[1]
        flat_histories = histories.reshape(
            batch_size * num_agents,
            histories.shape[2],
            2,
        )
        tokens = (
            self.history_projection(flat_histories)
            + self.positional_embedding
        )
        encoded = self.history_encoder(tokens)
        agent_embeddings = encoded[:, -1, :].reshape(
            batch_size,
            num_agents,
            -1,
        )
        interaction, _ = self.agent_attention(
            agent_embeddings,
            agent_embeddings,
            agent_embeddings,
        )
        agent_context = self.agent_norm(agent_embeddings + interaction)
        ego_context = agent_context[:, 0]

        route_tokens = (
            self.route_projection(routes)
            + self.route_positional_embedding
        )
        encoded_route = self.route_encoder(route_tokens)
        route_context = encoded_route[:, -1]

        return torch.cat((ego_context, route_context), dim=-1)

    def predict_noise(
        self,
        noisy_futures: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_context = self.time_embedding(timesteps)

        network_input = torch.cat(
            (
                noisy_futures.flatten(start_dim=1),
                condition,
                time_context,
            ),
            dim=-1,
        )
        predicted_noise = self.noise_network(network_input)

        return predicted_noise.reshape(
            noisy_futures.shape[0],
            self.future_steps,
            2,
        )

    def forward(
        self,
        noisy_futures: torch.Tensor,
        histories: torch.Tensor,
        routes: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode_condition(histories, routes)
        return self.predict_noise(noisy_futures, condition, timesteps)

    def add_noise(
        self,
        futures: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bars = self.alphas_cumprod[timesteps].reshape(-1, 1, 1)

        return (
            torch.sqrt(alpha_bars) * futures
            + torch.sqrt(1.0 - alpha_bars) * noise
        )

    @torch.no_grad()
    def sample(
        self,
        histories: torch.Tensor,
        routes: torch.Tensor,
        num_samples: int = 6,
    ) -> torch.Tensor:
        batch_size = histories.shape[0]
        condition = self.encode_condition(histories, routes)
        condition = condition.repeat_interleave(num_samples, dim=0)

        predictions = torch.randn(
            batch_size * num_samples,
            self.future_steps,
            2,
            device=histories.device,
        )

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full(
                (batch_size * num_samples,),
                step,
                device=histories.device,
                dtype=torch.long,
            )
            predicted_noise = self.predict_noise(
                predictions,
                condition,
                timesteps,
            )

            alpha_bar = self.alphas_cumprod[step]
            predicted_original = (
                predictions
                - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            predicted_original = predicted_original.clamp(-4.0, 4.0)

            if step > 0:
                previous_alpha_bar = self.alphas_cumprod[step - 1]
                predictions = (
                    torch.sqrt(previous_alpha_bar) * predicted_original
                    + torch.sqrt(1.0 - previous_alpha_bar)
                    * predicted_noise
                )
            else:
                predictions = predicted_original

        return predictions.reshape(
            batch_size,
            num_samples,
            self.future_steps,
            2,
        )
