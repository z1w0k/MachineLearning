import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        route_steps: int = 20,
        embedding_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.future_steps = future_steps
        self.input_projection = nn.Linear(2, embedding_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, history_steps, embedding_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.agent_norm = nn.LayerNorm(embedding_dim)
        self.route_projection = nn.Linear(2, embedding_dim)
        self.route_positional_embedding = nn.Parameter(
            torch.randn(1, route_steps, embedding_dim) * 0.02
        )
        route_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.route_encoder = nn.TransformerEncoder(
            route_encoder_layer,
            num_layers=1,
        )
        self.output_layer = nn.Linear(
            embedding_dim * 2,
            future_steps * 2,
        )

    def forward(
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
            self.input_projection(flat_histories)
            + self.positional_embedding
        )
        encoded = self.encoder(tokens)
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

        planning_context = torch.cat(
            (ego_context, route_context),
            dim=-1,
        )
        predictions = self.output_layer(planning_context)

        return predictions.reshape(
            batch_size,
            self.future_steps,
            2,
        )
