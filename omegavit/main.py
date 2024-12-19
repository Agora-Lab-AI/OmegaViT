import math
from typing import Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


@dataclass
class ViTConfig:
    """Configuration class for Vision Transformer model."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    num_experts: int = 8
    expert_capacity: int = 32
    num_layers: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    num_classes: int = 1000
    patch_size: int = 16
    ssm_state_size: int = 16


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        """
        Generate rotary embeddings.

        Args:
            t: Sequence length
            device: Target device

        Returns:
            Rotary embeddings tensor
        """
        seq = torch.arange(t, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, "n d -> 1 n 1 d")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    q_embed = (q * pos_emb.cos()) + (rotate_half(q) * pos_emb.sin())
    k_embed = (k * pos_emb.cos()) + (rotate_half(k) * pos_emb.sin())
    return q_embed, k_embed


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention module."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
        )

        self.q = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.qkv_bias,
        )
        self.k = nn.Linear(
            config.hidden_size, self.head_dim, bias=config.qkv_bias
        )
        self.v = nn.Linear(
            config.hidden_size, self.head_dim, bias=config.qkv_bias
        )

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config.attention_dropout_rate)

        logger.debug(
            f"Initialized MultiQueryAttention with {self.num_attention_heads} heads"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        logger.debug(f"MultiQueryAttention input shape: {x.shape}")

        q = self.q(x).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        )
        k = self.k(x).view(batch_size, seq_len, 1, self.head_dim)
        v = self.v(x).view(batch_size, seq_len, 1, self.head_dim)

        # Expand k, v for all heads
        k = k.expand(-1, -1, self.num_attention_heads, -1)
        v = v.expand(-1, -1, self.num_attention_heads, -1)

        # Apply rotary embeddings
        pos_emb = self.rotary_emb(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)

        # Attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b s h d -> b s (h d)")

        logger.debug(f"MultiQueryAttention output shape: {out.shape}")
        return out


class StateSpaceModel(nn.Module):
    """State Space Model (SSM) layer."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_size = config.ssm_state_size

        self.A = nn.Parameter(
            torch.randn(self.state_size, self.state_size)
        )
        self.B = nn.Parameter(
            torch.randn(self.state_size, self.hidden_size)
        )
        self.C = nn.Parameter(
            torch.randn(self.hidden_size, self.state_size)
        )

        logger.debug(
            f"Initialized SSM with state size: {self.state_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing discrete-time state space model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        logger.debug(f"SSM input shape: {x.shape}")

        # Initialize hidden state
        h = torch.zeros(batch_size, self.state_size, device=x.device)
        outputs = []

        # Recurrent processing
        for t in range(seq_len):
            h = torch.tanh(
                torch.matmul(h, self.A.T)
                + torch.matmul(x[:, t], self.B.T)
            )
            out = torch.matmul(h, self.C.T)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)
        logger.debug(f"SSM output shape: {out.shape}")
        return out


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.capacity = config.expert_capacity

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        config.hidden_size, config.hidden_size * 4
                    ),
                    nn.GELU(),
                    nn.Linear(
                        config.hidden_size * 4, config.hidden_size
                    ),
                )
                for _ in range(config.num_experts)
            ]
        )

        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        logger.debug(
            f"Initialized MoE with {self.num_experts} experts"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with top-k routing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        original_shape = x.shape
        batch_size, seq_len, hidden_size = original_shape

        # Reshape input
        x = x.view(-1, hidden_size)

        # Get expert weights and indices
        gates = self.gate(x)
        gates = F.softmax(gates, dim=-1)
        _, indices = torch.topk(gates, k=2, dim=-1)

        # Dispatch to experts
        final_output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            idx_mask = indices[:, 0] == expert_idx
            if idx_mask.any():
                expert_input = x[idx_mask]
                expert_output = self.experts[expert_idx](expert_input)
                final_output[idx_mask] = expert_output

        output = final_output.view(original_shape)
        logger.debug(f"MoE output shape: {output.shape}")
        return output


class TransformerBlock(nn.Module):
    """Transformer block with alternating attention and SSM layers."""

    def __init__(self, config: ViTConfig, use_ssm: bool = False):
        super().__init__()
        self.use_ssm = use_ssm
        self.norm1 = nn.LayerNorm(config.hidden_size)

        if use_ssm:
            self.attention = StateSpaceModel(config)
        else:
            self.attention = MultiQueryAttention(config)

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.moe = MixtureOfExperts(config)
        self.dropout = nn.Dropout(config.dropout_rate)

        logger.debug(f"Initialized TransformerBlock (SSM: {use_ssm})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        logger.debug(f"TransformerBlock input shape: {x.shape}")

        # First sublayer (attention or SSM)
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x

        # Second sublayer (MoE)
        residual = x
        x = self.norm2(x)
        x = self.moe(x)
        x = self.dropout(x)
        x = residual + x

        logger.debug(f"TransformerBlock output shape: {x.shape}")
        return x


class FlexiblePatchEmbedding(nn.Module):
    """Flexible patch embedding that can handle any input resolution."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.norm = nn.LayerNorm(config.hidden_size)

        logger.debug(
            f"Initialized FlexiblePatchEmbedding with patch size: {self.patch_size}"
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, num_patches, hidden_size)
                - Tuple of (height, width) in patches
        """
        logger.debug(f"PatchEmbedding input shape: {x.shape}")

        # Dynamic handling of input resolution
        b, c, h, w = x.shape
        h_patches = math.ceil(h / self.patch_size)
        w_patches = math.ceil(w / self.patch_size)

        # Pad if needed
        pad_h = h_patches * self.patch_size - h
        pad_w = w_patches * self.patch_size - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Project and reshape
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)

        logger.debug(f"PatchEmbedding output shape: {x.shape}")
        return x, (h_patches, w_patches)


class OmegaViT(nn.Module):
    """
    Advanced Vision Transformer with multi-query attention, rotary embeddings,
    state space models, and mixture of experts.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        logger.info("Initializing OmegaViT")

        # Patch embedding
        self.patch_embed = FlexiblePatchEmbedding(config)

        # Position embedding will be handled by rotary embeddings in attention

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    use_ssm=(i + 1) % 3
                    == 0,  # Use SSM every 3rd layer
                )
                for i in range(config.num_layers)
            ]
        )

        # Final layers
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

        logger.info(
            f"Initialized OmegaViT with {config.num_layers} layers"
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        logger.debug(f"OmegaViT input shape: {x.shape}")

        # Patch embedding
        x, (h_patches, w_patches) = self.patch_embed(x)
        batch_size, num_patches, _ = x.shape

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            logger.debug(f"Processing block {i}")
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final classification
        x = self.norm(x)
        x = self.head(x)

        logger.debug(f"OmegaViT output shape: {x.shape}")
        return x

    @torch.no_grad()
    def get_attention_maps(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get attention maps for visualization.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            List of attention maps from each non-SSM block
        """
        attention_maps = []

        # Get patch embeddings
        x, _ = self.patch_embed(x)

        # Collect attention maps from each non-SSM block
        for block in self.blocks:
            if not block.use_ssm:
                # Store attention weights before dropout
                block.attention.dropout.p = 0
                q = block.attention.q(x).view(
                    x.shape[0],
                    -1,
                    block.attention.num_attention_heads,
                    block.attention.head_dim,
                )
                k = block.attention.k(x).view(
                    x.shape[0], -1, 1, block.attention.head_dim
                )
                k = k.expand(
                    -1, -1, block.attention.num_attention_heads, -1
                )

                # Apply rotary embeddings
                pos_emb = block.attention.rotary_emb(
                    x.shape[1], x.device
                )
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

                # Calculate attention weights
                scale = block.attention.head_dim**-0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = F.softmax(attn, dim=-1)
                attention_maps.append(attn)

        return attention_maps


# Example usage and training setup
def create_advanced_vit(num_classes: int = 1000) -> OmegaViT:
    """
    Create an instance of the Advanced Vision Transformer.

    Args:
        num_classes: Number of output classes

    Returns:
        Configured OmegaViT model
    """
    logger.info("Creating OmegaViT instance")

    config = ViTConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_experts=8,
        expert_capacity=32,
        num_layers=12,
        num_classes=num_classes,
        patch_size=16,
        ssm_state_size=16,
    )

    return OmegaViT(config)


def train_step(
    model: OmegaViT,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    """
    Single training step.

    Args:
        model: The ViT model
        optimizer: PyTorch optimizer
        batch: Tuple of (images, labels)
        device: Target device

    Returns:
        Loss value
    """
    model.train()
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

