# OmegaViT: A State-of-the-Art Vision Transformer with Multi-Query Attention, State Space Modeling, and Mixture of Experts

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)




[![PyPI version](https://badge.fury.io/py/omegavit.svg)](https://badge.fury.io/py/omegavit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Agora-Lab-AI/OmegaViT/workflows/build/badge.svg)](https://github.com/Agora-Lab-AI/OmegaViT/actions)
[![Documentation Status](https://readthedocs.org/projects/omegavit/badge/?version=latest)](https://omegavit.readthedocs.io/en/latest/?badge=latest)

OmegaViT (ΩViT) is a cutting-edge vision transformer architecture that combines multi-query attention, rotary embeddings, state space modeling, and mixture of experts to achieve superior performance across various computer vision tasks. The model can process images of any resolution while maintaining computational efficiency.

## Key Features

- **Flexible Resolution Processing**: Handles arbitrary input image sizes through adaptive patch embedding
- **Multi-Query Attention (MQA)**: Reduces computational complexity while maintaining model expressiveness
- **Rotary Embeddings**: Enables better modeling of relative positions and spatial relationships
- **State Space Models (SSM)**: Integrates efficient sequence modeling every third layer
- **Mixture of Experts (MoE)**: Implements conditional computation for enhanced model capacity
- **Comprehensive Logging**: Built-in loguru integration for detailed execution tracking
- **Shape-Aware Design**: Continuous tensor shape tracking for reliable processing

## Architecture

```mermaid
flowchart TB
    subgraph Input
        img[Input Image]
    end
    
    subgraph PatchEmbed[Flexible Patch Embedding]
        conv[Convolution]
        norm1[LayerNorm]
        conv --> norm1
    end
    
    subgraph TransformerBlocks[Transformer Blocks x12]
        subgraph Block1[Block n]
            direction TB
            mqa[Multi-Query Attention]
            ln1[LayerNorm]
            moe1[Mixture of Experts]
            ln2[LayerNorm]
            ln1 --> mqa --> ln2 --> moe1
        end
        
        subgraph Block2[Block n+1]
            direction TB
            mqa2[Multi-Query Attention]
            ln3[LayerNorm]
            moe2[Mixture of Experts]
            ln4[LayerNorm]
            ln3 --> mqa2 --> ln4 --> moe2
        end
        
        subgraph Block3[Block n+2 SSM]
            direction TB
            ssm[State Space Model]
            ln5[LayerNorm]
            moe3[Mixture of Experts]
            ln6[LayerNorm]
            ln5 --> ssm --> ln6 --> moe3
        end
    end
    
    subgraph Output
        gap[Global Average Pooling]
        classifier[Classification Head]
    end
    
    img --> PatchEmbed --> TransformerBlocks --> gap --> classifier
```

## Multi-Query Attention Detail

```mermaid
flowchart LR
    input[Input Features]
    
    subgraph MQA[Multi-Query Attention]
        direction TB
        q[Q Linear]
        k[K Linear]
        v[V Linear]
        rotary[Rotary Embeddings]
        attn[Attention Weights]
        
        input --> q & k & v
        q & k --> rotary
        rotary --> attn
        attn --> v
    end
    
    MQA --> output[Output Features]

```

## Installation

```bash
pip install omegavit
```

## Quick Start

```python
import torch
from omegavit import create_advanced_vit

# Create model
model = create_advanced_vit(num_classes=1000)

# Example forward pass
batch_size = 8
x = torch.randn(batch_size, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [8, 1000]
```

## Model Configurations

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 768 | Dimension of transformer layers |
| num_attention_heads | 12 | Number of attention heads |
| num_experts | 8 | Number of expert networks in MoE |
| expert_capacity | 32 | Tokens per expert in MoE |
| num_layers | 12 | Number of transformer blocks |
| patch_size | 16 | Size of image patches |
| ssm_state_size | 16 | Hidden state size in SSM |

## Performance

*Note: Benchmarks coming soon*

## Citation

If you use OmegaViT in your research, please cite:

```bibtex
@article{omegavit2024,
  title={OmegaViT: A State-of-the-Art Vision Transformer with Multi-Query Attention, State Space Modeling, and Mixture of Experts},
  author={Agora Lab},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the Agora Lab AI team and the open-source community for their valuable contributions and feedback.
