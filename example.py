import sys
from omegavit.main import create_advanced_vit, train_step
import torch
from loguru import logger

def main():
    """Main training function."""
    logger.info("Starting training setup")

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = create_advanced_vit().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.05
    )

    # Example input for testing
    batch_size = 8
    example_input = torch.randn(batch_size, 3, 224, 224).to(device)
    example_labels = torch.randint(0, 1000, (batch_size,)).to(device)

    logger.info("Running forward pass with example input")
    output = model(example_input)
    logger.info(f"Output shape: {output.shape}")

    # Example training step
    loss = train_step(
        model, optimizer, (example_input, example_labels), device
    )
    logger.info(f"Example training step loss: {loss:.4f}")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        "advanced_vit.log",
        rotation="500 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(sys.stdout, level="INFO")

    main()
