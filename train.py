from pathlib import Path

import torch
import yaml

from data import (
    get_dataloader,
    get_encoder_decoder_fn,
    load_data,
    load_vocab,
    train_test_split,
)
from engine import train
from model import TransformerDecoderModel
from util import get_device, save_model, seed_everything

if __name__ == "__main__":
    # Load config and data
    script_dir = Path(__file__).parent
    with open(script_dir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed = config["seed"]
    lr = config["lr"]
    train_steps = config["train_steps"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    log_interval = config["log_interval"]
    n_heads = config["n_heads"]
    head_size = config["head_size"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    embed_size = config["embed_size"]
    eval_steps = config["eval_steps"]
    save_path = Path(config["save_path"])

    data = load_data()
    vocab = load_vocab()
    encode, _ = get_encoder_decoder_fn(vocab)
    encoded_data = encode(data)
    encoded_train_data, encoded_test_data = train_test_split(encoded_data, 0.9)
    train_data_loader = get_dataloader(
        encoded_train_data, block_size, batch_size, shuffle=True
    )
    test_data_loader = get_dataloader(
        encoded_test_data, block_size, batch_size, shuffle=False
    )

    # Build model
    seed_everything(config["seed"])
    model = TransformerDecoderModel(
        vocab_size=len(vocab),
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        head_size=head_size,
        dropout=dropout,
        embed_size=embed_size,
    )

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = get_device()
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        train_steps=train_steps,
        log_interval=log_interval,
        eval_steps=eval_steps,
        device=device,
    )

    # Save model
    save_model(model, save_path)
    print(f"Saved model to {save_path}")
