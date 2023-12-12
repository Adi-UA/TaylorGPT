from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from data import (
    get_dataloader,
    get_encoder_decoder_fn,
    load_data,
    load_vocab,
    train_test_split,
)
from engine import train
from lightning_engine import LitTransformer
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
    device = get_device()
    trainer = L.Trainer(
        max_steps=train_steps,
        limit_train_batches=train_steps,
        limit_val_batches=eval_steps,
        val_check_interval=eval_steps,
        deterministic=True,
        logger=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        precision="16",
        accelerator=device,
    )

    with trainer.init_module():
        model = TransformerDecoderModel(
            vocab_size=len(vocab),
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            head_size=head_size,
            dropout=dropout,
            embed_size=embed_size,
        )

        # Initialize LightningModule
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        lit_model = LitTransformer(model, loss_fn, optimizer)

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=test_data_loader,
    )

    device = get_device()

    # Save model
    save_model(lit_model.model, save_path)
    print(f"Saved model to {save_path}")
