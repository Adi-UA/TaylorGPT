from pathlib import Path

import torch
import yaml

from data import get_encoder_decoder_fn, load_vocab
from model import TransformerDecoderModel
from util import get_device, load_model

if __name__ == "__main__":
    # Load config and data
    script_dir = Path(__file__).parent
    with open(script_dir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed = config["seed"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    n_heads = config["n_heads"]
    head_size = config["head_size"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    embed_size = config["embed_size"]
    save_path = config["save_path"]

    vocab = load_vocab()
    (
        _,
        decode,
    ) = get_encoder_decoder_fn(vocab)

    # Load model
    device = get_device()
    model = TransformerDecoderModel(
        vocab_size=len(vocab),
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        head_size=head_size,
        dropout=dropout,
        embed_size=embed_size,
    ).to(device)

    model = load_model(model, save_path, device)

    # Generate text
    model.eval()
    with torch.inference_mode():
        x = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(x, 500)[0].tolist()
        print("Generated text:")
        print("=" * 20)
        print(decode(out).strip())
        print("=" * 20)
