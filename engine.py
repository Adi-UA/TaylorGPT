from pathlib import Path

import torch
from tqdm.auto import tqdm

from util import save_model


def sample_batch(data, block_size, batch_size):
    batch_starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[start : start + block_size] for start in batch_starts])
    y = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in batch_starts]
    )

    return x, y


def train_step(model, optimizer, loss_fn, train_data, batch_size, block_size, device):
    model.train()

    x, y = sample_batch(train_data, block_size, batch_size)
    x, y = x.to(device), y.to(device)

    logits = model(x)
    _, _, C = logits.shape
    logits = logits.view(-1, C)
    y = y.view(-1)
    loss = loss_fn(logits, y)

    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def evaluate_model(
    model, eval_epochs, loss_fn, train_data, test_data, batch_size, block_size, device
):
    model.eval()
    model.to(device)
    with torch.inference_mode():
        train_loss_total, test_loss_total = 0, 0
        for _ in tqdm(range(eval_epochs), desc="Evaluating", leave=False):
            x, y = sample_batch(train_data, block_size, batch_size)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            _, _, C = logits.shape
            logits = logits.view(-1, C)
            y = y.view(-1)
            train_loss = loss_fn(logits, y)

            x, y = sample_batch(test_data, block_size, batch_size)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            _, _, C = logits.shape
            logits = logits.view(-1, C)
            y = y.view(-1)
            test_loss = loss_fn(logits, y)

            train_loss_total += train_loss
            test_loss_total += test_loss

    return train_loss_total / eval_epochs, test_loss_total / eval_epochs


def train(
    model,
    optimizer,
    loss_fn,
    train_data,
    test_data,
    epochs,
    batch_size,
    block_size,
    log_interval,
    save_interval,
    eval_epochs,
    device,
):
    model = model.to(device)
    for epoch in tqdm(range(epochs), desc="Training"):
        train_step(
            model, optimizer, loss_fn, train_data, batch_size, block_size, device
        )

        if (epoch == 1) or (epoch + 1) % log_interval == 0:
            train_loss, test_loss = evaluate_model(
                model,
                eval_epochs,
                loss_fn,
                train_data,
                test_data,
                batch_size,
                block_size,
                device,
            )
            tqdm.write(
                f"Epoch {epoch + 1}: Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}"
            )

        if (epoch + 1) % save_interval == 0:
            checkpoint_dir = Path(f"checkpoints")
            save_path = checkpoint_dir / f"model_{epoch + 1}.pt"
            save_model(model, save_path)
            tqdm.write(f"Saved model to {save_path}")
