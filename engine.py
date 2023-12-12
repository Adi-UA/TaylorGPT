from pathlib import Path

import torch
from tqdm.auto import tqdm

from util import save_model


def train_step(model, optimizer, loss_fn, x, y):
    logits = model(x)

    _, _, C = logits.shape

    logits = logits.view(-1, C)
    y = y.view(-1)
    loss = loss_fn(logits, y)

    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def evaluate_model(model, steps, loss_fn, train_data_loader, test_data_loader, device):
    with torch.inference_mode():
        train_loss_total, test_loss_total = 0, 0
        train_iter, test_iter = iter(train_data_loader), iter(test_data_loader)
        for _ in tqdm(range(steps), desc="Evaluating", leave=False):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            _, _, C = logits.shape
            logits = logits.view(-1, C)
            y = y.view(-1)
            train_loss = loss_fn(logits, y)

            x, y = next(test_iter)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            _, _, C = logits.shape
            logits = logits.view(-1, C)
            y = y.view(-1)
            test_loss = loss_fn(logits, y)

            train_loss_total += train_loss
            test_loss_total += test_loss

    return train_loss_total / steps, test_loss_total / steps


def train(
    model,
    optimizer,
    loss_fn,
    train_data_loader,
    test_data_loader,
    train_steps,
    log_interval,
    eval_steps,
    device,
):
    model.to(device)
    model.train()
    train_iter = iter(train_data_loader)
    for epoch in tqdm(range(train_steps), desc="Training"):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        train_step(model, optimizer, loss_fn, x, y)

        if (epoch == 1) or (epoch + 1) % log_interval == 0:
            model.eval()
            train_loss, test_loss = evaluate_model(
                model,
                eval_steps,
                loss_fn,
                train_data_loader,
                test_data_loader,
                device,
            )
            tqdm.write(
                f"Epoch {epoch + 1}: Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}"
            )
            model.train()
