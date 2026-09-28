"""Pretrain the GPT model on the tokenized personal corpus (Raschka, ch. 5).

Usage:
    spllm-train --config configs/tiny.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from spllm import GPTModel, load_config
from spllm.data import create_dataloader, load_tokens
from spllm.generate import generate_text
from spllm.utils import get_device, save_checkpoint, set_seed


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


@torch.no_grad()
def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(loader), len(loader))
    total = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / num_batches


def evaluate(model, train_loader, val_loader, device, eval_iters):
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, eval_iters)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iters)
    model.train()
    return train_loss, val_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--sample-prompt", default="Today I")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg, tcfg = cfg.model, cfg.train
    set_seed(tcfg.seed)
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(tcfg.data_dir)
    train_loader = create_dataloader(
        load_tokens(data_dir / "train.bin"),
        batch_size=tcfg.batch_size,
        max_length=mcfg.context_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        load_tokens(data_dir / "val.bin"),
        batch_size=tcfg.batch_size,
        max_length=mcfg.context_length,
        shuffle=False,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise SystemExit("Not enough training tokens for one batch; add more data or reduce batch_size.")

    model = GPTModel(mcfg).to(device)
    print(f"Parameters: {model.num_parameters():,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
    )

    global_step = 0
    for epoch in range(tcfg.num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % tcfg.eval_freq == 0:
                train_loss, val_loss = evaluate(model, train_loader, val_loader, device, tcfg.eval_iters)
                print(f"Ep {epoch + 1} step {global_step:06d} | train {train_loss:.3f} | val {val_loss:.3f}")

        print(f"--- Epoch {epoch + 1} sample ---")
        print(generate_text(model, args.sample_prompt, max_new_tokens=40))
        save_checkpoint(Path(tcfg.out_dir) / "model.pt", model, optimizer, epoch=epoch + 1, step=global_step)

    print(f"Saved checkpoint to {Path(tcfg.out_dir) / 'model.pt'}")


if __name__ == "__main__":
    main()
