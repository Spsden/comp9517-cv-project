from __future__ import annotations

import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_grad_scaler(enabled: bool):
    """Construct a CUDA scaler across recent PyTorch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _batch_parts(batch):
    if len(batch) == 2:
        images, labels = batch
        return images, labels
    images, labels, _ = batch
    return images, labels


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    maximum_k = min(5, logits.shape[1])
    predictions = logits.topk(maximum_k, dim=1).indices
    matches = predictions.eq(labels.view(-1, 1))
    top1 = int(matches[:, :1].any(dim=1).sum().item())
    top5 = int(matches.any(dim=1).sum().item())
    return top1, top5


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler=None,
    use_amp: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top5_correct = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        images, labels = _batch_parts(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            logits = model(images)
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        batch_top1, batch_top5 = _topk_correct(logits.detach(), labels)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        top1_correct += batch_top1
        top5_correct += batch_top5
        progress.set_postfix(loss=f"{total_loss / total_examples:.3f}")

    return {
        "loss": total_loss / total_examples,
        "top1": top1_correct / total_examples,
        "top5": top5_correct / total_examples,
    }


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top5_correct = 0

    for batch in tqdm(loader, desc="validation", leave=False):
        images, labels = _batch_parts(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        batch_top1, batch_top5 = _topk_correct(logits, labels)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        top1_correct += batch_top1
        top5_correct += batch_top5

    return {
        "loss": total_loss / total_examples,
        "top1": top1_correct / total_examples,
        "top5": top5_correct / total_examples,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "class_to_idx": class_to_idx,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint

