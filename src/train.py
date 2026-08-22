import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "TRAINING_CONFIG",
            "configs/training_config.yaml",
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
    )

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / output_config["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_config["early_stopping_patience"]

    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        ),
        flush=True,
    )

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            json.dumps(
                {
                    "event": "epoch_completed",
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_accuracy, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_accuracy, 4),
                }
            ),
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "architecture": model_config["architecture"],
                    "num_classes": model_config["num_classes"],
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": str(checkpoint_path),
                    }
                ),
                flush=True,
            )
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                        }
                    ),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {
                "event": "training_completed",
                "best_val_loss": round(best_val_loss, 4),
            }
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()

