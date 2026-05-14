from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainOutput:
    loss: float
    accuracy: float


def examples_to_tensors(examples: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []

    for ex in examples:
        feat = ex["state_features"]
        handcrafted = [
            feat["normalized_length"],
            feat["normalized_word_count"],
            feat["normalized_digit_count"],
            feat["has_percent"],
            feat["has_money"],
            feat["has_ratio_words"],
            feat["has_multistep_hint"],
        ]

        embedding = ex["state_embedding"]
        x = handcrafted + embedding

        xs.append(x)
        ys.append(ex["label_action_idx"])

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


def build_dataloader(examples: List[dict], batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor, y_tensor = examples_to_tensors(examples)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights_from_examples(
    examples: List[dict],
    num_actions: int,
) -> torch.Tensor:
    _, y_tensor = examples_to_tensors(examples)
    counts = torch.bincount(y_tensor, minlength=num_actions).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer=None,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> TrainOutput:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y, weight=class_weights)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == batch_y).sum().item())
        total_count += int(batch_y.size(0))
        total_loss += float(loss.item()) * batch_y.size(0)

    return TrainOutput(
        loss=total_loss / max(total_count, 1),
        accuracy=total_correct / max(total_count, 1),
    )


@torch.no_grad()
def predict_actions(model: nn.Module, examples: List[dict], device: str = "cpu") -> List[int]:
    model.eval()
    x_tensor, _ = examples_to_tensors(examples)
    x_tensor = x_tensor.to(device)

    logits = model(x_tensor)
    preds = torch.argmax(logits, dim=-1)
    return preds.detach().cpu().tolist()