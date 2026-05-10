from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.preference.action_encoder import encode_action_features


def build_state_vector(state_features: Dict[str, float], state_embedding: List[float]) -> List[float]:
    handcrafted = [
        state_features["normalized_length"],
        state_features["normalized_word_count"],
        state_features["normalized_digit_count"],
        state_features["has_percent"],
        state_features["has_money"],
        state_features["has_ratio_words"],
        state_features["has_multistep_hint"],
    ]
    return handcrafted + state_embedding


class PreferenceDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], action_space):
        self.examples = examples
        self.action_space = action_space

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]

        state_vec = build_state_vector(
            state_features=ex["state_features"],
            state_embedding=ex["state_embedding"],
        )

        preferred_action_vec = encode_action_features(
            action_idx=ex["preferred_action_idx"],
            action_space=self.action_space,
        )

        rejected_action_vec = encode_action_features(
            action_idx=ex["rejected_action_idx"],
            action_space=self.action_space,
        )

        return {
            "state_x": torch.tensor(state_vec, dtype=torch.float32),
            "preferred_action_x": torch.tensor(preferred_action_vec, dtype=torch.float32),
            "rejected_action_x": torch.tensor(rejected_action_vec, dtype=torch.float32),
        }


def preference_collate_fn(batch):
    state_x = torch.stack([item["state_x"] for item in batch], dim=0)
    preferred_action_x = torch.stack([item["preferred_action_x"] for item in batch], dim=0)
    rejected_action_x = torch.stack([item["rejected_action_x"] for item in batch], dim=0)

    return {
        "state_x": state_x,
        "preferred_action_x": preferred_action_x,
        "rejected_action_x": rejected_action_x,
    }


def build_preference_dataloader(
    examples: List[Dict[str, Any]],
    action_space,
    batch_size: int = 32,
    shuffle: bool = True,
):
    dataset = PreferenceDataset(
        examples=examples,
        action_space=action_space,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=preference_collate_fn,
    )
    return loader


@dataclass
class PreferenceEpochResult:
    loss: float
    pair_accuracy: float


def run_preference_epoch(
    model,
    loader,
    optimizer,
    device: str,
):
    model.train()

    total_loss = 0.0
    total_pairs = 0
    total_correct_pairs = 0

    for batch in loader:
        state_x = batch["state_x"].to(device)
        preferred_action_x = batch["preferred_action_x"].to(device)
        rejected_action_x = batch["rejected_action_x"].to(device)

        preferred_scores = model(state_x, preferred_action_x)
        rejected_scores = model(state_x, rejected_action_x)

        diff = preferred_scores - rejected_scores
        loss = -torch.nn.functional.logsigmoid(diff).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = state_x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_pairs += batch_size
        total_correct_pairs += int((diff > 0).sum().item())

    avg_loss = total_loss / max(total_pairs, 1)
    pair_accuracy = total_correct_pairs / max(total_pairs, 1)

    return PreferenceEpochResult(
        loss=avg_loss,
        pair_accuracy=pair_accuracy,
    )