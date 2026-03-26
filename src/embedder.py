import os

import torch
from sentence_transformers import SentenceTransformer


class QuestionEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 64,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def encode_questions(self, questions: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            questions,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        return embeddings.float().cpu()


def build_embedding_cache(
    questions: list[str],
    cache_path: str,
    model_name: str,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    batch_size: int = 64,
    force_rebuild: bool = False,
) -> torch.Tensor:
    if os.path.exists(cache_path) and not force_rebuild:
        embeddings = torch.load(cache_path, map_location="cpu")
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        return embeddings.float()

    embedder = QuestionEmbedder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        batch_size=batch_size,
    )
    embeddings = embedder.encode_questions(questions)
    torch.save(embeddings, cache_path)
    return embeddings.float()