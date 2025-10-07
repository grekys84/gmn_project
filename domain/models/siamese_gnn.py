import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from domain.models.encoder import GNNEncoder


class SiameseGNN(nn.Module):
    """Siamese Graph Neural Network для сравнения двух графов."""

    def __init__(
        self, embed_dim=128, hidden_dim=64, num_layers=4, dropout=0.2
    ):
        """
        Инициализирует Siamese GNN.

        Args:
            embed_dim: Размер эмбеддинга графа
            hidden_dim: Скрытый размер GINConv
            num_layers: Число слоёв GINConv
            dropout: Вероятность dropout для регуляризации

        """
        super().__init__()
        self.encoder = GNNEncoder(
            in_dim=2,  # координаты (x, y)
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward_once(self, g):
        """
        Прямой проход для одного графа.

        Args:
            g: PyG граф с атрибутами x, edge_index, batch
        Returns:
            torch.Tensor: Нормализованный эмбеддинг графа

        """
        x, edge_index, batch = g.x, g.edge_index, g.batch
        emb = self.encoder(x, edge_index, batch)  # [B, hidden_dim]
        emb = self.proj(emb)  # [B, embed_dim]
        return F.normalize(emb, p=2, dim=1)  # L2-норма

    def forward(self, g1, g2):
        """
        Сравнивает два графа и возвращает косинусное сходство.

        Args:
            g1: Первый граф
            g2: Второй граф
        Returns:
            torch.Tensor: Косинусное сходство между графами

        """
        z1 = self.forward_once(g1)
        z2 = self.forward_once(g2)
        return F.cosine_similarity(z1, z2)

    def get_embedding(self, batch):
        """
        Вспомогательный метод.

        Для получения нормализованного эмбеддинга одного батча графов.
        Полезен для инференса.

        Args:
            batch: Батч графов в формате PyG
        Returns:
            torch.Tensor: Нормализованные эмбеддинги графов

        """
        emb = self.encoder(batch.x, batch.edge_index, batch.batch)
        emb = self.proj(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
