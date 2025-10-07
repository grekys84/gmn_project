import torch
import torch.nn as nn

class CosineContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для косинусного сходства.
    labels: 1 - положительная пара (похожие графы), 0 - отрицательная пара
    """
    def __init__(self, margin=0.5):
        """
        margin: порог для отрицательных пар (0 < margin <= 2).
                Для косинусного сходства лучше выбирать ~0.5-1.0
        """
        super().__init__()
        self.margin = margin

    def forward(self, similarities, labels):
        """
        similarities: тензор [B], значения косинусного сходства [-1, 1]
        labels: тензор [B], 1 - positive, 0 - negative
        """
        # Косинусное "расстояние": 0 для идентичных векторов, 2 для противоположных
        distances = 1 - similarities  # [0..2]

        # Положительные пары: минимизируем расстояние (чем ближе, тем лучше)
        positive_loss = labels * distances.pow(2)

        # Отрицательные пары: расстояние должно быть >= margin
        # Если расстояние меньше margin, штрафуем
        negative_loss = (1 - labels) * torch.clamp(self.margin - distances, min=0).pow(2)

        loss = 0.5 * (positive_loss + negative_loss)
        return loss.mean()


class TripletLoss(nn.Module):
    """Триплетная функция потерь на основе евклидовой дистанции."""

    def __init__(self, margin: float = 1.0):
        """
        margin: минимальная разница между положительной и отрицательной парами.
        """
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: тензоры эмбеддингов [B, D]
        """
        return self.loss_fn(anchor, positive, negative)
