import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch_geometric.nn import GINConv, global_add_pool
# try:
#     from torch_geometric.nn import GINConv, global_add_pool  # type: ignore
# except Exception:  # pragma: no cover - fallback to stub
#     from torch_geometric_stub.nn import GINConv, global_add_pool


class GNNEncoder(nn.Module):
    """
    Графовый энкодер на основе GIN (Graph Isomorphism Network).

    Для получения фиксированного по размеру эмбеддинга графа.

    Архитектура состоит из нескольких слоёв GINConv
     с ReLU-активациями и dropout'ом.
    В конце применяется глобальное суммарное пулингование (global_add_pool)
    для агрегации узловых представлений в один вектор,
    представляющий весь граф.

    Используется как часть Siamese GNN для сопоставления графов
     по косинусному сходству.
    """

    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.2):
        """
        Инициализирует GNN-энкодер.

        Args:
            in_dim (int): Размерность входных признаков узлов.
            hidden_dim (int): Размерность скрытых слоёв (эмбеддингов).
            num_layers (int, optional): Количество GIN-слоёв. По умолчанию 3.
            dropout (float, optional): Вероятность dropout'а в слоях.
            По умолчанию 0.2.

        """
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        """
        Прямой проход через GNN.

        Args:
            x (Tensor): Входные признаки узлов, форма [num_nodes, in_dim].
            edge_index (LongTensor): Индексы рёбер, форма [2, num_edges].
            batch (LongTensor): Индексы графов в батче, форма [num_nodes].

        Returns:
            Tensor: Эмбеддинги графов, форма [batch_size, hidden_dim].

        """
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.fc_out(x)
