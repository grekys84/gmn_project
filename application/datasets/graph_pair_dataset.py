import json
import math
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from infrastructure.scripts.graph_utils import (
    remove_frame_by_longest_edges_keep_nodes,
)
from domain.config.settings import settings

# # Используем настройки по умолчанию
# DEFAULT_AUGMENTATION = settings.augmentation_flags


class GraphPairDataset(Dataset):
    """
    Набор данных из пар графиков с координатами, выраженными в миллиметрах.

    Параметры
    ----------
     json_file : str, optional
        Путь к JSON-файлу, описывающему пары графов.
    noise_scale : float, default 0.1
        Стандартное отклонение применяемого гауссовского шума в миллиметрах.
    rotation_max_deg : float, default 10.0
        Максимальный угол поворота в градусах.
    scale_jitter : float, default 0.05
        Относительное колебание масштаба.
    flip_prob : float, default 0.0
        Вероятность переворачивания вдоль каждой оси.
    edge_dropout : float, default 0.1
        Вероятность удаления ребер.
    node_dropout : float, default 0.05
        Вероятность удаления узлов.
    return_index : bool, default False
        Если `True`, то `__getitem__` также возвращает индекс пары
    """

    def __init__(
        self,
        json_file=None,
        augmentation_config=None,
        return_index=False,
    ):
        """Инициализирует датасет, загружает пары графов из JSON."""
        super().__init__()
        self.return_index = return_index

        # Используем переданный конфиг или дефолтный
        self.cfg_flags = settings.augmentation_flags
        self.cfg_params = settings.augmentation

        from infrastructure.utils.prepare_pairs import OUTPUT_FILE

        if json_file is None:
            json_file = OUTPUT_FILE

        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        """Возвращает количество пар в датасете."""
        return len(self.data)

    def _augment_graph(self, graph: Data, apply_extra_aug=False) -> Data:
        """Применяет аугментации к графу."""
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()

        if apply_extra_aug:
            # 1. Случайное смещение
            if self.cfg_params.noise_scale and self.cfg_params.noise_scale > 0:
                noise = torch.randn_like(x) * self.cfg_params.noise_scale
                x += noise

            # 2. Случайный поворот
            if (
                self.cfg_flags.apply_rotation
                and self.cfg_params.rotation_max_deg > 0
            ):
                angle = math.radians(
                    random.uniform(
                        -self.cfg_params.rotation_max_deg,
                        self.cfg_params.rotation_max_deg,
                    )
                )
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rot_matrix = torch.tensor(
                    [[cos_a, -sin_a], [sin_a, cos_a]],
                    dtype=x.dtype,
                    device=x.device,
                )
                center = x.mean(dim=0, keepdim=True)
                x = (x - center) @ rot_matrix.T + center

            # 3. Случайное масштабирование
            if self.cfg_flags.apply_scale and self.cfg_params.scale_jitter > 0:
                scale_factor = 1.0 + random.uniform(
                    -self.cfg_params.scale_jitter, self.cfg_params.scale_jitter
                )
                center = x.mean(dim=0, keepdim=True)
                x = (x - center) * scale_factor + center

        # 4. Отражение — только если flip_prob > 0 И apply_extra_aug=True
        if self.cfg_flags.apply_flip and self.cfg_params.flip_prob > 0:
            if random.random() < self.cfg_params.flip_prob:
                x[:, 0] = -x[:, 0]  # flip X
            if random.random() < self.cfg_params.flip_prob:
                x[:, 1] = -x[:, 1]  # flip Y

        # 5. Дропаут рёбер — только при аугментации
        if (
            apply_extra_aug
            and self.cfg_flags.apply_edge_dropout
            and self.cfg_params.edge_dropout > 0
            and edge_index.size(1) > 0
        ):
            mask = (
                torch.rand(edge_index.size(1)) > self.cfg_params.edge_dropout
            )
            edge_index = edge_index[:, mask]

        return Data(x=x, edge_index=edge_index)

    def __getitem__(self, idx):
        """
        Возвращает пару графов, метку и, опционально, индекс.

        Args:
            idx (int): Индекс пары.

        Returns:
            tuple: Кортеж из:
                - g1 (Data): Первый граф.
                - g2 (Data): Второй граф (возможно, аугментированный).
                - label (Tensor): Метка пары (1 — совпадение, 0 — нет).
                - idx (int, optional): Индекс пары, если `return_index=True`.

        """
        item = self.data[idx]
        g1_path = item.get("graph1") or item.get("g1")
        g2_path = item.get("graph2") or item.get("g2")
        label = torch.tensor(item.get("label", 1), dtype=torch.long)
        remove_frame_flag = item.get("remove_frame", False)
        augment_flag = item.get("augment", False)

        try:
            g1 = torch.load(g1_path, weights_only=False)
            g2 = torch.load(g2_path, weights_only=False)
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки {g1_path} или {g2_path}: {e}")
            empty = Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)))
            return empty, empty, torch.tensor(0)

        # Удаляем рамку (только рёбра!)
        if remove_frame_flag:
            g1 = remove_frame_by_longest_edges_keep_nodes(
                g1, num_edges_to_remove=4
            )
            g2 = remove_frame_by_longest_edges_keep_nodes(
                g2, num_edges_to_remove=4
            )

        # Аугментация: только g2, если нужно
        g1 = self._augment_graph(g1, apply_extra_aug=False)  # Только flip_prob
        g2 = self._augment_graph(g2, apply_extra_aug=augment_flag)

        if self.return_index:
            return g1, g2, label, idx
        return g1, g2, label
