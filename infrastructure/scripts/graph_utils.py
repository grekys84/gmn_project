"""Утилиты конвертации графов между PyG и NetworkX."""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import torch
from torch_geometric.data import Data


def pyg_to_networkx(data: Data) -> nx.Graph:
    """Преобразует ``torch_geometric`` граф в ``networkx.Graph``."""

    graph = nx.Graph()
    for idx in range(data.x.shape[0]):
        graph.add_node(
            int(idx), x=data.x[idx, 0].item(), y=data.x[idx, 1].item()
        )

    if hasattr(data, "edge_index") and data.edge_index is not None:
        edges: Iterable[tuple[int, int]] = (
            (int(u), int(v)) for u, v in data.edge_index.t().tolist()
        )
        graph.add_edges_from(edges)

    return graph


def networkx_to_pyg(graph: nx.Graph) -> Data:
    """Преобразует ``networkx.Graph`` в ``torch_geometric.data.Data``."""

    node_mapping: dict[int, int] = {}
    node_coords: list[list[float]] = []
    for new_id, (node_id, attrs) in enumerate(graph.nodes(data=True)):
        node_mapping[int(node_id)] = new_id
        node_coords.append([float(attrs["x"]), float(attrs["y"])])

    if node_coords:
        x = torch.tensor(node_coords, dtype=torch.float)
    else:
        x = torch.empty((0, 2), dtype=torch.float)

    edges: list[list[int]] = []
    for u, v in graph.edges():
        if u in node_mapping and v in node_mapping:
            edges.append([node_mapping[int(u)], node_mapping[int(v)]])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def remove_frame_by_longest_edges_keep_nodes(
    graph_data: Data, num_edges_to_remove: int = 4
) -> Data:
    """Удаляет из графа рёбра максимальной длины, сохраняя узлы."""

    if (
        graph_data is None
        or not hasattr(graph_data, "x")
        or graph_data.x is None
        or graph_data.x.shape[0] == 0
    ):
        return graph_data

    graph = pyg_to_networkx(graph_data)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return graph_data

    edge_lengths: list[tuple[float, int, int]] = []
    for u, v in graph.edges():
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        length = math.hypot(x2 - x1, y2 - y1)
        edge_lengths.append((length, int(u), int(v)))

    edge_lengths.sort(reverse=True)
    edges_to_remove = [(u, v) for _, u, v in edge_lengths[:num_edges_to_remove]]
    graph.remove_edges_from(edges_to_remove)

    return networkx_to_pyg(graph)

