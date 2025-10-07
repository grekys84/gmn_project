import torch

try:  # optional dependency
    from torch_geometric.data import Data, Batch
except Exception:
    try:
        from torch_geometric_stub.data import Data, Batch
    except Exception:
        Data = Batch = None

def warmup_model(
    model: torch.nn.Module,
    runs: int = 3,
    num_nodes: int = 10,
    num_edges: int = 20,
    feature_dim: int = 2
) -> None:
    """
    Выполнить несколько проходов, чтобы разогреть модель.

    Параметры
    ----------
    model:
        Экземпляр SiameseGNN.
    runs:
        Количество итераций для прогрева.
    num_nodes:
        Количество узлов в фиктивном графе.
    num_edges:
        Количество ребер в фиктивном графе.
    feature_dim:
        Размерность координат узлов (например 2 для координат x,y).
    """
    if Data is None or Batch is None:
        raise RuntimeError("torch_geometric is required for model warmup")

    device = next(model.parameters()).device

    x = torch.rand(num_nodes, feature_dim, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    dummy = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([dummy])

    model.eval()
    with torch.no_grad():
        for _ in range(runs):
            model.get_embedding(batch)
