import json
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import torch.nn.functional as F  # noqa N812

from domain.models.siamese_gnn import SiameseGNN
from domain.config import load_settings


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
settings = load_settings(CONFIG_PATH)

MODEL_PATH = Path(settings.models.current_model) / "GMN_v1.0.0.pt"
GRAPH_DB_DIR = Path(settings.data.master_graph)
OUTPUT_EMB_PATH = Path(settings.service.embedding_db_path) / "db_embeddings.pt"
OUTPUT_META_PATH = Path(settings.service.embedding_db_path) / "db_meta.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# --- Загрузка модели ---
model = SiameseGNN(embed_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


@torch.no_grad()
def get_embedding(graph: Data) -> torch.Tensor:
    graph = graph.to(device)
    batch = Batch.from_data_list([graph]).to(device)
    emb = model.forward_once(batch)
    emb = emb.view(-1)  # [128]
    return emb.cpu()


# --- Основная процедура ---
if __name__ == "__main__":
    graph_db = []
    embeddings = []

    for graph_path in tqdm(sorted(GRAPH_DB_DIR.glob("g_*.pt"))):
        meta_path = graph_path.with_name(graph_path.stem + "_meta.json")

        # Загружаем граф
        try:
            graph = torch.load(graph_path, weights_only=False)
            with meta_path.open("r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке {graph_path.name}: {e}")
            continue

        emb = get_embedding(graph)
        embeddings.append(emb)
        graph_db.append(
            {
                "id": meta["id"],
                "source": meta["source"],
                "num_nodes": meta["num_nodes"],
            }
        )

    # --- Сохраняем эмбеддинги и метаданные ---
    # Собираем в тензор [N, 128]
    embeddings = torch.stack(embeddings)  # [N, 128]
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2-нормализация

    torch.save(embeddings, OUTPUT_EMB_PATH)

    with OUTPUT_META_PATH.open("w") as f:
        json.dump(graph_db, f, indent=2, ensure_ascii=False)

    print(f"Сохранено {len(graph_db)} графов.")
    print(f"Эмбеддинги: {OUTPUT_EMB_PATH} (shape: {embeddings.shape})")
