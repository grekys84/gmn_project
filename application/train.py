import json
import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from application.datasets.graph_pair_dataset import GraphPairDataset
from domain.config.settings import settings
from domain.models.losses import CosineContrastiveLoss, TripletLoss
from domain.models.siamese_gnn import SiameseGNN


# --- Основные параметры ---
# Загружаем путь из конфига
PAIRS_JSON = settings.data.pair_file
MODEL_SAVE_DIR = settings.models.current_model
VAL_ERRORS_DIR = settings.models.val_errors
EPOCHS = settings.training.epochs
BATCH_SIZE = settings.training.batch_size
LR = settings.training.lr
PATIENCE = settings.training.early_stopping_patience
BASE_MARGIN = settings.training.margin
LOSS_NAME = settings.training.loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


# Путь к файлу лога
log_file = settings.logging.log_dir / "TrainSiameseGNN.log"

# === Настройка логгера ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("TrainSiameseGNN")

# ===================================================


def graph_pair_collate(batch):
    """
    Collate-функция для пар графов с ID.

    Args:
        batch: Батч данных из датасета
    Returns:
        tuple: Кортеж из батчей графов, меток и индексов

    """
    g1_list, g2_list, labels, idxs = zip(
        *batch, strict=False
    )  # strict=False по умолчанию
    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    labels = torch.stack(labels).view(-1)
    idxs = torch.tensor(idxs)
    return batch1, batch2, labels, idxs


def save_top_errors(similarities, labels, idxs, epoch, top_n=10):
    """
    Сохраняет топ-N ошибок (False Positives и False Negatives) для анализа.

    Args:
        similarities (Tensor): Сходства моделей, размер [batch_size].
        labels (Tensor): Истинные метки, размер [batch_size].
        idxs (Tensor): Индексы пар.
        epoch (int): Номер эпохи.
        top_n (int): Количество ошибок для сохранения.

    """
    sims = similarities.cpu().numpy()
    lbls = labels.cpu().numpy()
    ids = idxs.cpu().numpy()

    # False Positives: pred=1 (sim>0.5), label=0
    fp_mask = (lbls == 0) & (sims > 0.5)
    fn_mask = (lbls == 1) & (sims < 0.5)

    false_positives = sorted(
        [(ids[i], sims[i]) for i in np.where(fp_mask)[0]],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    false_negatives = sorted(
        [(ids[i], sims[i]) for i in np.where(fn_mask)[0]], key=lambda x: x[1]
    )[:top_n]

    errors = {
        "epoch": epoch,
        "false_positives": [
            {"id": int(i), "sim": float(s)} for i, s in false_positives
        ],
        "false_negatives": [
            {"id": int(i), "sim": float(s)} for i, s in false_negatives
        ],
    }

    save_path = VAL_ERRORS_DIR / f"val_errors_epoch_{epoch}.json"
    with save_path.open("w") as f:
        json.dump(errors, f, indent=4)

    logger.info(f"Топ-{top_n} FP/FN сохранены: {save_path}")


def main():
    """Основная функция обучения Siamese GNN."""
    # --- Датасет ---
    dataset = GraphPairDataset(
        json_file=PAIRS_JSON,
        augmentation_config=settings.augmentation,
        return_index=True,
    )

    # Делим на train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=graph_pair_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=graph_pair_collate,
    )

    # --- Модель ---
    model = SiameseGNN(embed_dim=128, hidden_dim=64, num_layers=4).to(device)
    if LOSS_NAME == "contrastive":
        criterion = CosineContrastiveLoss(margin=BASE_MARGIN)
    elif LOSS_NAME == "triplet":
        criterion = TripletLoss(margin=BASE_MARGIN)
    else:
        raise ValueError(f"Unknown loss type: {LOSS_NAME}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # --- Логи ---
    train_losses, val_losses, auc_scores = [], [], []
    best_val_loss = float("inf")
    best_model_path = MODEL_SAVE_DIR / "GMN_v1.0.0.pt"
    early_stopping_counter = 0

    logger.info("Начало обучения Siamese GNN...")

    for epoch in range(1, EPOCHS + 1):
        # === TRAIN ===
        model.train()
        total_loss = 0
        for batch1, batch2, labels, idxes in tqdm(
            train_loader, desc=f"Эпоха {epoch}/{EPOCHS}"
        ):
            batch1, batch2, labels, idxes = (
                batch1.to(device),
                batch2.to(device),
                labels.to(device),
                idxes.to(device),
            )
            optimizer.zero_grad()

            similarities = model(batch1, batch2)  # [-1, 1]

            loss = criterion(similarities, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === VALIDATION ===
        model.eval()
        val_loss = 0
        all_similarities = []
        all_labels = []
        all_idxs = []

        with torch.no_grad():
            for batch1, batch2, labels, idxes in val_loader:
                batch1, batch2, labels, idxes = (
                    batch1.to(device),
                    batch2.to(device),
                    labels.to(device),
                    idxes.to(device),
                )
                similarities = model(batch1, batch2)

                loss = criterion(similarities, labels.float())
                val_loss += loss.item()

                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_idxs.extend(idxes.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ROC-AUC
        y_true = np.array(all_labels)
        y_scores = np.array(all_similarities)
        auc_score = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc_score)

        logger.info(
            f"Эпоха {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"AUC={auc_score:.4f}, margin={criterion.margin:.2f}"
        )

        # --- Сохранение топ-ошибок ---
        save_top_errors(
            torch.tensor(all_similarities),
            torch.tensor(all_labels),
            torch.tensor(all_idxs),
            epoch,
            top_n=10,
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0

            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Новая лучшая модель сохранена: {best_model_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE:
                logger.info(f"Early Stopping сработал на {epoch} эпохе.")
                break

    logger.info("Обучение завершено!")

    # === JSON-логирование метрик ===
    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "auc": auc_scores,
    }
    metrics_path = MODEL_SAVE_DIR / "training_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Метрики сохранены в JSON: {metrics_path}")


if __name__ == "__main__":
    main()
