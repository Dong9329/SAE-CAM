import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    losses: dict,
                    device: torch.device):
    """
    训练一个 epoch
    Args:
        model: PyTorch 模型
        loader: DataLoader
        optimizer: 优化器
        losses: {"ce": CrossEntropyLoss, "scl": SupConLoss}
        device: cuda/cpu
    """
    model.train()
    log = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "ort": 0.0, "dyn": 0.0}

    for batch in tqdm(loader, desc="train", leave=False):
        x, y = batch["x"].to(device), batch["y"].to(device)

        out = model(x)
        logits = out["logits"]
        z = out["z"]

        # 基本损失
        loss_ce = losses["ce"](logits, y)
        loss_scl = losses["scl"](z, y)
        loss_ort = out.get("loss_ort", torch.tensor(0.0, device=device))
        loss_dyn = out.get("loss_dyn", torch.tensor(0.0, device=device))

        loss = loss_ce + 0.1 * loss_scl + 0.05 * loss_ort + 0.05 * loss_dyn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log["loss"] += loss.item()
        log["ce"] += loss_ce.item()
        log["scl"] += loss_scl.item()
        log["ort"] += float(loss_ort.item())
        log["dyn"] += float(loss_dyn.item())

    # 求平均
    for k in log:
        log[k] /= max(1, len(loader))
    return log


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    在验证集上评估准确率
    """
    model.eval()
    correct, total = 0, 0

    for batch in tqdm(loader, desc="val", leave=False):
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)["logits"]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return {"acc": correct / max(1, total)}
