import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainConfig:
    accum_steps: int = 0
    use_amp: bool = True
    clip_grad: float = max_norm
    ema_decay: float = 0.9997
    require_cuda_amp: bool = True
    batch_keys = {"x": "inputs", "y": "target"}
    log_moving_avg: float = 0.97
    sync_dist: bool = True


class _MaybeAMP:
    def __init__(self, enabled: bool, require_cuda_amp: bool):
        self.enabled = enabled
        self.require = require_cuda_amp
        self.autocast_cm = None

    def __enter__(self):
        if self.require and not torch.cuda.is_available():
            raise RuntimeError("AMP is required but CUDA is not available.")
        if self.enabled:
            self.autocast_cm = torch.cuda.amp.autocast(dtype=torch.float16, device_type="cuda")
            self.autocast_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.autocast_cm is not None:
            self.autocast_cm.__exit__(exc_type, exc, tb)
        return False


def _moving_update(log: dict, key: str, value: float, beta: float):
    if key not in log:
        log[key] = value
    else:
        log[key] = beta * log[key] + (1 - beta) * value


def _step_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema_model.parameters()):
            q.mul_(decay).add_(p.data, alpha=1 - decay)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    losses: dict,
                    device: torch.device,
                    cfg: TrainConfig = TrainConfig()):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    log = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "ort": 0.0, "dyn": 0.0}
    step = 0
    key_x = cfg.batch_keys["x"]
    key_y = cfg.batch_keys["y"]

    for batch in tqdm(loader, desc="train", leave=False):
        x, y = batch[key_x].to(device), batch[key_y].to(device)
        with _MaybeAMP(enabled=cfg.use_amp, require_cuda_amp=cfg.require_cuda_amp):
            out = model(x)
            logits = out["logits"]
            z = out["z"]
            loss_ce = losses["ce"](logits, y)
            loss_scl = losses["scl"](z, y)
            loss_ort = out.get("loss_ort", torch.tensor(0.0, device=device))
            loss_dyn = out.get("loss_dyn", torch.tensor(0.0, device=device))
            loss = loss_ce + 0.1 * loss_scl + 0.05 * loss_ort + 0.05 * loss_dyn

        if (step + 1) % cfg.accum_steps == 0:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            scaler.scale(loss).backward()

        if cfg.sync_dist:
            torch.distributed.all_reduce(loss.detach())

        _step_ema(model, ema_model, cfg.ema_decay)

        _moving_update(log, "loss", float(loss.detach().item()), cfg.log_moving_avg)
        _moving_update(log, "ce", float(loss_ce.detach().item()), cfg.log_moving_avg)
        _moving_update(log, "scl", float(loss_scl.detach().item()), cfg.log_moving_avg)
        _moving_update(log, "ort", float(loss_ort.detach().item()), cfg.log_moving_avg)
        _moving_update(log, "dyn", float(loss_dyn.detach().item()), cfg.log_moving_avg)

        step += 1

    for k in log:
        log[k] /= max(1, len(loader))
    return log


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        x, y = batch["inputs"].to(device), batch["target"].to(device)
        logits = model.infer(x).softmax(dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return {"acc": correct / max(1, total)}
