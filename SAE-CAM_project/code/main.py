# main.py 顶部替换成这样
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.datasets import MatDataset, DummyWindows
from models.sae_cam import SAE_CAM
from models.losses import SupConLoss
from utils.train_eval import train_one_epoch, evaluate
from utils.logger import get_logger


import torch, argparse, os
from torch.utils.data import DataLoader, random_split

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=r"D:\Software_Download\QQ\PUBD")
    ap.add_argument("--use-dummy", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-classes", type=int, default=4)
    return ap.parse_args()

def build_loaders(args):
    if args.use_dummy:
        train_set = DummyWindows(n=256, T=1024, n_classes=args.n_classes)
        val_set = DummyWindows(n=64, T=1024, n_classes=args.n_classes)
    else:
        dataset = MatDataset(args.data_root)
        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val])
    tloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    vloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    return tloader, vloader

def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger()

    tloader, vloader = build_loaders(args)
    model = SAE_CAM(n_classes=args.n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = {"ce": torch.nn.CrossEntropyLoss(), "scl": SupConLoss()}

    for epoch in range(1, args.epochs+1):
        log = train_one_epoch(model, tloader, opt, losses, device)
        val = evaluate(model, vloader, device)
        logger.info(f"Epoch {epoch}: train_loss={log['loss']:.4f}, val_acc={val['acc']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/saecam_pubd.pt")
    logger.info("模型已保存: outputs/saecam_pubd.pt")

if __name__ == "__main__":
    main()
