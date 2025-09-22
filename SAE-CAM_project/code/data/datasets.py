import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
import glob


class MatDataset(Dataset):
    def __init__(self, root_dir, target_len=256000):
        self.files = glob.glob(os.path.join(root_dir, "*.mat"))
        assert self.files, f"在 {root_dir} 里没找到 .mat 文件"

        self.target_len = target_len

        # 类别映射：文件名前缀
        prefixes = sorted(set([os.path.basename(f).split("_")[0] for f in self.files]))
        self.class_to_idx = {p: i for i, p in enumerate(prefixes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.files[idx])
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        sample = mat[key][0, 0]

        Y = sample["Y"].squeeze()

        sig_dict = {}
        for rec in Y:
            name = str(rec["Name"][0])
            data = rec["Data"].squeeze().astype(np.float32)
            sig_dict[name] = data

        # 提取两个信号
        vib = sig_dict.get("vibration_1")
        cur = sig_dict.get("phase_current_1")

        # 对齐长度
        T = self.target_len
        def fix_length(sig):
            if len(sig) >= T:
                return sig[:T]
            else:
                pad = np.zeros(T - len(sig), dtype=np.float32)
                return np.concatenate([sig, pad])
        vib = fix_length(vib)
        cur = fix_length(cur)

        x = np.stack([vib, cur], axis=0)

        # 标签：文件名前缀
        fname = os.path.basename(self.files[idx])
        prefix = fname.split("_")[0]
        y = self.class_to_idx[prefix]

        return {"x": torch.tensor(x), "y": torch.tensor(y, dtype=torch.long)}





class WindowNpyDataset(Dataset):
    """Generic dataset: each .npy file stores [M, T] modalities (float32).
    Args:
        root/train|val/class_k/*.npy
        vib_idx: int index for vibration modality
        cur_idx: int index for current modality
    """
    def __init__(self, split_dir: str, vib_idx: int = 0, cur_idx: int = 1):
        self.files, self.labels = [], []
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for f in glob.glob(os.path.join(split_dir, c, "*.npy")):
                self.files.append(f)
                self.labels.append(self.class_to_idx[c])
        assert self.files, f"No .npy found in {split_dir}"
        self.vib_idx, self.cur_idx = vib_idx, cur_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)  # [M, T]
        if arr.ndim == 1:
            arr = arr[None, :]  # [1, T]
        vib = arr[self.vib_idx]
        cur = arr[self.cur_idx]

        def standardize(x):
            m, s = float(x.mean()), float(x.std() + 1e-6)
            return (x - m) / s

        vib, cur = standardize(vib), standardize(cur)
        x = np.stack([vib, cur], axis=0)  # [2, T]
        y = self.labels[idx]
        return {"x": torch.from_numpy(x), "y": torch.tensor(y, dtype=torch.long)}


class DummyWindows(Dataset):
    """随机数据生成器，用于快速测试流水线"""
    def __init__(self, n=256, T=1024, n_classes=4, seed=0):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, n_classes, size=n)
        self.x = []
        for lab in self.y:
            t = np.linspace(0, 1, T, dtype=np.float32)
            vib = np.sin(2*np.pi*(lab+1)*t) + 0.1*rng.standard_normal(T)
            cur = np.cos(2*np.pi*(lab+1)*t) + 0.1*rng.standard_normal(T)
            self.x.append(np.stack([vib, cur], axis=0))
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.from_numpy(self.x[idx]), "y": torch.tensor(int(self.y[idx]))}
