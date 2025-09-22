
from __future__ import annotations
import logging, os, sys
from datetime import datetime

def get_logger(name: str = "saecam", level: int=logging.INFO, log_dir: str|None="logs"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"))
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
