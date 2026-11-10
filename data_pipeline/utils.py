"""Shared helpers: config, paths, seeding, JSONL I/O, logging, hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DEFAULT_CONFIG = PKG_DIR / "config.yaml"


# ---------------------------------------------------------------- config
class Config(dict):
    """dict with attribute access and path resolution."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    @property
    def data_root(self) -> Path:
        env = os.environ.get("CIRA_DATA_ROOT")
        root = Path(env) if env else Path(self["paths"]["data_root"])
        return root if root.is_absolute() else REPO_ROOT / root

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.data_root / p


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("CIRA_CONFIG") or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base" in cfg:  # small override files: `_base: config.yaml` + only the changed keys
        base = load_config(path.parent / cfg.pop("_base"))
        base.pop("_config_path", None)
        _deep_update(base, cfg)
        cfg = dict(base)
    if overrides:
        _deep_update(cfg, overrides)
    cfg["_config_path"] = str(path)
    return Config(cfg)


def _deep_update(d: dict, u: dict) -> None:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def config_sha256(cfg: Config) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- seeding
def derive_seed(master: int, *names: Any) -> int:
    """Stable per-step / per-item seed (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256(("|".join([str(master), *map(str, names)])).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def seed_everything(seed: int, log: logging.Logger | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    if log:
        log.info("seed=%d", seed)


# ---------------------------------------------------------------- logging
def get_logger(name: str, cfg: Config | None = None) -> logging.Logger:
    log = logging.getLogger(f"cira.{name}")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    if cfg is not None:
        d = cfg.path("log_dir")
        d.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(d / f"{name}.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    return log


# ---------------------------------------------------------------- io
def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
            n += 1
    return n


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


class RejectionLog:
    """Append-only JSONL log: one line per rejected item, with a machine-readable reason."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w")
        self.counts: dict[str, int] = {}

    def __call__(self, item_id: str, reason: str, **details):
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.f.write(json.dumps({"id": item_id, "reason": reason, **details}, sort_keys=True) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
