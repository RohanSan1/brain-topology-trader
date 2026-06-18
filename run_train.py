"""Standalone training script for Lightning AI (no Modal dependencies)."""
import os
import sys
import urllib.request
import torch
from datetime import datetime, timezone

os.makedirs("/data", exist_ok=True)
sys.path.insert(0, "/app")

import config
from model.train import HistoricalTrainer

_CKPT_URL = "https://github.com/Rohan5commit/brain-topology-trader/releases/download/v2-checkpoint/v2_latest.pt"
_CKPT_EPOCH = 18

_epoch_file = "/data/checkpoint_epoch.txt"
_weights_latest = "/data/ncp_weights_latest.pt"
_weights_base = "/data/ncp_weights_base.pt"

if not os.path.exists(_weights_latest):
    print(f"[{datetime.now(timezone.utc).isoformat()}] Downloading epoch-{_CKPT_EPOCH} checkpoint...", flush=True)
    urllib.request.urlretrieve(_CKPT_URL, _weights_latest)
    with open(_epoch_file, "w") as f:
        f.write(str(_CKPT_EPOCH))
    print(f"[{datetime.now(timezone.utc).isoformat()}] Checkpoint ready, resuming from epoch {_CKPT_EPOCH}", flush=True)

start_epoch = 0
weights_path = None
if os.path.exists(_epoch_file) and os.path.exists(_weights_latest):
    with open(_epoch_file) as f:
        start_epoch = int(f.read().strip())
    weights_path = _weights_latest
    print(f"[{datetime.now(timezone.utc).isoformat()}] Resuming from epoch {start_epoch}", flush=True)
else:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting fresh", flush=True)


def _checkpoint(model, epoch):
    torch.save(model.state_dict(), _weights_latest)
    with open(_epoch_file, "w") as f:
        f.write(str(epoch))
    print(f"[{datetime.now(timezone.utc).isoformat()}] Checkpoint saved: epoch {epoch}", flush=True)


trainer = HistoricalTrainer()
model = trainer.train(
    tickers=config.TICKER_UNIVERSE,
    start_date=config.HISTORICAL_START,
    end_date=config.HISTORICAL_END,
    checkpoint_fn=_checkpoint,
    start_epoch=start_epoch,
    weights_path=weights_path,
)

torch.save(model.state_dict(), _weights_base)
torch.save(model.state_dict(), _weights_latest)
print(f"[{datetime.now(timezone.utc).isoformat()}] Training complete — saved to {_weights_base}", flush=True)

