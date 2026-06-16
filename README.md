# brain-topology-trader

Daily autonomous trading bot using a Neural Circuit Policy (NCP / Liquid Time-Constant) model.

## Overview

| Property | Value |
|---|---|
| Universe | Top ~800 US stocks by market cap |
| Account | Alpaca Paper Trading |
| Leverage | 2× ($100K margin → $200K effective) |
| Directions | Long **and** Short |
| Model | LTC / AutoNCP (ncps PyTorch library) |
| Compute | Modal (Fluid Compute + A10G GPU) |
| State | Modal Volume `trading-data` at `/data` |

## Architecture

```
modal_app.py        ← Modal app, image, cron entrypoints
config.py           ← constants, TICKER_UNIVERSE, hyperparams
data/
  ingest.py         ← Twelve Data (8-key rotation), FRED, Finnhub
  features.py       ← TA indicators + macro + sentiment + cross-sectional rank
model/
  ncp_model.py      ← LTC + AutoNCP + stock embedding
  train.py          ← one-time 25-year supervised training
  update.py         ← daily online RL (policy gradient)
execution/
  signals.py        ← 3-day smoothing, confidence threshold, ranking
  sizing.py         ← Kelly criterion, position caps
  broker.py         ← Alpaca Paper Trading client
reward/
  compute.py        ← composite reward: direction accuracy + PnL return
utils/
  logger.py         ← structured logging
  notify.py         ← email / Telegram daily report
requirements.txt
```

## Cron Schedule

| Job | UTC | IST | ET |
|---|---|---|---|
| Inference + Execution | 17:30 | ~23:00 | 13:00 |
| EOD Weight Update | 22:00 | ~03:30+1 | 17:30 |

## Setup

### 1. Modal Secrets (create once)

```bash
# Alpaca Paper Trading
modal secret create alpaca-secret \
  ALPACA_API_KEY=... ALPACA_SECRET_KEY=... \
  ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Twelve Data (8 keys)
modal secret create twelvedata-secret \
  TWELVE_DATA_KEY_1=... TWELVE_DATA_KEY_2=... \
  TWELVE_DATA_KEY_3=... TWELVE_DATA_KEY_4=... \
  TWELVE_DATA_KEY_5=... TWELVE_DATA_KEY_6=... \
  TWELVE_DATA_KEY_7=... TWELVE_DATA_KEY_8=...

modal secret create fred-secret FRED_API_KEY=...
modal secret create finnhub-secret FINNHUB_API_KEY=...

# Notify — pick one:
# Email:    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL
# Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
modal secret create notify-secret NOTIFY_EMAIL=... SMTP_HOST=... SMTP_PORT=587 SMTP_USER=... SMTP_PASS=...
```

### 2. Deploy

```bash
git clone https://github.com/Rohan5commit/brain-topology-trader.git
cd brain-topology-trader
modal deploy modal_app.py
```

### 3. One-time Historical Training

```bash
modal run modal_app.py::train_historical
```

This fetches 25 years of OHLCV (2000–2025), trains the NCP on 3-class next-day direction labels, and saves `ncp_weights_base.pt` to the Modal Volume. Expect 2–6 hours on A10G.

## State Files (Modal Volume `/data`)

| File | Purpose |
|---|---|
| `ncp_weights_base.pt` | Base trained weights |
| `ncp_weights_latest.pt` | Daily-updated weights |
| `signals_history.parquet` | Last 3 days of raw model outputs |
| `positions_state.parquet` | Open positions + entry dates |
| `features_cache.parquet` | Latest feature snapshot |

## Notes

- No secrets are hardcoded anywhere — all via `modal.Secret.from_name(...)`
- All persistent state lives in Modal Volume, not git
- Alpaca base URL is always `https://paper-api.alpaca.markets` (paper trading only)
