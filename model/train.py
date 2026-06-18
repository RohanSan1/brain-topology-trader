"""Historical supervised training — v4: quartile excess-return labels, focal loss, attention head."""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from data.features import FeatureEngineer, FEATURE_NAMES
from model.ncp_model import NCPTradingModel

log = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so the model focuses on hard ones."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class HistoricalTrainer:
    def train(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        checkpoint_fn=None,
        start_epoch: int = 0,
        weights_path: str = None,
    ) -> NCPTradingModel:
        import yfinance as yf

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s | tickers: %d", device, len(tickers))

        # ── Fetch macro once ─────────────────────────────────────────────────
        log.info("Fetching macro history (^VIX, ^TNX, ^IRX, SPY)…")
        macro_df = None
        spy_5d_fwd = None
        try:
            macro_raw = yf.download(
                ["^VIX", "^TNX", "^IRX", "SPY"],
                start=start_date, end=end_date,
                interval="1d", auto_adjust=True, progress=False,
            )
            mc = macro_raw["Close"] if isinstance(macro_raw.columns, pd.MultiIndex) else macro_raw
            mc.index = pd.to_datetime(mc.index).normalize()
            mc = mc.rename(columns={"^VIX": "vix"})
            mc["yield_curve_slope"] = mc["^TNX"] - mc["^IRX"]
            mc["spy_1d_return"] = mc["SPY"].pct_change(1)
            spy_5d_fwd = mc["SPY"].pct_change(5).shift(-5)        # 5-day forward SPY return
            macro_df = mc[["vix", "yield_curve_slope", "spy_1d_return"]].ffill().fillna(0.0)
            log.info("Macro loaded: %d rows", len(macro_df))
        except Exception as exc:
            log.warning("Macro fetch failed: %s — using stubs", exc)

        spy_3d_cum = None
        if macro_df is not None:
            spy_3d_cum = macro_df["spy_1d_return"].rolling(3).sum().fillna(0.0)

        # ── Pass 1: download all tickers, build per-ticker feat_dfs ─────────
        log.info("Pass 1: downloading OHLCV and building features…")
        engineer = FeatureEngineer()
        all_feat_dfs: dict[str, pd.DataFrame] = {}
        all_close: dict[str, pd.Series] = {}
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        for batch_start in range(0, len(tickers), 100):
            batch = tickers[batch_start: batch_start + 100]
            log.info("Downloading batch %d–%d / %d", batch_start + 1, batch_start + len(batch), len(tickers))
            try:
                raw = yf.download(
                    batch, start=start_date, end=end_date,
                    interval="1d", auto_adjust=True, group_by="ticker", progress=False,
                )
            except Exception as exc:
                log.warning("Batch download failed: %s", exc)
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        if ticker in raw.columns.get_level_values(0):
                            df = raw[ticker].copy()
                        elif ticker in raw.columns.get_level_values(1):
                            df = raw.xs(ticker, level=1, axis=1).copy()
                        else:
                            continue
                    else:
                        continue

                    df = df.rename(columns=str.lower).dropna(how="all")
                    df.index = pd.to_datetime(df.index).normalize()
                    if len(df) < config.SEQUENCE_LENGTH + config.RETURN_HORIZON + 5:
                        continue

                    feat_df = engineer._stock_features(df)
                    if feat_df is None or len(feat_df) < config.SEQUENCE_LENGTH + config.RETURN_HORIZON:
                        continue

                    if macro_df is not None:
                        aligned = macro_df.reindex(feat_df.index, method="ffill")
                        feat_df["vix"] = aligned["vix"].fillna(20.0).values
                        feat_df["yield_curve_slope"] = aligned["yield_curve_slope"].fillna(0.5).values
                        feat_df["spy_1d_return"] = aligned["spy_1d_return"].fillna(0.0).values
                    else:
                        feat_df["vix"] = 20.0
                        feat_df["yield_curve_slope"] = 0.5
                        feat_df["spy_1d_return"] = 0.0

                    all_feat_dfs[ticker] = feat_df
                    all_close[ticker] = df["close"]

                except Exception as exc:
                    log.warning("Skipping %s: %s", ticker, exc)

            log.info("Feature DFs collected: %d tickers so far", len(all_feat_dfs))

        if not all_feat_dfs:
            raise RuntimeError("No feature data collected")

        log.info("Pass 1 done: %d tickers with features", len(all_feat_dfs))

        # ── Cross-sectional momentum rank ─────────────────────────────────────
        log.info("Computing cross-sectional momentum rank panel…")
        returns_panel = pd.DataFrame({t: fd["returns_20d"] for t, fd in all_feat_dfs.items()})
        rank_panel = returns_panel.rank(axis=1, pct=True).fillna(0.5)

        # ── Cross-sectional quartile labels (excess return over SPY) ─────────
        log.info("Computing cross-sectional quartile label panel…")
        fwd_excess: dict[str, pd.Series] = {}
        for ticker, feat_df in all_feat_dfs.items():
            close = all_close[ticker].reindex(feat_df.index)
            stock_5d = close.pct_change(5).shift(-5)
            if spy_5d_fwd is not None:
                spy_aligned = spy_5d_fwd.reindex(feat_df.index, method="ffill").fillna(0.0)
                fwd_excess[ticker] = stock_5d - spy_aligned
            else:
                fwd_excess[ticker] = stock_5d

        excess_df = pd.DataFrame(fwd_excess)
        q_low = excess_df.quantile(config.QUARTILE_THRESHOLD, axis=1)   # bottom 25%
        q_high = excess_df.quantile(1 - config.QUARTILE_THRESHOLD, axis=1)  # top 25%
        log.info("Quartile panel: %d dates × %d tickers", len(excess_df), len(excess_df.columns))

        # ── Pass 2: fill cross-sectional features + build samples ────────────
        log.info("Pass 2: building training samples…")
        all_X: list[np.ndarray] = []
        all_y: list[int] = []
        all_idx: list[int] = []
        all_sector: list[int] = []

        for ticker, feat_df in all_feat_dfs.items():
            try:
                feat_df = feat_df.copy()

                if ticker in rank_panel.columns:
                    feat_df["momentum_rank"] = (
                        rank_panel[ticker].reindex(feat_df.index).ffill().fillna(0.5).values
                    )
                else:
                    feat_df["momentum_rank"] = 0.5

                stock_3d = feat_df["returns_1d"].rolling(3).sum().fillna(0.0)
                if spy_3d_cum is not None:
                    spy_aligned = spy_3d_cum.reindex(feat_df.index, method="ffill").fillna(0.0)
                    abnormal = stock_3d - spy_aligned
                else:
                    abnormal = stock_3d
                feat_df["sentiment_3d"] = abnormal.clip(-0.15, 0.15) / 0.15

                feat_arr = feat_df[FEATURE_NAMES].values.astype(np.float32)
                feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

                ticker_idx = ticker_to_idx.get(ticker, 0)
                sector_idx = config.TICKER_SECTOR.get(ticker, 12)

                ticker_excess = fwd_excess.get(ticker)

                for i in range(config.SEQUENCE_LENGTH, len(feat_arr) - config.RETURN_HORIZON):
                    if ticker_excess is None:
                        continue
                    date = feat_df.index[i]
                    if date not in q_low.index:
                        continue
                    exc_ret = ticker_excess.iloc[i] if i < len(ticker_excess) else np.nan
                    if pd.isna(exc_ret):
                        continue
                    ql, qh = q_low.loc[date], q_high.loc[date]
                    if pd.isna(ql) or pd.isna(qh):
                        continue
                    if exc_ret >= qh:
                        label = 1   # top quartile — outperforms market
                    elif exc_ret <= ql:
                        label = 0   # bottom quartile — underperforms market
                    else:
                        continue    # middle 50% — skip noisy zone

                    all_X.append(feat_arr[i - config.SEQUENCE_LENGTH: i])
                    all_y.append(label)
                    all_idx.append(ticker_idx)
                    all_sector.append(sector_idx)

            except Exception as exc:
                log.warning("Skipping %s in pass 2: %s", ticker, exc)

        if not all_X:
            raise RuntimeError("No training samples generated")

        log.info("Total training samples: %d", len(all_X))

        y_np = np.array(all_y)
        counts = np.bincount(y_np, minlength=2).astype(np.float32)
        log.info("Class distribution — underperform: %d | outperform: %d", *counts.astype(int))
        weights = 1.0 / counts.clip(min=1)
        weights = weights / weights.mean()
        log.info("Class weights — underperform: %.3f | outperform: %.3f", *weights)

        X = torch.FloatTensor(np.array(all_X)).to(device)
        y = torch.LongTensor(all_y).to(device)
        idx_t = torch.LongTensor(all_idx).to(device)
        sec_t = torch.LongTensor(all_sector).to(device)
        log.info("Dataset on %s — X: %.1f GB", device, X.element_size() * X.nelement() / 1e9)

        dataset = TensorDataset(X, idx_t, sec_t, y)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        # ── Model ────────────────────────────────────────────────────────────
        model = NCPTradingModel(
            num_stocks=len(tickers),
            num_features=config.NUM_FEATURES,
            input_size=config.INPUT_SIZE,
            ncp_units=config.NCP_UNITS,
            ncp_output_size=config.NCP_OUTPUT_SIZE,
            ncp_sparsity=config.NCP_SPARSITY,
            embedding_dim=config.EMBEDDING_DIM,
            num_sectors=config.NUM_SECTORS,
            sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
            dropout=config.DROPOUT,
        ).to(device)

        if weights_path and start_epoch > 0:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            log.info("Loaded checkpoint weights from %s (resuming epoch %d)", weights_path, start_epoch)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        for pg in optimizer.param_groups:
            pg['initial_lr'] = config.LEARNING_RATE
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.SGDR_T0, T_mult=config.SGDR_T_MULT,
            last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
        )
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=class_weights)

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(start_epoch, config.HISTORICAL_EPOCHS):
            model.train()
            total_loss, correct, n = 0.0, 0, 0
            for xb, ib, sb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb, ib, sb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                n += len(yb)
            scheduler.step()
            log.info(
                "Epoch %d/%d | loss=%.4f | acc=%.4f | lr=%.2e",
                epoch + 1, config.HISTORICAL_EPOCHS,
                total_loss / n, correct / n, scheduler.get_last_lr()[0],
            )
            if checkpoint_fn is not None:
                checkpoint_fn(model, epoch + 1)

        return model
