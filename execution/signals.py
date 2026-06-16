"""3-day signal smoothing, confidence thresholding, and trade ranking."""
import logging
import os
from collections import deque

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# Class indices
BUY, HOLD, SELL = 0, 1, 2


class SignalProcessor:
    def __init__(self) -> None:
        self._history: dict[str, deque] = {}

    def _load_history(self) -> dict[str, list[list[float]]]:
        """Load last N days of raw signals from Modal Volume parquet."""
        if not os.path.exists(config.SIGNALS_PATH):
            return {}
        df = pd.read_parquet(config.SIGNALS_PATH)
        history: dict[str, list[list[float]]] = {}
        for ticker in df.index:
            row = df.loc[ticker]
            history[ticker] = []
            for day in range(config.SIGNAL_SMOOTH_DAYS):
                col_p = f"day{day}_probs"
                if col_p in row.index and isinstance(row[col_p], list):
                    history[ticker].append(row[col_p])
        return history

    def smooth_and_rank(
        self,
        raw_signals: dict[str, list[float]],  # {ticker: [p_buy, p_hold, p_sell]}
    ) -> dict[str, dict]:
        """
        Merge today's raw signals with the last N-1 days from Volume.
        Returns {ticker: {side, score, confidence, p_buy, p_sell}}.
        """
        history = self._load_history()
        smoothed: dict[str, dict] = {}

        for ticker, probs in raw_signals.items():
            past = history.get(ticker, [])
            window = past[-(config.SIGNAL_SMOOTH_DAYS - 1):] + [probs]
            arr = np.array(window)                    # (<=3, 3)
            mean_probs = arr.mean(axis=0)             # (3,)

            p_buy = float(mean_probs[BUY])
            p_sell = float(mean_probs[SELL])
            score = p_buy - p_sell                    # positive → bullish

            if p_buy > p_sell:
                side = "buy"
                confidence = p_buy
            else:
                side = "sell"
                confidence = p_sell

            smoothed[ticker] = {
                "side": side,
                "score": score,
                "confidence": confidence,
                "p_buy": p_buy,
                "p_sell": p_sell,
            }

        log.info("Smoothed signals for %d tickers", len(smoothed))
        return smoothed

    def save_signals(self, raw_signals: dict[str, list[float]]) -> None:
        """Persist today's raw signals, rotating out oldest day."""
        history = self._load_history()
        records = []
        for ticker, probs in raw_signals.items():
            past = history.get(ticker, [])
            window = (past + [probs])[-config.SIGNAL_SMOOTH_DAYS:]
            row = {"ticker": ticker}
            for i, p in enumerate(window):
                row[f"day{i}_probs"] = p
            records.append(row)

        if not records:
            return
        df = pd.DataFrame(records).set_index("ticker")
        df.to_parquet(config.SIGNALS_PATH)
        log.info("Saved signals for %d tickers → %s", len(records), config.SIGNALS_PATH)
