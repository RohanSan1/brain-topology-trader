"""Feature engineering: 22 features per stock per day."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "returns_1d",          # 0
    "returns_5d",          # 1
    "returns_20d",         # 2
    "volume_zscore_20d",   # 3
    "obv_norm",            # 4
    "rsi_14",              # 5
    "macd_line",           # 6
    "macd_signal",         # 7
    "roc_10",              # 8
    "atr_14",              # 9
    "bollinger_width",     # 10
    "vix",                 # 11
    "yield_curve_slope",   # 12
    "spy_1d_return",       # 13
    "sentiment_3d",        # 14
    "momentum_rank",       # 15
    "price_to_sma20",      # 16
    "dist_52w_high",       # 17  — distance from 52-week high (mean reversion)
    "dist_52w_low",        # 18  — distance from 52-week low  (support signal)
    "volume_trend",        # 19  — vol MA20/MA60 ratio (institutional activity)
    "roc_60",              # 20  — 60-day momentum (intermediate trend)
    "volatility_ratio",    # 21  — current ATR vs 60-day avg (vol regime)
]
assert len(FEATURE_NAMES) == 22


class FeatureEngineer:
    def compute_features(
        self,
        ohlcv: dict[str, pd.DataFrame],
        macro: dict[str, float],
        sentiment: dict[str, float],
    ) -> dict[str, np.ndarray | None]:
        """Return {ticker: ndarray(n_days, 22)} for all tickers with enough data."""

        momentum_20d: dict[str, float] = {}
        stock_features: dict[str, pd.DataFrame] = {}

        for ticker, df in ohlcv.items():
            try:
                feat = self._stock_features(df)
                if feat is not None and len(feat) >= 21:
                    stock_features[ticker] = feat
                    momentum_20d[ticker] = float(feat["returns_20d"].iloc[-1])
            except Exception as exc:
                log.debug("Feature error %s: %s", ticker, exc)

        mom_series = pd.Series(momentum_20d)
        ranks = mom_series.rank(pct=True)

        vix = macro.get("vix", 20.0)
        ycs = macro.get("yield_curve_slope", 0.5)
        spy_ret = macro.get("spy_1d_return", 0.0)

        result: dict[str, np.ndarray | None] = {}
        for ticker, feat in stock_features.items():
            feat = feat.copy()
            feat["vix"] = vix
            feat["yield_curve_slope"] = ycs
            feat["spy_1d_return"] = spy_ret
            feat["sentiment_3d"] = sentiment.get(ticker, 0.0)
            feat["momentum_rank"] = float(ranks.get(ticker, 0.5))
            arr = feat[FEATURE_NAMES].values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            result[ticker] = arr

        log.info("Features assembled for %d tickers", len(result))
        return result

    def _stock_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if len(df) < 21:
            return None
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]

        feat = pd.DataFrame(index=df.index)

        feat["returns_1d"] = close.pct_change(1)
        feat["returns_5d"] = close.pct_change(5)
        feat["returns_20d"] = close.pct_change(20)

        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std().replace(0, 1)
        feat["volume_zscore_20d"] = (volume - vol_mean) / vol_std

        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        obv_std = obv.rolling(20).std().replace(0, 1)
        feat["obv_norm"] = obv / obv_std

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
        feat["rsi_14"] = 100 - 100 / (1 + gain / loss)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        feat["macd_line"] = macd
        feat["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        feat["roc_10"] = close.pct_change(10) * 100

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        raw_atr = tr.rolling(14).mean()
        feat["atr_14"] = raw_atr / close.replace(0, 1)

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        feat["bollinger_width"] = (sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20.replace(0, 1)
        feat["price_to_sma20"] = (close / sma20.replace(0, 1)) - 1

        # 52-week high/low distance
        high_252 = close.rolling(252, min_periods=20).max()
        low_252 = close.rolling(252, min_periods=20).min()
        feat["dist_52w_high"] = (close / high_252.replace(0, 1)) - 1
        feat["dist_52w_low"] = (close / low_252.replace(0, 1)) - 1

        # Volume trend: 20-day vs 60-day moving average
        vol_ma20 = volume.rolling(20).mean()
        vol_ma60 = volume.rolling(60, min_periods=20).mean().replace(0, 1)
        feat["volume_trend"] = (vol_ma20 / vol_ma60) - 1

        # 60-day rate of change
        feat["roc_60"] = close.pct_change(60) * 100

        # Volatility regime: current ATR vs 60-day average
        atr_ma60 = raw_atr.rolling(60, min_periods=14).mean().replace(0, 1)
        feat["volatility_ratio"] = (raw_atr / atr_ma60) - 1

        # Macro + cross-sectional — filled by caller
        feat["vix"] = 0.0
        feat["yield_curve_slope"] = 0.0
        feat["spy_1d_return"] = 0.0
        feat["sentiment_3d"] = 0.0
        feat["momentum_rank"] = 0.0

        return feat.dropna(subset=["returns_20d"])
