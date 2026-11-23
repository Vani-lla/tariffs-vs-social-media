"""ETF-y na chińskie spółki (KWEB, CQQQ, FXI, MCHI, ASHR) + SPY: pobranie z yfinance, kontrola, zwroty nadzwyczajne.

Zwrot nadzwyczajny (model rynkowy skorygowany betą): AR_t = r_t - beta * r_SPY,t; beta z okna estymacji
`cfg.china_estimation` (2017, czyli przed okresem badania). Alfy NIE odejmujemy: w 2017 (hossa w Chinach) wynosiła
+3..+12 pb/dzień i odejmowana codziennie dawała sztuczny dryf AR ok. -50% w 2 lata. Alfa jest raportowana w `params`.
Zwroty logarytmiczne z cen skorygowanych (dywidendy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Config, show
from .downloaders import ensure_pip


class ChinaETFs:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tickers = list(cfg.china_etfs) + [cfg.china_benchmark]
        self.path = Path(cfg.china_etf_csv)
        self.params: pd.DataFrame | None = None

    # ------------------------------------------------------------------ pobranie
    def download(self, start: str = "2016-12-01", end: str = "2020-04-01", force: bool = False) -> pd.DataFrame:
        if self.path.exists() and not force:
            return self.load()
        yf = ensure_pip("yfinance")
        d = yf.download(self.tickers, start=start, end=end, auto_adjust=True, progress=False)
        if d.empty:
            raise RuntimeError("yfinance nie zwrócił danych (sprawdź połączenie / tickery).")
        long = (d.stack(level="Ticker", future_stack=True).reset_index()
                .rename(columns={"Date": "day", "Ticker": "ticker"}))
        long.columns = [str(c).lower() for c in long.columns]
        long = long[["day", "ticker", "open", "high", "low", "close", "volume"]].sort_values(["ticker", "day"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(self.path, index=False)
        print(f"Zapisano {len(long):,} wierszy -> {self.path}")
        return long

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, parse_dates=["day"])

    # ------------------------------------------------------------------ kontrola
    def check(self, long: pd.DataFrame | None = None) -> pd.DataFrame:
        long = self.load() if long is None else long
        px = long.pivot(index="day", columns="ticker", values="close")
        r = np.log(px).diff()
        rep = pd.DataFrame({
            "pierwszy_dzień": px.apply(lambda s: s.first_valid_index()),
            "ostatni_dzień": px.apply(lambda s: s.last_valid_index()),
            "sesje": px.notna().sum(),
            "braki_w_środku": px.apply(lambda s: int(s.loc[s.first_valid_index():s.last_valid_index()].isna().sum())),
            "ceny<=0": (px <= 0).sum(),
            "|zwrot|>15%": (r.abs() > 0.15).sum(),
            "max_|zwrot|_%": (100 * r.abs().max()).round(2),
        })
        show(rep, "ETF-y chińskie: kontrola pokrycia")
        return rep

    # ------------------------------------------------------------------ zwroty
    def returns(self, long: pd.DataFrame | None = None) -> pd.DataFrame:
        """Tabela dzienna: {T}_ret (log), {T}_ar (nadzwyczajny vs SPY), CHN_TECH_ar (KWEB+CQQQ), CHN_ALL_ar (średnia)."""
        long = self.load() if long is None else long
        px = long.pivot(index="day", columns="ticker", values="close").sort_index()
        r = np.log(px).diff()
        b = self.cfg.china_benchmark
        a0, a1 = (pd.Timestamp(x) for x in self.cfg.china_estimation)
        est = r.loc[a0:a1]
        rows, out = [], pd.DataFrame(index=r.index)
        out[f"{b}_ret"] = r[b]
        for t in self.cfg.china_etfs:
            e = est[[t, b]].dropna()
            beta, alpha = np.polyfit(e[b], e[t], 1)
            resid = e[t] - (alpha + beta * e[b])
            rows.append({"ticker": t, "alfa_pb": 1e4 * alpha, "beta": beta, "R2": 1 - resid.var() / e[t].var(),
                         "sd_AR_pb": 1e4 * resid.std(), "n_est": len(e)})
            out[f"{t}_ret"] = r[t]
            out[f"{t}_ar"] = r[t] - beta * r[b]
        out["CHN_TECH_ar"] = out[["KWEB_ar", "CQQQ_ar"]].mean(axis=1, skipna=False)
        out["CHN_ALL_ar"] = out[[f"{t}_ar" for t in self.cfg.china_etfs]].mean(axis=1, skipna=False)
        self.params = pd.DataFrame(rows)
        return out
