"""Narzędzia do świec OHLCV (używane i dla data.taz, i dla Dukascopy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import CleaningLog, TableProfiler


class OHLCVTools:
    TIME_ALIASES = {"time", "date", "datetime", "timestamp", "gmt time", "local time", "date_time",
                    "gmt_time", "time_utc", "dt", "data", "czas"}
    RENAME = {"o": "open", "h": "high", "l": "low", "c": "close", "vol": "volume",
              "tick volume": "volume", "tickvol": "volume", "tick_volume": "volume",
              "adj close": "adj_close"}
    PRICE_COLS = ["open", "high", "low", "close"]

    # ------------------------------------------------------------------ normalizacja
    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = [str(c).strip().strip("<>").lower() for c in df.columns]
        df.columns = [cls.RENAME.get(c, c) for c in cols]
        # styl MT4: osobne kolumny date + time
        if "date" in df.columns and "time" in df.columns and df["time"].astype(str).str.len().max() <= 8:
            df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
            df = df.drop(columns=["date"])
        return df

    @classmethod
    def detect_time_column(cls, df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if str(c) in cls.TIME_ALIASES:
                return c
        for c in df.columns:  # zgadywanie po zawartości
            s = df[c].dropna().head(200)
            if len(s) and not pd.api.types.is_numeric_dtype(s):
                if pd.to_datetime(s, errors="coerce").notna().mean() > 0.9:
                    return c
        return None

    @staticmethod
    def to_utc(s: pd.Series, assume_tz: str = "UTC") -> pd.Series:
        """Czas -> UTC. Dla czasu bez strefy przyjmuje `assume_tz` (to założenie trzeba zweryfikować)."""
        if pd.api.types.is_numeric_dtype(s):
            unit = "ms" if s.dropna().median() > 1e11 else "s"
            return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        sample = s.dropna().astype(str).head(200)
        # format z kropkami (DD.MM.YYYY, eksporty Dukascopy/MT4) jest dzień-pierwszy; pandas domyślnie czyta miesiąc-pierwszy
        dayfirst = bool(len(sample) and sample.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}").mean() > 0.9)
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        if not pd.api.types.is_datetime64_any_dtype(parsed):  # mieszane strefy
            return pd.to_datetime(s, errors="coerce", utc=True)
        if parsed.dt.tz is None:
            return parsed.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        return parsed.dt.tz_convert("UTC")

    @classmethod
    def prepare(cls, df: pd.DataFrame, assume_tz: str = "UTC") -> tuple[pd.DataFrame, dict]:
        """Znormalizowane nazwy kolumn + indeks czasu UTC. Zwraca (df, info)."""
        d = cls.normalize_columns(df)
        info: dict = {"kolumny": list(d.columns)}
        if isinstance(d.index, pd.DatetimeIndex):
            tcol = None
            t = pd.Series(d.index, index=d.index)
        else:
            tcol = cls.detect_time_column(d)
            if tcol is None:
                info["błąd"] = "nie znaleziono kolumny czasu"
                return d, info
            t = d[tcol]
        t_utc = cls.to_utc(pd.Series(t.values, index=d.index), assume_tz)
        info.update({"kolumna_czasu": tcol or "(indeks)", "założona_strefa_dla_czasu_naiwnego": assume_tz,
                     "czas_nieparsowalny": int(t_utc.isna().sum())})
        d = d.drop(columns=[tcol]) if tcol else d
        d.index = pd.DatetimeIndex(t_utc, name="time_utc")
        d = d[~d.index.isna()]
        return d, info

    # ------------------------------------------------------------------ eksploracja
    @classmethod
    def explore(cls, df: pd.DataFrame, assume_tz: str = "UTC", gap_factor: float = 5.0) -> dict:
        raw_overview = TableProfiler.column_overview(df)
        d, info = cls.prepare(df, assume_tz)
        res: dict = {"info": info, "przegląd_kolumn": raw_overview, "head": df.head(3)}
        if "błąd" in info or d.empty:
            return res
        reg, gaps = TableProfiler.time_regularity(d.index, gap_factor=gap_factor)
        res["czas"] = {**reg, "min": d.index.min(), "max": d.index.max()}
        res["największe_luki"] = gaps
        res["rok_liczba"] = TableProfiler.yearly_counts(pd.Series(d.index))

        q: dict = {}
        broken = cls.open_is_broken(d)
        q["open_zepsuty (odtworzony z close[t-1])"] = broken
        if broken:
            d = cls.repair_open(d)
        px = [c for c in cls.PRICE_COLS if c in d.columns]
        for c in px + [x for x in ("volume",) if x in d.columns]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        if px:
            q["ceny_NaN (bez open po przerwie)"] = int(d[[c for c in px if c != "open"]].isna().any(axis=1).sum())
            q["ceny_<=0"] = int((d[px] <= 0).any(axis=1).sum())
        if set(cls.PRICE_COLS) <= set(d.columns):
            hi_bad = d["high"] < d[["open", "close"]].max(axis=1) - 1e-9
            lo_bad = d["low"] > d[["open", "close"]].min(axis=1) + 1e-9
            q["naruszenia_OHLC(high/low)"] = int((hi_bad | lo_bad | (d["high"] < d["low"])).sum())
            q["świece_płaskie(high==low)_%"] = round(100 * float((d["high"] == d["low"]).mean()), 2)
        if "close" in d.columns:
            c = d["close"].where(d["close"] > 0)
            lr = np.log(c).diff()
            dt = d.index.to_series().diff()
            intra = (dt <= 2 * dt.median()).to_numpy()
            r = lr[intra].dropna()
            if len(r) > 20:
                sig = 1.4826 * float((r - r.median()).abs().median())
                if sig > 0:
                    out = r[(r - r.median()).abs() > 10 * sig]
                    q["skoki_zwrotu_wewnątrz_sesji_>10 sigma(MAD)"] = len(out)
                    res["przykładowe_skoki"] = out.abs().sort_values(ascending=False).head(5).rename("|log zwrot|")
            gj = lr[~intra].dropna().abs().sort_values(ascending=False).head(5)
            if len(gj):
                res["skoki_po_przerwie"] = gj.rename("|log zwrot| przez przerwę/weekend")
        if "volume" in d.columns:
            q["wolumen_==0_%"] = round(100 * float((d["volume"] == 0).mean()), 2)
        q["wiersze_w_sobotę_UTC"] = int((d.index.dayofweek == 5).sum())
        q["kontrola_strefy"] = cls.tz_check(d)
        res["jakość"] = q
        return res

    @staticmethod
    def print_report(name: str, res: dict, show_fn):
        show_fn(res["info"], f"{name}: parsowanie czasu")
        if "czas" not in res:
            show_fn(res["head"], f"{name}: head (surowe)")
            return
        show_fn(res["head"], f"{name}: head (surowe)")
        show_fn(res["czas"], f"{name}: oś czasu")
        show_fn(res["jakość"], f"{name}: kontrole jakości")
        if len(res.get("największe_luki", [])):
            show_fn(res["największe_luki"], f"{name}: największe luki")
        if "przykładowe_skoki" in res:
            show_fn(res["przykładowe_skoki"], f"{name}: największe skoki zwrotu")
        if "skoki_po_przerwie" in res:
            show_fn(res["skoki_po_przerwie"], f"{name}: skoki PRZEZ przerwę (otwarcie po weekendzie, mogą być realną reakcją)")

    # ------------------------------------------------------------------ czyszczenie
    @classmethod
    def clean(cls, df: pd.DataFrame, name: str, assume_tz: str = "UTC") -> tuple[pd.DataFrame, CleaningLog]:
        """Konserwatywne czyszczenie: błędy formalne usuwa, wartości podejrzane tylko flaguje."""
        log = CleaningLog(name)
        d, info = cls.prepare(df, assume_tz)
        log.add("parsowanie czasu -> UTC (usunięto NaT)", len(df), len(d), f"strefa naiwna = {assume_tz}")
        if "błąd" in info:
            return d, log

        for c in cls.PRICE_COLS + ["volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        px = [c for c in cls.PRICE_COLS if c in d.columns]
        if cls.open_is_broken(d):
            d = cls.repair_open(d)
            d["flag_open_brak_po_przerwie"] = d["open"].isna()
            d["open"] = d["open"].fillna(d["close"])
            log.add_count("open naprawiony (zaokrąglony w źródle)", len(d), "open[t]=close[t-1]; po przerwie open=close + flaga")

        n = len(d)
        d = d.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        log.add("sortowanie + duplikaty znaczników (zostaje ostatni)", n, len(d))

        if px:
            n = len(d)
            d = d.dropna(subset=px)
            d = d[(d[px] > 0).all(axis=1)]
            log.add("ceny NaN lub <= 0", n, len(d))
        if set(cls.PRICE_COLS) <= set(d.columns):
            n = len(d)
            bad = (d["high"] < d[["open", "close"]].max(axis=1) - 1e-9) | \
                  (d["low"] > d[["open", "close"]].min(axis=1) + 1e-9) | (d["high"] < d["low"])
            d = d[~bad]
            log.add("naruszenia high/low względem open/close", n, len(d))
        if "close" in d.columns and len(d) > 20:
            r = np.log(d["close"]).diff()
            sig = 1.4826 * float((r - r.median()).abs().median())
            d["flag_skok_zwrotu"] = (r - r.median()).abs() > 10 * sig if sig > 0 else False
            log.add_count("FLAGA (nie usunięto): skok zwrotu > 10 sigma", int(d["flag_skok_zwrotu"].sum()),
                          "sprawdź ręcznie, to mogą być realne wydarzenia (np. ogłoszenia ceł)")
        return d, log

    # ------------------------------------------------------------------ naprawa `open`
    @classmethod
    def open_is_broken(cls, d: pd.DataFrame) -> bool:
        """open zaokrąglony do liczb całkowitych przy nieokrągłym close (błąd obserwowany w cache Dukascopy)."""
        if not {"open", "close"} <= set(d.columns):
            return False
        o = pd.to_numeric(d["open"], errors="coerce").dropna()
        c = pd.to_numeric(d["close"], errors="coerce").dropna()
        if len(o) < 20 or len(c) < 20:
            return False
        return bool((o % 1 == 0).mean() > 0.95 and (c % 1 == 0).mean() < 0.5)

    @classmethod
    def repair_open(cls, d: pd.DataFrame, max_gap_steps: float = 2.0) -> pd.DataFrame:
        """open[t] := close[t-1], o ile nie ma luki > max_gap_steps kroków (po przerwie NaN, bez zgadywania).
        high/low rozszerzane o nowy open. Oryginał w kolumnie open_src."""
        d = d.sort_index().copy()
        d = d[~d.index.duplicated(keep="last")]
        if not {"open", "high", "low", "close"} <= set(d.columns):
            return d
        for c in ("open", "high", "low", "close"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        dt = d.index.to_series().diff()
        step = dt.median()
        d["open_src"] = d["open"]
        d["open"] = d["close"].shift(1).where(dt <= max_gap_steps * step)
        d["high"] = d[["high", "open"]].max(axis=1)
        d["low"] = d[["low", "open"]].min(axis=1)
        return d

    @classmethod
    def start_price(cls, d: pd.DataFrame) -> pd.Series:
        """Cena na początku świecy (open; naprawiony, jeśli zepsuty). Bez informacji z przyszłości,
        w odróżnieniu od close świecy stemplowanej czasem otwarcia."""
        d = d.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if cls.open_is_broken(d):
            d = cls.repair_open(d)
        return pd.to_numeric(d["open"], errors="coerce").dropna()

    @staticmethod
    def tz_check(d: pd.DataFrame) -> dict:
        """FX/indeksy CFD zamykają się w piątek ok. 21-22 UTC i otwierają w niedzielę ok. 21-22 UTC."""
        s = pd.Series(d.index, index=d.index)
        f = s[s.dt.dayofweek == 4]
        u = s[s.dt.dayofweek == 6]
        return {"ostatnia_świeca_piątek_godz_UTC": f.groupby(f.dt.date).max().dt.hour.mode().tolist(),
                "pierwsza_świeca_niedziela_godz_UTC": u.groupby(u.dt.date).min().dt.hour.mode().tolist(),
                "interpretacja": "godziny 20-22 = zgodne z UTC; 15-17 oznaczałoby czas NY"}
