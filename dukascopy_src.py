"""Dukascopy (dukascopy-python >= 5): pobieranie miesiącami z cache + eksploracja próbek wokół zdarzeń.

API biblioteki: fetch_candles(instrument=I_EUR_USD, interval=1, time_unit=TimeUnit.MINUTE,
offer_side=OfferSide.BID, start, end, version=2). Nazwy instrumentów NIE są zgadywane na sztywno:
`list_instruments("EURUSD")` pokazuje stałe I_* z zainstalowanej wersji, a `resolve()` dopasowuje po fragmencie
(ignoruje wielkość liter oraz znaki _ i /).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .base import CleaningLog, Config, parse_utc, show
from .ohlcv import OHLCVTools


class DukascopyFetcher:
    VERSIONS = (2, 1)          # kolejność prób wersji feedu: najpierw 2, przy błędzie 1
    UNIT_NAMES = {"SEC": ["SECOND", "SEC"], "MIN": ["MINUTE", "MIN"], "HOUR": ["HOUR"],
                  "DAY": ["DAY"], "WEEK": ["WEEK"], "MONTH": ["MONTH"]}
    COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, cfg: Config):
        self.cache_dir = Path(cfg.dukascopy_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ biblioteka i instrumenty
    @staticmethod
    def _lib():
        try:
            import dukascopy_python as duka
            from dukascopy_python import instruments as ins
        except ImportError as ex:
            raise ImportError("Zainstaluj: pip install -U dukascopy-python") from ex
        if not hasattr(duka, "fetch_candles"):
            raise RuntimeError("Zainstalowana wersja dukascopy-python jest stara (brak fetch_candles). "
                               "Zaktualizuj: pip install -U dukascopy-python (wymagane >= 5).")
        return duka, ins

    @staticmethod
    def _norm(x) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(x).upper())

    @classmethod
    def _catalog(cls) -> dict:
        _, ins = cls._lib()
        return {n: getattr(ins, n) for n in dir(ins) if n.startswith("I_")}

    @classmethod
    def list_instruments(cls, pattern: str | None = None) -> list[str]:
        cat = cls._catalog()
        if not pattern:
            return sorted(cat)
        p = cls._norm(pattern)
        return sorted(n for n, v in cat.items() if p in cls._norm(n) or p in cls._norm(v))

    @classmethod
    def resolve(cls, name_or_fragment: str) -> tuple[str, object]:
        """Zwraca (nazwa stałej, wartość instrumentu). Błąd, jeśli dopasowanie nie jest jednoznaczne."""
        cat = cls._catalog()
        if name_or_fragment in cat:
            return name_or_fragment, cat[name_or_fragment]
        frag = cls._norm(name_or_fragment)
        exact = [n for n in cat if cls._norm(n) in (frag, "I" + frag)]   # np. "EURUSD" -> I_EUR_USD
        if len(exact) == 1:
            return exact[0], cat[exact[0]]
        hits = cls.list_instruments(name_or_fragment)
        if len(hits) != 1:
            raise ValueError(f"'{name_or_fragment}' pasuje do {len(hits)} instrumentów: {hits[:15]}")
        return hits[0], cat[hits[0]]

    # ------------------------------------------------------------------ interwał, strony, daty
    @classmethod
    def _time_unit(cls, duka, token: str):
        for cand in cls.UNIT_NAMES.get(token, [token]):
            if hasattr(duka.TimeUnit, cand):
                return getattr(duka.TimeUnit, cand)
        avail = [m for m in dir(duka.TimeUnit) if m.isupper()]
        raise ValueError(f"Brak TimeUnit dla '{token}'. Dostępne: {avail}")

    @classmethod
    def parse_interval(cls, duka, interval: str):
        """'MIN_1' -> (1, TimeUnit.MINUTE); 'HOUR_1' -> (1, TimeUnit.HOUR)."""
        m = re.match(r"^([A-Z]+)_(\d+)$", interval.upper())
        if not m:
            raise ValueError(f"Interwał '{interval}': oczekiwany format np. MIN_1, MIN_5, HOUR_1, DAY_1")
        return int(m.group(2)), cls._time_unit(duka, m.group(1))

    @staticmethod
    def _side(duka, side: str):
        return getattr(duka.OfferSide, side.upper())

    @staticmethod
    def _utc(x) -> pd.Timestamp:
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    @classmethod
    def _months(cls, start, end):
        cur, end = cls._utc(start).normalize(), cls._utc(end).normalize()
        while cur < end:
            nxt = min(cur + pd.offsets.MonthBegin(1), end)
            yield cur, nxt
            cur = nxt

    # ------------------------------------------------------------------ pobieranie
    @classmethod
    def _candles_to_df(cls, candles) -> pd.DataFrame:
        rows = [{"time_utc": c.timestamp, **{k: getattr(c, k, None) for k in cls.COLS}} for c in candles]
        if not rows:
            return pd.DataFrame(columns=cls.COLS)
        df = pd.DataFrame(rows)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df.set_index("time_utc").sort_index()

    def _fetch_chunk(self, duka, inst, n, unit, side, a, b) -> pd.DataFrame:
        last = None
        for v in self.VERSIONS:
            try:
                cs = duka.fetch_candles(instrument=inst, interval=n, time_unit=unit, offer_side=side,
                                        start=a.to_pydatetime(), end=b.to_pydatetime(), version=v)
                return self._candles_to_df(list(cs))
            except Exception as ex:   # próbujemy kolejną wersję feedu
                last = ex
        raise last

    def fetch(self, instrument: str, start, end, interval: str = "MIN_1", side: str = "BID",
              force: bool = False) -> pd.DataFrame:
        """Pobiera [start, end) w kawałkach miesięcznych; każdy kawałek trafia do cache (csv)."""
        duka, _ = self._lib()
        const_name, inst = self.resolve(instrument)
        n, unit = self.parse_interval(duka, interval)
        sd = self._side(duka, side)
        parts = []
        for a, b in self._months(start, end):
            f = self.cache_dir / f"{const_name}_{interval}_{side}_{a:%Y%m%d}_{b:%Y%m%d}.csv"
            part = None
            if f.exists() and not force:
                try:
                    part = pd.read_csv(f, index_col=0)
                except EmptyDataError:
                    part = pd.DataFrame(columns=self.COLS)
            if part is None:
                part = self._fetch_chunk(duka, inst, n, unit, sd, a, b)
                if len(part):
                    part.index.name = "time_utc"
                    part.to_csv(f)
                else:   # pusty wynik też zapisujemy (sam nagłówek), żeby nie pytać ponownie
                    pd.DataFrame(columns=["time_utc"] + self.COLS).to_csv(f, index=False)
            if len(part):
                part.index = pd.DatetimeIndex(parse_utc(pd.Series(part.index)))
                parts.append(part)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).sort_index()
        df.index.name = "time_utc"
        if OHLCVTools.open_is_broken(df):
            print(f"  {const_name}: open zaokrąglony w źródle -> odtwarzam z close[t-1]")
            df = OHLCVTools.repair_open(df)
        return df


class DukascopyExplorer:
    # okna wokół znanych zdarzeń: wystarczą do oceny jakości, bez pobierania lat danych
    DEFAULT_WINDOWS = {
        "2018-03-22 (sekcja 301, Chiny)": ("2018-03-20", "2018-03-27"),
        "2018-07-06 (cła 25% na towary z Chin za 34 mld USD)": ("2018-07-04", "2018-07-11"),
        "2019-05-05 (cła 10->25%)": ("2019-05-02", "2019-05-09"),
        "2019-08-01 (cła 10% na kolejne 300 mld USD)": ("2019-07-30", "2019-08-07"),
    }
    DEFAULT_INSTRUMENTS = ["EURUSD", "USA500", "USDCNH"]   # fragmenty nazw; sprawdź list_instruments()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fetcher = DukascopyFetcher(cfg)
        self.out_dir = cfg.out("dukascopy")
        self.raw: dict = {}
        self.results: dict = {}

    def fetch_samples(self, instruments=None, windows=None, interval: str = "MIN_1"):
        for inst in (instruments or self.DEFAULT_INSTRUMENTS):
            for wname, (s, e) in (windows or self.DEFAULT_WINDOWS).items():
                key = f"{inst} | {wname}"
                try:
                    df = self.fetcher.fetch(inst, s, e, interval=interval)
                    self.raw[key] = df
                    print(f"{key}: {len(df):,} świec")
                except Exception as ex:
                    print(f"{key}: BŁĄD ({type(ex).__name__}: {ex})")
        return {k: len(v) for k, v in self.raw.items()}

    def explore(self):
        for key, df in self.raw.items():
            if df.empty:
                print(f"\n{key}: pusto")
                continue
            res = OHLCVTools.explore(df, assume_tz="UTC", gap_factor=5)
            self.results[key] = res
            print("\n" + "#" * 80 + f"\n# {key}\n" + "#" * 80)
            OHLCVTools.print_report(key, res, show)

    def clean(self) -> pd.DataFrame:
        logs = []
        for key, df in self.raw.items():
            if df.empty:
                continue
            d, log = OHLCVTools.clean(df, key, assume_tz="UTC")
            fname = "".join(ch if ch.isalnum() else "_" for ch in key)[:80]
            d.to_csv(self.out_dir / f"{fname}_clean.csv")
            lf = log.to_frame(); lf.insert(0, "seria", key)
            logs.append(lf)
        full = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        full.to_csv(self.out_dir / "dukascopy_cleaning_log.csv", index=False)
        return full