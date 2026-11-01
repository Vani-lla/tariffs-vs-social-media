"""Wspólne narzędzia: konfiguracja, log czyszczenia, profilowanie tabel, parsowanie dat.

Żadna klasa w tym pakiecie NIE łączy źródeł danych. Każde źródło jest czyszczone osobno
i zapisywane do własnego pliku w `cfg.out_root`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # w Jupyterze ładne tabele, w zwykłym Pythonie zwykły print
    from IPython.display import display as _display
except Exception:  # pragma: no cover
    _display = None


# --------------------------------------------------------------------------- konfiguracja
@dataclass
class Config:
    """Jedno miejsce na ścieżki. Zmień tylko to, reszta notebooka zostaje."""
    fnspid_csv: str = "data/FNSPID.csv"                       # surowy FNSPID: używa go tylko FNSPIDExplorer, notebook go pomija
    fnspid_clean_csv: str = "data/fnspid_tariffs_clean.csv"   # plik po wstępnym czyszczeniu/filtrze ceł
    # źródła postów Trumpa: nazwa -> ścieżka. Tylko archiwum Twittera, bo Truth Social (od 2022) jest poza okresem badania.
    trump_sources: dict = field(default_factory=lambda: {
        "twitter": "data/trump_tweets.csv",
        "truth_social": "data/truth_social_posts.csv",  # poza okresem 2018-2019, eksplorator tylko to zaraportuje
    })
    # --- automatyczne pobieranie (downloaders.ensure_all)
    kaggle_datasets: dict = field(default_factory=lambda: {
        "twitter": "codebreaker619/donald-trump-tweets-dataset",
        "truth_social": "troykueh/donald-trumps-truth-social-posts-nlp-analysis",
    })
    gta_actors: dict = field(default_factory=lambda: {"USA": 840, "CHN": 156})  # kody UN
    gta_types: dict = field(default_factory=lambda: {"Import tariff": 47, "Other import charges": 73})
    fr_queries: dict = field(default_factory=lambda: {
        "301": '"Section 301" China tariff', "232": '"Section 232" tariff',
        "IEEPA": "IEEPA tariff", "reciprocal": "reciprocal tariff"})
    fr_types: list = field(default_factory=lambda: ["RULE", "NOTICE", "PRESDOCU"])
    market_archive: str = "data/market_data"
    gkg_cache_dir: str = "gdelt_data/gkg"          # ten sam katalog, którego używa GDELTAnalyzer
    gta_csv: str = "events_out/gta_interventions.csv"
    fr_csv: str = "events_out/fr_cache_v2.csv"
    dukascopy_cache: str = "data_raw/dukascopy"
    market_work_dir: str = "data_raw/market"
    vix_csv: str = "data/vix_daily.csv"
    market_instruments: dict = field(default_factory=lambda: {
        "S&P 500": "usa500idxusd", "Nasdaq": "usatechidxusd", "Hang Seng": "hkgidxhkd",
        "EURUSD": "eurusd", "TLT": "tltususd"})
    # daty główne (do weryfikacji); używane np. do gęstego pobrania GKG
    key_events: dict = field(default_factory=lambda: {
        "2018-03-08 stal/aluminium (232)": "2018-03-08", "2018-03-22 memorandum 301": "2018-03-22",
        "2018-06-15 lista 25% na 50 mld": "2018-06-15", "2018-07-06 cła wchodzą": "2018-07-06",
        "2018-09-17 10% na 200 mld": "2018-09-17", "2019-05-05 tweet 10->25%": "2019-05-05",
        "2019-05-10 25% wchodzi": "2019-05-10", "2019-08-01 tweet 10% na 300 mld": "2019-08-01",
        "2019-08-05 Chiny manipulantem": "2019-08-05", "2019-08-23 odwet Chin": "2019-08-23",
        "2019-12-13 faza pierwsza": "2019-12-13"})
    out_root: str = "clean_out"
    market_assume_tz: str = "UTC"                  # strefa plików bez informacji o strefie (ZAŁOŻENIE, sprawdź!)
    periods: dict = field(default_factory=lambda: {
        "2018-2019": ("2018-01-01", "2019-12-31"),
    })
    # horyzonty analizy przyczynowej względem t = czas publikacji (etykieta -> odstęp)
    horizons: dict = field(default_factory=lambda: {
        "1m": "1min", "5m": "5min", "1h": "1h", "3h": "3h", "1d": "1D", "3d": "3D", "7d": "7D"})

    def horizon_deltas(self) -> dict:
        return {k: pd.Timedelta(v) for k, v in self.horizons.items()}

    def out(self, name: str) -> Path:
        p = Path(self.out_root) / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def period_bounds(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Okresy jako (start, koniec_wyłącznie) w UTC."""
        return [(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1))
                for s, e in self.periods.values()]


# --------------------------------------------------------------------------- wyświetlanie
def show(obj, title: str | None = None):
    if title:
        print(f"\n=== {title} ===")
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{k}: {v}")
    elif _display is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
        _display(obj)
    else:
        print(obj)


# --------------------------------------------------------------------------- log czyszczenia
class CleaningLog:
    """Zapisuje ile wierszy usunął każdy krok, żeby czyszczenie było audytowalne."""

    def __init__(self, name: str):
        self.name = name
        self.steps: list[dict] = []

    def add(self, step: str, before: int, after: int, note: str = ""):
        self.steps.append({"krok": step, "wiersze_przed": int(before), "wiersze_po": int(after),
                           "usunięte": int(before - after), "uwaga": note})

    def add_count(self, step: str, removed: int, note: str = ""):
        self.steps.append({"krok": step, "wiersze_przed": np.nan, "wiersze_po": np.nan,
                           "usunięte": int(removed), "uwaga": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def save(self, path: Path):
        self.to_frame().to_csv(path, index=False)


# --------------------------------------------------------------------------- daty i pliki
def parse_utc(s: pd.Series) -> pd.Series:
    """Parsuje daty tekstowe do datetime64[ns, UTC]; nieparsowalne -> NaT."""
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, utc=True, errors="coerce")


def sniff_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = "".join(f.readline() for _ in range(20))
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(path, nrows: int | None = None, **kw) -> pd.DataFrame:
    """Czyta csv/txt/tsv/parquet/json/xlsx; separator csv jest zgadywany."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    if suf == ".json":
        df = pd.read_json(path)
        return df.head(nrows) if nrows else df
    if suf in (".csv", ".txt", ".tsv", ".dat"):
        return pd.read_csv(path, sep=sniff_sep(path), nrows=nrows, low_memory=False, **kw)
    raise ValueError(f"Nieobsługiwany format pliku: {path.name}")


TABULAR_SUFFIXES = {".csv", ".txt", ".tsv", ".dat", ".parquet", ".pq", ".json", ".xlsx", ".xls"}


# --------------------------------------------------------------------------- profilowanie
class TableProfiler:
    @staticmethod
    def column_overview(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        n = max(len(df), 1)
        for c in df.columns:
            s = df[c]
            nn = int(s.notna().sum())
            try:
                nu = int(s.nunique(dropna=True))
            except TypeError:
                nu = np.nan
            empty = np.nan
            if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
                empty = int((s.dropna().astype(str).str.strip() == "").sum())
            ex = s.dropna().iloc[0] if nn else None
            rows.append({"kolumna": c, "dtype": str(s.dtype), "niepuste": nn,
                         "braki_%": round(100 * (1 - nn / n), 2), "puste_napisy": empty,
                         "unikalne": nu, "przykład": str(ex)[:70]})
        return pd.DataFrame(rows)

    @staticmethod
    def datetime_summary(s: pd.Series) -> dict:
        s = pd.Series(s)
        ok = s.dropna()
        out = {"n": len(s), "NaT": int(s.isna().sum())}
        if len(ok):
            out.update({"min": ok.min(), "max": ok.max(),
                        "tz": str(getattr(ok.dt, "tz", None))})
        return out

    @staticmethod
    def yearly_counts(s: pd.Series) -> pd.Series:
        ok = pd.Series(s).dropna()
        return ok.dt.year.value_counts().sort_index().rename("n")

    @staticmethod
    def time_regularity(idx, gap_factor: float = 5.0, top: int = 8) -> tuple[dict, pd.DataFrame]:
        """Krok czasowy, duplikaty znaczników, największe luki (z oznaczeniem luk weekendowych)."""
        idx = pd.DatetimeIndex(idx)
        idx = idx[~idx.isna()]
        info = {"n": len(idx), "rosnący_w_pliku": bool(idx.is_monotonic_increasing),
                "zduplikowane_znaczniki": int(idx.duplicated().sum())}
        uniq = idx.unique().sort_values()
        if len(uniq) < 2:
            return info, pd.DataFrame()
        diffs = pd.Series(uniq[1:] - uniq[:-1], index=uniq[1:])
        med = diffs.median()
        info["mediana_kroku"] = med
        gaps = diffs[diffs > gap_factor * med]
        g = pd.DataFrame({"koniec_luki": gaps.index, "długość": gaps.values})
        g["początek_luki"] = g["koniec_luki"] - g["długość"]
        g["weekendowa"] = (g["początek_luki"].dt.dayofweek.isin([4, 5])) & (g["długość"] <= pd.Timedelta(hours=72))
        info[f"luki_>{gap_factor:g}x_mediany"] = len(g)
        info["w_tym_weekendowe"] = int(g["weekendowa"].sum())
        g = g.sort_values("długość", ascending=False).head(top)[
            ["początek_luki", "koniec_luki", "długość", "weekendowa"]].reset_index(drop=True)
        return info, g