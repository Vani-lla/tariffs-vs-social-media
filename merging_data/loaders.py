"""Wczytanie WYCZYSZCZONYCH plików z `cfg.out_root` (wynik `run_cleaning`).

Ten moduł niczego nie czyści i niczego nie łączy: wczytuje każde źródło osobno, z poprawnymi typami
(czas postów/artykułów/świec w UTC ze strefą, daty dzienne GTA/FR jako daty BEZ strefy, kolumny logiczne jako bool).
Tylko FAKTY: czas, liczności, identyfikatory, ceny. Kolumn semantycznych (ton, sentyment) nie wczytujemy.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning_data.base import Config, parse_utc


def as_bool(s: pd.Series) -> pd.Series:
    """'True'/'False' z CSV (także z brakami) -> bool."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "t", "yes"])


def tag_of(label: str) -> str:
    """Ta sama nazwa pliku co w MarketCleaner: 'S&P 500' -> 'S_P_500'."""
    return re.sub(r"\W+", "_", label).strip("_")


def fast_utc(s: pd.Series) -> pd.Series:
    """Szybkie parsowanie dat ISO (pliki zapisane przez pandas); w razie czego wolniejszy parse_utc."""
    s = pd.Series(s)
    try:
        return pd.to_datetime(s, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        return parse_utc(s.astype(str))


def naive_day(s: pd.Series) -> pd.Series:
    """Data dzienna bez strefy (północ). Działa dla napisów z '+00:00' i bez strefy (GTA/FR są dzienne)."""
    t = pd.to_datetime(pd.Series(s), errors="coerce", utc=True)
    return t.dt.tz_localize(None).dt.normalize()


def to_naive_day(t: pd.Series) -> pd.Series:
    """Czas UTC -> data (północ, bez strefy), do agregacji dziennych."""
    return pd.Series(t).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()


class CleanLoader:
    GKG_COLS = ["SourceCommonName", "DocumentIdentifier", "time_utc", "wordcount", "url_key"]   # bez tonu

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.out_root)

    def _need(self, p: Path) -> Path:
        if not p.exists():
            raise FileNotFoundError(f"Brak {p}. Uruchom najpierw czyszczenie (run_cleaning) w poprzednim notebooku.")
        return p

    def optional(self, name: str, *a, **kw):
        """Wczytuje źródło albo zwraca None z komunikatem (brak pliku nie zatrzymuje scalania)."""
        try:
            return getattr(self, name)(*a, **kw)
        except FileNotFoundError as ex:
            print(f"UWAGA: {name}: {ex}")
            return None

    # ------------------------------------------------------------------ Trump
    def trump(self, source: str = "twitter") -> pd.DataFrame:
        d = pd.read_csv(self._need(self.root / "trump" / "trump_clean.csv"), low_memory=False, dtype={"post_id": str})
        d["time_utc"] = fast_utc(d["time_utc"])
        for c in ("is_retweet", "relevant", "relevant_strict", "china_trade", "flag_tylko_link"):
            if c in d:
                d[c] = as_bool(d[c])
        if "relevant" not in d:                       # starsza wersja czyszczenia
            d["relevant"] = d["relevant_strict"] if "relevant_strict" in d else False
        if "is_retweet" not in d:
            d["is_retweet"] = False
        if "source" in d and source:
            d = d[d["source"] == source]
        d["thread"] = d["thread"].fillna("").astype(str) if "thread" in d else ""
        for c in ("favs", "rts"):
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        d["text"] = d["text"].fillna("").astype(str) if "text" in d else ""
        d = d.dropna(subset=["time_utc", "post_id"])
        return d.drop_duplicates("post_id").sort_values("time_utc").reset_index(drop=True)

    # ------------------------------------------------------------------ FNSPID
    def fnspid(self) -> pd.DataFrame:
        d = pd.read_csv(self._need(self.root / "fnspid" / "fnspid_clean.csv"), low_memory=False)
        d["t_utc"] = fast_utc(d["t_utc"])
        for c in ("is_topic", "is_sector_note", "has_time", "is_first_in_story"):
            if c in d:
                d[c] = as_bool(d[c])
        for c, v in (("is_topic", True), ("is_sector_note", False)):
            if c not in d:
                d[c] = v
        d = d.dropna(subset=["t_utc"])
        d["day"] = to_naive_day(d["t_utc"])
        if "thread" in d:
            d["thread"] = d["thread"].fillna("").astype(str)
        return d.sort_values("t_utc").reset_index(drop=True)

    # ------------------------------------------------------------------ GKG
    def gkg(self) -> pd.DataFrame:
        p = self._need(self.root / "gkg" / "gkg_clean.csv")
        have = pd.read_csv(p, nrows=0).columns
        d = pd.read_csv(p, usecols=[c for c in self.GKG_COLS if c in have], low_memory=False)
        d["time_utc"] = fast_utc(d["time_utc"])
        return d.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

    def gkg_files(self) -> pd.DatetimeIndex:
        """Znaczniki WSZYSTKICH pobranych paczek 15-min (także bez artykułów o cłach): mianownik stawek na paczkę.
        Gdy zipy usunięto, bierze listę zapisaną wcześniej w clean_out/gkg/gkg_files.csv."""
        listing = self.root / "gkg" / "gkg_files.csv"
        files = sorted(Path(self.cfg.gkg_cache_dir).glob("*.gkg.csv.zip"))
        if files:
            st = pd.to_datetime(pd.Series([p.name[:14] for p in files]), format="%Y%m%d%H%M%S", utc=True, errors="coerce")
            idx = pd.DatetimeIndex(st.dropna().drop_duplicates().sort_values())
            listing.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"file_utc": idx}).to_csv(listing, index=False)
            return idx
        if listing.exists():
            return pd.DatetimeIndex(fast_utc(pd.read_csv(listing)["file_utc"]).sort_values())
        print(f"UWAGA: brak paczek GKG w {self.cfg.gkg_cache_dir} i brak {listing}: stawki GKG nie będą liczone.")
        return pd.DatetimeIndex([], tz="UTC")

    # ------------------------------------------------------------------ GTA / FR (daty dzienne, bez strefy)
    def gta(self) -> pd.DataFrame:
        d = pd.read_csv(self._need(self.root / "gta" / "gta_clean.csv"), low_memory=False)
        for c in ("date_announced", "date_published", "date_implemented", "date_removed"):
            if c in d:
                d[c] = naive_day(d[c])
        if "actor" not in d:
            d["actor"] = "?"
        return d

    def fr(self) -> pd.DataFrame:
        d = pd.read_csv(self._need(self.root / "federal_register" / "fr_clean.csv"), low_memory=False)
        for c in ("publication_date", "effective_on", "signing_date", "event_date"):
            if c in d:
                d[c] = naive_day(d[c])
        for c in ("core", "relevant"):
            if c in d:
                d[c] = as_bool(d[c])
        return d

    # ------------------------------------------------------------------ rynek
    def market(self, label: str) -> pd.DataFrame:
        """mid_open, mid_close (5-min), indeks UTC. Świece ze skrzyżowanym bid/ask są zerowane (NaN)."""
        p = self._need(self.root / "market" / f"{tag_of(label)}_mid.csv")
        d = pd.read_csv(p, index_col=0)
        d.index = pd.DatetimeIndex(fast_utc(pd.Series(d.index)))
        d.index.name = "time_utc"
        d = d[d.index.notna()]
        for c in ("mid_open", "mid_close"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
            d.loc[d[c] <= 0, c] = np.nan                       # log z ceny <= 0 nie istnieje
        if "flag_crossed" in d:
            bad = as_bool(d["flag_crossed"])
            d.loc[bad.to_numpy(), ["mid_open", "mid_close"]] = np.nan
        d = d.sort_index()
        return d[~d.index.duplicated(keep="last")]

    def market_all(self) -> dict:
        out = {}
        for label in self.cfg.market_instruments:
            try:
                out[label] = self.market(label)
            except FileNotFoundError as ex:
                print(f"UWAGA: {label}: {ex}")
        return out

    # ------------------------------------------------------------------ inwentarz
    def inventory(self) -> pd.DataFrame:
        rows = []
        for name, p in [("Trump", self.root / "trump" / "trump_clean.csv"),
                        ("FNSPID", self.root / "fnspid" / "fnspid_clean.csv"),
                        ("GKG", self.root / "gkg" / "gkg_clean.csv"),
                        ("GTA", self.root / "gta" / "gta_clean.csv"),
                        ("Federal Register", self.root / "federal_register" / "fr_clean.csv")] + \
                       [(f"Rynek: {lab}", self.root / "market" / f"{tag_of(lab)}_mid.csv") for lab in self.cfg.market_instruments]:
            rows.append({"źródło": name, "plik": str(p), "istnieje": p.exists(),
                         "MB": round(p.stat().st_size / 1e6, 2) if p.exists() else None})
        n_zip = len(list(Path(self.cfg.gkg_cache_dir).glob("*.gkg.csv.zip")))
        rows.append({"źródło": "GKG: paczki 15-min (mianownik)", "plik": self.cfg.gkg_cache_dir, "istnieje": n_zip > 0,
                     "MB": None, "pliki": n_zip})
        return pd.DataFrame(rows)
