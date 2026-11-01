"""Czyszczenie wszystkich źródeł (BEZ łączenia). Uruchamiane jedną funkcją `run_cleaning`.

Zasady:
- błędy formalne (nieparsowalna data, duplikaty, ceny <= 0, poza okresem badania) usuwamy i zapisujemy w logu,
- wartości podejrzane, ale możliwe do wyjaśnienia (np. GTA: wdrożenie przed ogłoszeniem, skoki cen) tylko FLAGUJEMY,
- każde źródło zapisuje własny plik w `cfg.out_root/<źródło>/` oraz log kroków.
"""
from __future__ import annotations

import html
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CleaningLog, Config, parse_utc, read_table, show
from .fnspid import FNSPIDTopics
from .gta_fr import FederalRegisterExplorer
from .ohlcv import OHLCVTools


def period_mask(cfg: Config, s: pd.Series) -> pd.Series:
    """True, gdy znacznik czasu (naiwny = UTC) mieści się w którymś z okresów cfg.periods."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    m = pd.Series(False, index=t.index)
    for a, b in cfg.period_bounds():
        m |= (t >= a) & (t < b)
    return m


# ======================================================================= FNSPID
class FNSPIDCleaner:
    """Czyści `fnspid_tariffs_clean.csv` (już po wstępnym filtrze). Nic nie usuwa poza błędami formalnymi:
    notki zbiorcze i powtórzenia tej samej historii dla wielu tickerów są FLAGOWANE."""
    CANDS = {"t": ["date_utc", "date", "datetime", "published", "time"],
             "title": ["article_title", "title", "headline"],
             "ticker": ["stock_symbol", "ticker", "symbol"],
             "publisher": ["publisher", "source"],
             "url": ["url", "link"]}

    def __init__(self, cfg: Config, path: str | None = None):
        self.cfg = cfg
        self.path = path or cfg.fnspid_clean_csv
        self.out_dir = cfg.out("fnspid")

    @staticmethod
    def _col(df, cands):
        low = {str(c).lower(): c for c in df.columns}
        return next((low[c] for c in cands if c in low), None)

    def run(self, drop_non_topic: bool = False, min_title_len: int = 10,
            out_name: str = "fnspid_clean.csv") -> pd.DataFrame:
        raw = read_table(self.path)
        c = {k: self._col(raw, v) for k, v in self.CANDS.items()}
        if c["t"] is None or c["title"] is None:
            raise ValueError(f"FNSPID: nie rozpoznano kolumny daty/tytułu. Kolumny: {list(raw.columns)}")

        def get(k):
            return raw[c[k]] if c[k] else pd.Series(pd.NA, index=raw.index)

        d = pd.DataFrame({"t": parse_utc(get("t").astype(str)), "title": get("title").astype("string"),
                          "ticker": get("ticker").astype("string"), "publisher": get("publisher").astype("string"),
                          "url": get("url").astype("string")})
        log = CleaningLog("FNSPID")
        log.add("wczytane wiersze", len(d), len(d))

        n = len(d); d = d.dropna(subset=["t"]); log.add("nieparsowalna data", n, len(d))
        n = len(d); d = d[period_mask(self.cfg, d["t"])]
        log.add("poza okresem badania", n, len(d), str(list(self.cfg.periods.values())))

        d["title"] = (d["title"].map(lambda x: html.unescape(x) if isinstance(x, str) else x).astype("string")
                      .str.replace(r"\s+", " ", regex=True).str.strip())
        n = len(d); d = d[d["title"].fillna("").str.len() >= min_title_len]
        log.add(f"pusty lub krótki tytuł (< {min_title_len} znaków)", n, len(d))

        d["ticker"] = d["ticker"].str.upper().str.strip()
        dup = pd.DataFrame({"a": d["title"].str.lower(), "b": d["t"], "c": d["ticker"]}).duplicated()
        n = len(d); d = d[~dup.to_numpy()]
        log.add("duplikaty (tytuł, czas, ticker)", n, len(d))

        d = FNSPIDTopics.prepare(d, tcol="title", scol="ticker", pcol="publisher", dcol="t")
        d = d.sort_values("t").reset_index(drop=True)
        d["is_first_in_story"] = ~d["story_key"].duplicated()
        if drop_non_topic:
            n = len(d); d = d[d["is_topic"]]; log.add("tytuły bez słownika ceł/handlu", n, len(d))

        log.add_count("FLAGA is_first_in_story=False (ta sama historia dla kolejnego tickera)",
                      int((~d["is_first_in_story"]).sum()), "do analiz na poziomie historii filtruj is_first_in_story")
        log.add_count("FLAGA is_sector_note (notki 'Shares of several...')", int(d["is_sector_note"].sum()),
                      "skutek rynkowy, nie pierwotny news")
        log.add_count("FLAGA has_time=False (czas 00:00:00, tylko data)", int((~d["has_time"]).sum()),
                      "NIE używać w horyzontach krótszych niż 1d")
        log.add_count("FLAGA is_topic=False", int((~d["is_topic"]).sum()), "poza słownikiem ceł/handlu")

        cols = ["t", "title", "ticker", "publisher", "publisher_norm", "url", "is_topic", "is_sector_note",
                "thread", "has_time", "story_key", "is_first_in_story"]
        out = d[[x for x in cols if x in d.columns]].rename(columns={"t": "t_utc"})
        out.to_csv(self.out_dir / out_name, index=False)
        log.save(self.out_dir / "fnspid_cleaning_log.csv")
        print(f"FNSPID: zapisano {len(out):,} wierszy ({int(out['is_first_in_story'].sum())} niezależnych historii) "
              f"-> {self.out_dir / out_name}")
        return log.to_frame()


# ======================================================================= Federal Register
class FRCleaner:
    """Federal Register: okres badania, deduplikacja z zachowaniem zapytań, event_date = podpis (jeśli jest) lub publikacja."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = cfg.out("federal_register")

    def run(self, out_name: str = "fr_clean.csv") -> pd.DataFrame:
        d = FederalRegisterExplorer(self.cfg).load()
        log = CleaningLog("FR")
        log.add("wczytane wiersze", len(d), len(d))
        n = len(d); d = d.dropna(subset=["document_number", "publication_date"])
        log.add("brak numeru dokumentu lub daty publikacji", n, len(d))
        n = len(d); d = d[period_mask(self.cfg, d["publication_date"])]
        log.add("poza okresem badania (publication_date)", n, len(d))
        if "query" in d:
            qmap = d.groupby("document_number")["query"].agg(lambda s: ";".join(sorted(set(map(str, s)))))
        n = len(d); d = d.drop_duplicates("document_number").copy()
        log.add("duplikaty document_number", n, len(d), "ten sam dokument z kilku zapytań; zapytania scalone w 'queries'")
        if "query" in d:
            d["queries"] = d["document_number"].map(qmap)
        d["title"] = d["title"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        abstract = d["abstract"] if "abstract" in d else pd.Series("", index=d.index)
        text = d["title"].fillna("") + " " + abstract.fillna("")
        d["relevant"] = text.str.contains(FederalRegisterExplorer.REL_RE)
        d["core"] = text.str.contains(FederalRegisterExplorer.STRICT_RE) & ~text.str.contains(FederalRegisterExplorer.NOISE_RE)
        sign = d["signing_date"] if "signing_date" in d else pd.Series(pd.NaT, index=d.index)
        d["event_date"] = sign.fillna(d["publication_date"])
        d["event_date_source"] = np.where(sign.notna(), "signing_date", "publication_date")
        d["lag_sign_to_pub_d"] = (d["publication_date"] - sign).dt.days
        d["has_effective_on"] = d["effective_on"].notna()
        d["flag_eff_przed_pub"] = d["effective_on"] < d["publication_date"]
        d["flag_sign_po_pub"] = sign > d["publication_date"]
        log.add_count("FLAGA core=False (szum: poza ścisłym słownikiem ceł lub słownik szumu)", int((~d["core"]).sum()),
                      "zostaje w pliku; do analiz filtruj core")
        log.add_count("FLAGA event_date z signing_date", int((d["event_date_source"] == "signing_date").sum()),
                      "reszta: event_date = publication_date (to NIE jest moment ogłoszenia)")
        log.add_count("FLAGA effective_on przed publikacją", int(d["flag_eff_przed_pub"].sum()))
        log.add_count("FLAGA podpis po publikacji", int(d["flag_sign_po_pub"].sum()), "logicznie niemożliwe: sprawdź ręcznie")
        d = d.sort_values("event_date").reset_index(drop=True)
        d.to_csv(self.out_dir / out_name, index=False)
        log.save(self.out_dir / "fr_cleaning_log.csv")
        print(f"FR: zapisano {len(d):,} dokumentów ({int(d['core'].sum())} core) -> {self.out_dir / out_name}")
        return log.to_frame()


# ======================================================================= rynek (data.taz)
class MarketCleaner:
    """Czyści TYLKO instrumenty z cfg.market_instruments (bid i ask), nie całe archiwum (~3 GB).
    Zachowuje okres badania + bufor na horyzont t+7d. Nic nie interpoluje przez weekendy i przerwy."""

    def __init__(self, cfg: Config, mk):
        self.cfg, self.mk = cfg, mk
        self.out_dir = cfg.out("market")

    def run(self, sides=("bid", "ask"), buffer_days: int = 14, instruments: dict | None = None) -> pd.DataFrame:
        bounds = self.cfg.period_bounds()
        lo = min(a for a, _ in bounds)
        hi = max(b for _, b in bounds) + pd.Timedelta(days=buffer_days)
        logs, summ = [], []
        for label, frag in (instruments or self.cfg.market_instruments).items():
            tag = re.sub(r"\W+", "_", label).strip("_")
            cleaned = {}
            for side in sides:
                try:
                    p = self.mk.find(frag, side)
                except FileNotFoundError as ex:
                    print(f"{label}/{side}: pominięto ({ex})")
                    continue
                d, log = OHLCVTools.clean(read_table(p), f"{label}_{side}", assume_tz=self.cfg.market_assume_tz)
                n = len(d)
                d = d[(d.index >= lo) & (d.index < hi)]
                log.add("poza okresem badania + bufor", n, len(d), f"{lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")
                d.index.name = "time_utc"
                d.to_csv(self.out_dir / f"{tag}_{side}_clean.csv")
                cleaned[side] = d
                lf = log.to_frame(); lf.insert(0, "seria", f"{label}_{side}"); logs.append(lf)
                summ.append({"seria": f"{label}_{side}", "świece": len(d), "od": d.index.min(), "do": d.index.max(),
                             "flagi_skoków": int(d["flag_skok_zwrotu"].sum()) if "flag_skok_zwrotu" in d else None})
            if {"bid", "ask"} <= set(cleaned):
                b, a = cleaned["bid"], cleaned["ask"]
                j = b[["open", "close"]].join(a[["open", "close"]], how="inner", lsuffix="_bid", rsuffix="_ask")
                spread = j["close_ask"] - j["close_bid"]
                crossed = int((spread < 0).sum())
                out = pd.DataFrame({"mid_open": (j["open_bid"] + j["open_ask"]) / 2,
                                    "mid_close": (j["close_bid"] + j["close_ask"]) / 2,
                                    "spread_close": spread, "flag_crossed": spread < 0})
                out.index.name = "time_utc"
                out.to_csv(self.out_dir / f"{tag}_mid.csv")
                summ.append({"seria": f"{label}_mid", "świece": len(out), "od": out.index.min(), "do": out.index.max(),
                             "flagi_skoków": None, "ask<bid (skrzyżowane)": crossed})
        show(pd.DataFrame(summ), "Rynek: podsumowanie po czyszczeniu")
        full = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        full.to_csv(self.out_dir / "market_cleaning_log.csv", index=False)
        return full


# ======================================================================= orkiestracja
def run_cleaning(cfg: Config, trump=None, gkg=None, market=None, dukascopy=None,
                 gkg_max_files: int | None = None, drop_retweets: bool = True, only: list[str] | None = None) -> dict:
    """Czyści wszystkie źródła po kolei; błąd jednego nie zatrzymuje reszty.
    Obiekty trump/gkg/market/dukascopy można przekazać z notebooka, inaczej powstają od nowa."""
    logs: dict = {}

    def step(name, fn):
        if only and name not in only:
            return
        print(f"\n######## czyszczenie: {name}")
        try:
            logs[name] = fn()
        except Exception:
            print(f"!!! BŁĄD w '{name}':"); traceback.print_exc(limit=4)

    def _trump():
        from .trump_truth import TrumpExplorer
        t = trump or TrumpExplorer(cfg)
        if not len(t.df):
            t.load()
        return t.clean(drop_retweets=drop_retweets)

    def _gkg():
        from .gdelt_gkg import GKGExplorer
        return (gkg or GKGExplorer(cfg)).clean(max_files=gkg_max_files)

    def _gta():
        from .gta_fr import GTAExplorer
        g = GTAExplorer(cfg)
        d = g.load()
        g.df = d[period_mask(cfg, d["date_announced"])].copy()
        print(f"GTA: {len(d):,} interwencji -> {len(g.df):,} w okresie badania")
        return g.clean()

    def _market():
        mk = market
        if mk is None:
            from .market_archive import MarketArchiveExplorer
            mk = MarketArchiveExplorer(cfg)
        return MarketCleaner(cfg, mk).run()

    def _duka():
        if dukascopy is None or not getattr(dukascopy, "raw", None):
            print("Dukascopy: brak pobranych próbek (uruchom sekcję 7), pomijam.")
            return pd.DataFrame()
        return dukascopy.clean()

    if Path(cfg.fnspid_clean_csv).exists():
        step("FNSPID", lambda: FNSPIDCleaner(cfg).run())
    else:
        print(f"FNSPID: brak {cfg.fnspid_clean_csv}, pomijam.")
    step("Trump", _trump)
    if Path(cfg.gkg_cache_dir).exists() and any(Path(cfg.gkg_cache_dir).glob("*.gkg.csv.zip")):
        step("GKG", _gkg)
    if Path(cfg.gta_csv).exists():
        step("GTA", _gta)
    if Path(cfg.fr_csv).exists():
        step("FR", lambda: FRCleaner(cfg).run())
    if Path(cfg.market_archive).exists():
        step("Rynek", _market)
    step("Dukascopy", _duka)
    for k, v in logs.items():
        show(v, f"log czyszczenia: {k}")
    return logs


# ======================================================================= audyt wyników
class CleanAudit:
    """Sprawdza pliki w cfg.out_root: liczba wierszy, zakres czasu, udział w okresie, duplikaty znaczników (rynek)."""
    TIME_COLS = ["t_utc", "time_utc", "event_date", "publication_date", "date_announced"]

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        rows = []
        for p in sorted(Path(self.cfg.out_root).rglob("*.csv")):
            if "cleaning_log" in p.name:
                continue
            hdr = list(pd.read_csv(p, nrows=0).columns)
            tc = next((c for c in self.TIME_COLS if c in hdr), None)
            if tc is None:
                rows.append({"plik": str(p.relative_to(self.cfg.out_root)), "uwaga": "brak kolumny czasu"})
                continue
            t = pd.to_datetime(pd.read_csv(p, usecols=[tc])[tc], utc=True, errors="coerce")
            series = p.parent.name in ("market", "dukascopy")
            rows.append({
                "plik": str(p.relative_to(self.cfg.out_root)), "wiersze": len(t), "kolumna_czasu": tc,
                "od": t.min(), "do": t.max(), "NaT": int(t.isna().sum()),
                "w_okresie_%": round(100 * float(period_mask(self.cfg, t).mean()), 1) if len(t) else None,
                "zduplikowane_znaczniki": int(t.duplicated().sum()) if series else None,
                "rosnący": bool(t.is_monotonic_increasing) if series else None})
        res = pd.DataFrame(rows)
        show(res, "AUDYT plików po czyszczeniu")
        bad = res[(res.get("NaT", 0) > 0) | (res.get("zduplikowane_znaczniki", 0) > 0) | (res.get("rosnący", True) == False)] \
            if len(res) and "NaT" in res else pd.DataFrame()
        print("\nPROBLEMY:" if len(bad) else "\nAudyt: brak NaT, duplikatów znaczników ani nieposortowanych szeregów.")
        if len(bad):
            show(bad)
        return res
