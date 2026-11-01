"""FNSPID (nagłówki/artykuły finansowe powiązane z tickerami). Plik jest ogromny, więc wszystko jest strumieniowe."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import re

from .base import CleaningLog, Config, TableProfiler, parse_utc, show


class FNSPIDExplorer:
    CORE_COLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author"]
    TEXT_COLS = ["Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]

    def __init__(self, cfg: Config, chunksize: int = 200_000):
        self.cfg = cfg
        self.path = cfg.fnspid_csv
        self.chunksize = chunksize
        self.out_dir = cfg.out("fnspid")
        self.stats: dict = {}
        self.sample = pd.DataFrame()

    # ------------------------------------------------------------------ szybki wgląd
    def schema(self) -> pd.DataFrame:
        cols = list(pd.read_csv(self.path, nrows=0).columns)
        expected = set(self.CORE_COLS + self.TEXT_COLS)
        df = pd.DataFrame({"kolumna": cols, "oczekiwana": [c in expected for c in cols]})
        missing = [c for c in self.CORE_COLS if c not in cols]
        if missing:
            print("UWAGA: brak oczekiwanych kolumn:", missing)
        return df

    def peek(self, n: int = 5, width: int = 100) -> pd.DataFrame:
        """Pierwsze n wierszy z obciętym tekstem (żeby nie zalać ekranu artykułem)."""
        df = pd.read_csv(self.path, nrows=n, dtype=str)
        for c in df.columns:
            df[c] = df[c].astype(str).str.slice(0, width)
        return df

    # ------------------------------------------------------------------ skan strumieniowy
    def scan(self, max_chunks: int | None = None, article_stats: bool = False,
             check_dups: bool = True, sample_frac: float = 0.002, seed: int = 0):
        """Jeden przebieg po pliku: braki, zakres dat, liczby per rok/wydawca/ticker, duplikaty.

        max_chunks=None -> cały plik. UWAGA: przy max_chunks dostajemy tylko POCZĄTEK pliku,
        a plik może być posortowany po tickerze, więc to próbka obciążona.
        """
        header = list(pd.read_csv(self.path, nrows=0).columns)
        use = [c for c in self.CORE_COLS if c in header]
        if article_stats and "Article" in header:
            use.append("Article")

        n_rows = 0
        nulls = Counter()
        years, publishers, symbols = Counter(), Counter(), Counter()
        period_counts = Counter()
        dmin, dmax = None, None
        bad_dates = 0
        title_short = 0
        art_empty, art_len_sum = 0, 0
        hashes = []
        samples = []
        bounds = self.cfg.period_bounds()
        names = list(self.cfg.periods.keys())

        reader = pd.read_csv(self.path, usecols=use, dtype=str, chunksize=self.chunksize)
        for i, ch in enumerate(reader):
            if max_chunks is not None and i >= max_chunks:
                break
            n_rows += len(ch)
            for c in ch.columns:
                nulls[c] += int(ch[c].isna().sum() + (ch[c].fillna("x").str.strip() == "").sum())
            dt = parse_utc(ch["Date"])
            bad_dates += int(dt.isna().sum())
            ok = dt.dropna()
            if len(ok):
                dmin = ok.min() if dmin is None else min(dmin, ok.min())
                dmax = ok.max() if dmax is None else max(dmax, ok.max())
                years.update(ok.dt.year.tolist())
                for nm, (a, b) in zip(names, bounds):
                    period_counts[nm] += int(((ok >= a) & (ok < b)).sum())
            if "Publisher" in ch:
                publishers.update(ch["Publisher"].fillna("(brak)").str.strip().tolist())
            if "Stock_symbol" in ch:
                symbols.update(ch["Stock_symbol"].fillna("(brak)").str.strip().str.upper().tolist())
            if "Article_title" in ch:
                title_short += int((ch["Article_title"].fillna("").str.strip().str.len() < 10).sum())
            if "Article" in ch:
                ln = ch["Article"].fillna("").str.len()
                art_empty += int((ln == 0).sum())
                art_len_sum += int(ln.sum())
            if check_dups and "Article_title" in ch:
                key = pd.DataFrame({"t": ch["Article_title"].fillna("").str.lower().str.strip(),
                                    "d": dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                                    "s": ch["Stock_symbol"].fillna("").str.upper().str.strip()})
                hashes.append(pd.util.hash_pandas_object(key, index=False).to_numpy())
            samples.append(ch.drop(columns=["Article"], errors="ignore").sample(frac=sample_frac, random_state=seed))

        n_dup = None
        if check_dups and hashes:
            h = np.concatenate(hashes)
            n_dup = int(len(h) - len(np.unique(h)))

        self.stats = {
            "wiersze_przeskanowane": n_rows,
            "pełny_plik": max_chunks is None,
            "data_min": dmin, "data_max": dmax,
            "daty_nieparsowalne": bad_dates,
            "braki_lub_puste_per_kolumna": dict(nulls),
            "tytuły_<10_znaków": title_short,
            "duplikaty(tytuł,data,ticker)": n_dup,
            "unikalne_tickery": len(symbols),
            "unikalni_wydawcy": len(publishers),
            "wiersze_per_rok": dict(sorted(years.items())),
            "wiersze_w_okresach_badania": dict(period_counts),
        }
        if article_stats:
            self.stats["Article_puste"] = art_empty
            self.stats["Article_średnia_długość"] = round(art_len_sum / max(n_rows, 1), 1)
        self.top_publishers = pd.Series(publishers).sort_values(ascending=False).head(25).rename("n")
        self.top_symbols = pd.Series(symbols).sort_values(ascending=False).head(25).rename("n")
        self.sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        self.sample["date_utc"] = parse_utc(self.sample["Date"]) if "Date" in self.sample else pd.NaT
        return self.stats

    def report(self):
        if not self.stats:
            print("Najpierw uruchom scan().")
            return
        show(self.stats, "FNSPID: statystyki skanu")
        show(self.top_publishers, "Top wydawcy")
        show(self.top_symbols, "Top tickery")
        show(TableProfiler.column_overview(self.sample.drop(columns=["date_utc"], errors="ignore")),
             f"Przegląd kolumn na losowej próbce ({len(self.sample):,} wierszy)")
        print("\nPytania do odpowiedzi po obejrzeniu: czy dane obejmują 2018-2019 i 2025? "
              "czy Date ma strefę? czy Publisher/Author mają dużo braków? ile duplikatów?")

    # ------------------------------------------------------------------ czyszczenie
    def clean(self, periods: dict | None = None, keep_article: bool = False,
              title_regex: str | None = None, out_name: str = "fnspid_clean.csv") -> pd.DataFrame:
        """Strumieniowe czyszczenie. Domyślnie tylko okresy badania (cfg.periods), bez treści Article.

        title_regex (opcjonalnie) to SELEKCJA tematyczna, nie czyszczenie, więc domyślnie wyłączona.
        Deduplikacja po (tytuł, data co do sekundy, ticker) wymaga pamięci na zbiór hashy kept-wierszy,
        dlatego zawężanie do okresów jest ważne.
        """
        cfg_periods = periods or self.cfg.periods
        bounds = [(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1))
                  for s, e in cfg_periods.values()]
        header = list(pd.read_csv(self.path, nrows=0).columns)
        use = [c for c in self.CORE_COLS if c in header]
        if keep_article and "Article" in header:
            use.append("Article")

        out_path = self.out_dir / out_name
        if out_path.exists():
            out_path.unlink()
        log = CleaningLog("FNSPID")
        tot = dict(read=0, bad_date=0, out_of_period=0, empty_title=0, dup=0, regex=0, kept=0)
        seen: set = set()
        first = True

        for ch in pd.read_csv(self.path, usecols=use, dtype=str, chunksize=self.chunksize):
            tot["read"] += len(ch)
            for c in ch.columns:
                ch[c] = ch[c].str.strip()
            ch["date_utc"] = parse_utc(ch["Date"])
            n = len(ch); ch = ch[ch["date_utc"].notna()]; tot["bad_date"] += n - len(ch)
            mask = np.zeros(len(ch), dtype=bool)
            for a, b in bounds:
                mask |= ((ch["date_utc"] >= a) & (ch["date_utc"] < b)).to_numpy()
            n = len(ch); ch = ch[mask]; tot["out_of_period"] += n - len(ch)
            n = len(ch); ch = ch[ch["Article_title"].fillna("") != ""]; tot["empty_title"] += n - len(ch)
            if "Stock_symbol" in ch:
                ch["Stock_symbol"] = ch["Stock_symbol"].str.upper()
            if title_regex:
                n = len(ch); ch = ch[ch["Article_title"].str.contains(title_regex, case=False, regex=True, na=False)]
                tot["regex"] += n - len(ch)
            key = pd.DataFrame({"t": ch["Article_title"].str.lower(),
                                "d": ch["date_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                                "s": ch.get("Stock_symbol", pd.Series("", index=ch.index)).fillna("")})
            h = pd.util.hash_pandas_object(key, index=False).to_numpy()
            keep = np.array([x not in seen for x in h], dtype=bool)
            # duplikaty wewnątrz paczki
            _, first_idx = np.unique(h, return_index=True)
            in_chunk = np.zeros(len(h), dtype=bool); in_chunk[first_idx] = True
            keep &= in_chunk
            seen.update(h[keep].tolist())
            tot["dup"] += int((~keep).sum())
            ch = ch[keep].drop(columns=["Date"])
            tot["kept"] += len(ch)
            ch.to_csv(out_path, mode="a", header=first, index=False)
            first = False

        log.add_count("wczytane wiersze", 0, str(tot["read"]))
        log.add_count("nieparsowalna data", tot["bad_date"])
        log.add_count("poza okresami badania", tot["out_of_period"], str(list(cfg_periods)))
        log.add_count("pusty tytuł", tot["empty_title"])
        if title_regex:
            log.add_count("filtr tematyczny title_regex", tot["regex"], title_regex)
        log.add_count("duplikaty (tytuł, data, ticker)", tot["dup"])
        log.add_count("ZOSTAJE", 0, str(tot["kept"]))
        log.save(self.out_dir / "fnspid_cleaning_log.csv")
        print(f"Zapisano {tot['kept']:,} wierszy -> {out_path}")
        return log.to_frame()


class FNSPIDTopics:
    """Tagowanie wyczyszczonego FNSPID: temat (słownik z granicami słów), wątek, notki zbiorcze, niezależne historie."""
    TOPIC_RE = re.compile(
        r"\b(?:tariffs?|duty|duties|levy|levies|trade\s+wars?|trade\s+disputes?|trade\s+deals?|trade\s+talks?|"
        r"trade\s+tensions?|trade\s+deficits?|trade\s+agreements?|unfair\s+trade|retaliat\w*|countermeasures?|"
        r"protectionis\w*|nafta|usmca|section\s*(?:232|301)|sanctions?|embargo|anti-?dumping|quotas?)\b", re.I)
    THREADS = [
        ("USA-Chiny", re.compile(r"\bchin(?:a|ese)\b|\bbeijing\b|\bhuawei\b|\bzte\b|\byuan\b", re.I)),
        ("Meksyk/Kanada", re.compile(r"\bmexic\w*|\bcanad\w*|\bnafta\b|\busmca\b|\btrudeau\b", re.I)),
        ("UE", re.compile(r"\bE\.?U\.?\b|european union|\beurope\w*|\bjuncker\b", re.I)),
        ("Japonia/Korea/Indie", re.compile(r"\bjapan\w*|\bkorea\w*|\bindia\w*", re.I)),
    ]
    TICKER_THREAD = {"EWW": "Meksyk/Kanada", "EWC": "Meksyk/Kanada", "BABA": "USA-Chiny", "FXI": "USA-Chiny",
                     "ASHR": "USA-Chiny", "ATHM": "USA-Chiny", "EWJ": "Japonia/Korea/Indie", "EWY": "Japonia/Korea/Indie"}
    SECTOR_RE = re.compile(r"^shares of (?:several|multiple|some)\b", re.I)

    @classmethod
    def thread(cls, title: pd.Series, sym: pd.Series) -> pd.Series:
        out = pd.Series("inne", index=title.index, dtype=object)
        for name, rx in reversed(cls.THREADS):
            out[title.str.contains(rx)] = name
        tk = sym.fillna("").astype(str).str.upper().map(cls.TICKER_THREAD)
        return out.where(tk.isna() | (out != "inne"), tk)

    @classmethod
    def prepare(cls, df: pd.DataFrame, tcol="Article_title", scol="Stock_symbol", pcol="Publisher", dcol="t") -> pd.DataFrame:
        d = df.copy()
        title = d[tcol].fillna("").astype(str)
        d["publisher_norm"] = d[pcol].fillna("(brak)").astype(str).str.replace("_", " ", regex=False).str.strip()
        d["is_topic"] = title.str.contains(cls.TOPIC_RE)
        d["is_sector_note"] = title.str.contains(cls.SECTOR_RE)
        d["thread"] = cls.thread(title, d[scol])
        d["has_time"] = d[dcol].dt.strftime("%H:%M:%S") != "00:00:00"
        d["day"] = d[dcol].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        norm = title.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.slice(0, 60)
        d["story_key"] = d["day"].dt.strftime("%Y-%m-%d") + "|" + norm      # ta sama historia, wiele tickerów
        return d

    @staticmethod
    def daily(d: pd.DataFrame, days, only_topic: bool = True, independent: bool = True) -> pd.Series:
        x = d[d["is_topic"]] if only_topic else d
        if independent:
            x = x.drop_duplicates("story_key")
        return x.groupby("day").size().reindex(days, fill_value=0)