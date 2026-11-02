"""GDELT GKG 2.1: eksploracja i czyszczenie na cache'u pobranym przez GDELTAnalyzer (gdelt_data/gkg)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CleaningLog, Config, TableProfiler, show


class GKGExplorer:
    GKG_COLS = [
        "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName", "DocumentIdentifier",
        "Counts", "V2Counts", "Themes", "V2Themes", "Locations", "V2Locations", "Persons", "V2Persons",
        "Organizations", "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage", "RelatedImages",
        "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations", "AllNames", "Amounts", "TranslationInfo", "Extras",
    ]
    USECOLS = ["GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName", "DocumentIdentifier",
               "V2Themes", "V2Persons", "V2Organizations", "V2Tone", "TranslationInfo"]
    TONE_COLS = ["tone", "pos", "neg", "polarity", "act_density", "self_density", "wordcount"]
    STRONG_THEMES = ["SANCTIONS", "ECON_TRADE_DISPUTE", "ECON_BOYCOTT", "WB_778_NON_TARIFF_MEASURES",
                     "EPU_CATS_TRADE_POLICY", "WB_776_TRADE_POLICY", "WB_775_TRADE_POLICY_AND_INTEGRATION"]
    TRACKED = ["foxnews.com", "cnn.com", "businessinsider.com", "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com"]
    URL_RE = re.compile(r"tariff|sanction|trade[-_ ]war|export[-_ ]control|embargo|duties", re.I)
    THEME_RE = re.compile(r"(?:^|;)(?:" + "|".join(sorted(map(re.escape, STRONG_THEMES), key=len, reverse=True))
                          + r"),\d+")
    NOISE_SOURCES = {"iheart.com", "joker.com", "animalforum.com", "ehfcl.com", "wkrb13.com", "dailypolitical.com"}

    def __init__(self, cfg: Config, tracked: list[str] | None = None):
        self.cfg = cfg
        self.cache_dir = Path(cfg.gkg_cache_dir)
        self.out_dir = cfg.out("gkg")
        self.tracked = tracked or self.TRACKED
        self.file_stats = pd.DataFrame()
        self.coverage = pd.DataFrame()
        self.sample = pd.DataFrame()
        self.source_counts = pd.Series(dtype="int64")

    # ------------------------------------------------------------------ pomocnicze
    def _files(self) -> list[Path]:
        return sorted(self.cache_dir.glob("*.gkg.csv.zip"))

    def _read(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep="\t", header=None, names=self.GKG_COLS, usecols=self.USECOLS, dtype=str,
                               encoding="utf-8", encoding_errors="replace", quoting=3, on_bad_lines="skip",
                               compression="zip")
        except pd.errors.EmptyDataError:            # paczka odchudzona do cłów, w której nie było żadnego artykułu
            return pd.DataFrame(columns=self.USECOLS, dtype=str)

    @staticmethod
    def _thinned(path: Path) -> bool:
        """Paczka z download_gkg_dense(thin=True): zawiera tylko artykuły o cłach (nie nadaje się do statystyk udziału)."""
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                return z.comment == b"topic-only"
        except (zipfile.BadZipFile, OSError):
            return False

    @classmethod
    def parse_tone(cls, s: pd.Series) -> pd.DataFrame:
        parts = s.fillna("").str.split(",", expand=True)
        n_fields = parts.notna().sum(axis=1) if parts.shape[1] else pd.Series(0, index=s.index)
        parts = parts.reindex(columns=range(7))
        parts.columns = cls.TONE_COLS
        out = parts.apply(pd.to_numeric, errors="coerce")
        out["n_pól_tone"] = n_fields.values if len(n_fields) else []
        return out

    @staticmethod
    def url_key(s: pd.Series) -> pd.Series:
        return s.fillna("").str.lower().str.replace(r"[?#].*$", "", regex=True).str.rstrip("/")

    # ------------------------------------------------------------------ inwentarz plików
    def inventory(self) -> dict:
        files = self._files()
        if not files:
            raise FileNotFoundError(f"Brak plików *.gkg.csv.zip w {self.cache_dir}.")
        s = pd.Series(pd.to_datetime([p.name[:14] for p in files], format="%Y%m%d%H%M%S")).sort_values().reset_index(
            drop=True)
        sizes = np.array([p.stat().st_size for p in files])
        gap = s.diff()
        new_win = gap > pd.Timedelta(days=1)
        win = s.groupby(new_win.cumsum()).agg(od="min", do="max", plików="size")
        win["oczekiwanych_15min"] = ((win["do"] - win["od"]) / pd.Timedelta(minutes=15)).round().astype(int) + 1
        steps = gap[~new_win & gap.notna()]
        info = {"plików": len(files), "od": s.min(), "do": s.max(), "okien_zdarzeń": len(win),
                "typowy_krok_w_oknie": steps.mode().iloc[0] if len(steps) else pd.NaT,
                "pokrycie_siatki_15min_w_oknach_%": round(100 * len(s) / win["oczekiwanych_15min"].sum(), 1),
                "pliki_0_bajtów": int((sizes == 0).sum()),
                "rozmiar_pliku_MB(min/mediana/max)": (round(float(sizes.min()) / 1e6, 2),
                                                      round(float(np.median(sizes)) / 1e6, 2),
                                                      round(float(sizes.max()) / 1e6, 2))}
        show(info, "GKG: inwentarz plików")
        show(win, "GKG: okna (krok ok. 2h15 = próbka równomierna z plan_gkg_stamps, NIE ciągły szereg)")
        return info

    # ------------------------------------------------------------------ skan
    def scan(self, max_files: int | None = None, sample_per_file: int = 300, seed: int = 0):
        """Statystyki per plik + mała losowa próbka wierszy. Nie trzyma całych danych w pamięci."""
        rows, cov, samples = [], [], []
        src_counter: Counter = Counter()
        files = [p for p in self._files() if not self._thinned(p)][:max_files]   # odchudzone paczki zafałszowałyby udziały
        for i, p in enumerate(files, 1):
            try:
                df = self._read(p)
            except Exception as ex:
                rows.append({"plik": p.name, "błąd": str(ex)[:80]})
                continue
            stamp = pd.Timestamp(p.name[:14])
            n = len(df)
            url = df["DocumentIdentifier"].fillna("")
            by_url = url.str.contains(self.URL_RE)
            by_theme = df["V2Themes"].fillna("").str.contains(self.THEME_RE)
            topic = by_url | by_theme
            translated = df["GKGRECORDID"].fillna("").str.contains("-T", regex=False) | df["TranslationInfo"].notna()
            tone = self.parse_tone(df["V2Tone"])
            bad_tone = int(((tone["n_pól_tone"] < 7) | tone[self.TONE_COLS].isna().any(axis=1)).sum())
            rows.append({
                "plik": p.name, "time": stamp, "n": n,
                "url_brak": int(url.eq("").sum()), "źródło_brak": int(df["SourceCommonName"].isna().sum()),
                "przetłumaczone": int(translated.sum()),
                "DATE≠nazwa_pliku": int((df["DATE"] != stamp.strftime("%Y%m%d%H%M%S")).sum()),
                "tone_błędny": bad_tone, "wordcount_0": int((tone["wordcount"] == 0).sum()),
                "temat": int(topic.sum()), "temat_tylko_URL": int((by_url & ~by_theme).sum()),
                "temat_tylko_themes": int((by_theme & ~by_url).sum()),
            })
            src_counter.update(df["SourceCommonName"].dropna().value_counts().to_dict())
            for s in self.tracked:
                m = df["SourceCommonName"].fillna("").str.endswith(s)  # łapie też subdomeny, np. uk.reuters.com
                cov.append({"time": stamp, "source": s, "n_all": int(m.sum()), "n_topic": int((m & topic).sum())})
            smp = pd.concat([df, tone[self.TONE_COLS]], axis=1)
            smp["topic"] = topic.values
            smp["translated"] = translated.values
            samples.append(smp.sample(min(n, sample_per_file), random_state=seed))
            if i % 50 == 0:
                print(f"  przeskanowano {i}/{len(files)} plików")
        self.file_stats = pd.DataFrame(rows)
        self.coverage = pd.DataFrame(cov)
        self.sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        self.source_counts = pd.Series(src_counter).sort_values(ascending=False)
        return self.file_stats

    def daily_rate(self) -> pd.DataFrame:
        """Artykuły o cłach NA PACZKĘ (a nie na dzień): pliki są próbką co ok. 2h15, więc sumy dzienne nie są porównywalne."""
        cov = self.coverage.copy()
        cov["day"] = cov["time"].dt.normalize()
        g = cov.groupby(["day", "source"]).agg(n_topic=("n_topic", "sum"), pliki=("time", "nunique")).reset_index()
        g["na_paczkę"] = g["n_topic"] / g["pliki"]
        return g.pivot(index="day", columns="source", values="na_paczkę")

    def report(self):
        fs = self.file_stats
        if fs.empty:
            print("Najpierw uruchom scan().")
            return
        ok = fs[fs.get("błąd").isna()] if "błąd" in fs else fs
        tot = ok["n"].sum()
        summary = {
            "plików_ze_skanem": len(ok), "plików_z_błędem": len(fs) - len(ok),
            "rekordów_łącznie": int(tot),
            "rekordów_na_plik(min/mediana/max)": (int(ok["n"].min()), int(ok["n"].median()), int(ok["n"].max())),
            "brak_URL_%": round(100 * ok["url_brak"].sum() / tot, 3),
            "brak_źródła_%": round(100 * ok["źródło_brak"].sum() / tot, 3),
            "przetłumaczone_%": round(100 * ok["przetłumaczone"].sum() / tot, 2),
            "DATE≠nazwa_pliku_%": round(100 * ok["DATE≠nazwa_pliku"].sum() / tot, 3),
            "błędny_V2Tone_%": round(100 * ok["tone_błędny"].sum() / tot, 3),
            "wordcount=0_%": round(100 * ok["wordcount_0"].sum() / tot, 3),
            "pasuje_do_filtra_tematu_%": round(100 * ok["temat"].sum() / tot, 2),
            "…z_tego_tylko_przez_URL_%": round(100 * ok["temat_tylko_URL"].sum() / max(ok["temat"].sum(), 1), 1),
            "…z_tego_tylko_przez_themes_%": round(100 * ok["temat_tylko_themes"].sum() / max(ok["temat"].sum(), 1), 1),
        }
        show(summary, "GKG: podsumowanie skanu")
        show(self.source_counts.head(20).rename("rekordów"), "Top 20 źródeł (wszystkie rekordy)")

        if not self.coverage.empty:
            cov = self.coverage.copy()
            cov["day"] = cov["time"].dt.date
            piv = cov.pivot_table(index="day", columns="source", values="n_topic", aggfunc="sum", fill_value=0)
            show(piv, "Artykuły o cłach/sankcjach per dzień per śledzone medium (kontrola pokrycia)")
            zero = (cov.groupby("source")["n_all"].sum() == 0)
            if zero.any():
                print("UWAGA: zero rekordów dla:", list(zero[zero].index), "(zła nazwa domeny albo medium nieindeksowane)")

        if not self.sample.empty:
            t = self.sample
            show(t[self.TONE_COLS].describe().T.round(2), "V2Tone na próbce (tone powinien być ok. -10..+10, wordcount > 0)")
            hump = t[t["tone"] > 9]
            if len(hump):
                show(hump["SourceCommonName"].value_counts().head(10).rename("n"), "Źródła w garbie tonu > +9 (komunikaty prasowe/spam?)")
            dup_share = 1 - self.url_key(t["DocumentIdentifier"]).nunique() / max(len(t), 1)
            print(f"\nDuplikaty URL w próbce: {dup_share:.2%} (dolna granica; prawdziwe duplikaty są między plikami)")
            show(t[t["topic"]][["time" if "time" in t else "DATE", "SourceCommonName", "DocumentIdentifier", "tone"]].head(8),
                 "Przykładowe artykuły pasujące do filtra tematu")

    # ------------------------------------------------------------------ czyszczenie
    def clean(self, topic_only: bool = True, drop_translated: bool = True, max_files: int | None = None,
              out_name: str = "gkg_clean.csv") -> pd.DataFrame:
        """Strumieniowo: parsowanie czasu i tonu, filtr tematu (opcjonalnie), bez tłumaczeń, dedup po URL."""
        out_path = self.out_dir / out_name
        if out_path.exists():
            out_path.unlink()
        files = self._files()[:max_files]
        c = Counter()
        seen: set = set()
        first = True
        for p in files:
            try:
                df = self._read(p)
            except Exception:
                c["pliki_nieczytelne"] += 1
                continue
            dense = self._thinned(p)                 # paczka gęsta (download_gkg_dense), nie z próbki równomiernej
            c["wczytane"] += len(df)
            url = df["DocumentIdentifier"].fillna("")
            if topic_only:
                m = url.str.contains(self.URL_RE) | df["V2Themes"].fillna("").str.contains(self.THEME_RE)
                c["poza_tematem"] += int((~m).sum())
                df = df[m]
            if drop_translated:
                tr = df["GKGRECORDID"].fillna("").str.contains("-T", regex=False) | df["TranslationInfo"].notna()
                c["przetłumaczone"] += int(tr.sum())
                df = df[~tr]
            n0 = len(df)
            df = df[~df["SourceCommonName"].isin(self.NOISE_SOURCES)]
            c["źródła_szumowe"] += n0 - len(df)
            df = df.copy()
            df["time_utc"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
            tone = self.parse_tone(df["V2Tone"])
            df = pd.concat([df.reset_index(drop=True), tone[self.TONE_COLS].reset_index(drop=True)], axis=1)
            bad = df["time_utc"].isna() | df["DocumentIdentifier"].isna() | df["SourceCommonName"].isna() \
                  | df[self.TONE_COLS].isna().any(axis=1) | (df["wordcount"] <= 0)
            c["brak_czasu/URL/źródła/tonu_lub_wordcount<=0"] += int(bad.sum())
            df = df[~bad].copy()
            df["url_key"] = self.url_key(df["DocumentIdentifier"])
            df["file_dense"] = dense
            keep = ~df["url_key"].isin(seen) & ~df["url_key"].duplicated()
            c["duplikaty_URL"] += int((~keep).sum())
            seen.update(df.loc[keep, "url_key"])
            df = df[keep].drop(columns=["V2Themes", "TranslationInfo", "V2Tone", "DATE"], errors="ignore")
            c["zostaje"] += len(df)
            df.to_csv(out_path, mode="a", header=first, index=False)
            first = False
        log = CleaningLog("GKG")
        for k, v in c.items():
            log.add_count(k, v)
        log.save(self.out_dir / "gkg_cleaning_log.csv")
        print(f"Zapisano {c['zostaje']:,} artykułów -> {out_path}")
        return log.to_frame()
