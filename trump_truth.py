"""Posty Trumpa (Twitter 2018-2019, Truth Social od 2022): wczytanie, ujednolicenie, eksploracja, czyszczenie."""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CleaningLog, Config, TableProfiler, parse_utc, read_table, show

NY = "America/New_York"


class TrumpExplorer:
    TIME_CANDS = ["created_at", "date", "datetime", "timestamp", "time", "posted_at", "created"]
    TEXT_CANDS = ["text", "content", "full_text", "tweet", "tweet_text", "post", "body", "message"]
    ID_CANDS = ["id", "tweet_id", "post_id", "status_id"]
    FAV_CANDS = ["favorites", "favourites_count", "favorite_count", "favorites_count", "likes", "like_count"]
    RT_CANDS = ["retweets", "reblogs_count", "retweet_count", "reposts", "reposts_count", "shares"]
    RTFLAG_CANDS = ["isretweet", "is_retweet", "retweeted"]

    # SŁOWNIK ŚCISŁY: granice słów (\b), więc "IMPORTANT" nie pasuje do "import".
    TOPIC_RE = re.compile(
        r"\b(?:tariffs?|duty|duties|trade\s+wars?|trade\s+deals?|trade\s+talks?|trade\s+negotiat\w*|"
        r"trade\s+deficits?|trade\s+agreements?|trade\s+barriers?|unfair\s+trade|retaliat\w*|protectionis\w*|"
        r"nafta|usmca|section\s*(?:232|301)|sanctions?|embargo|currency\s+manipulat\w*|wto|anti-?dumping|"
        r"countervailing|reciprocal)\b", re.I)
    # tweety typu "China talks are going very well!": Chiny + kontekst handlowy
    CHINA_RE = re.compile(r"\bchin(?:a|ese)\b|\bxi\b|\bbeijing\b|\bhuawei\b|\bzte\b", re.I)
    TRADE_CTX_RE = re.compile(
        r"\b(?:talks?|deals?|trade|tariffs?|farmers?|soybeans?|negotiat\w*|agreements?|purchas\w*)\b", re.I)
    # wątki: pierwszy pasujący wygrywa
    THREADS = [
        ("USA-Chiny", re.compile(r"\bchin(?:a|ese)\b|\bxi\b|\bbeijing\b|\bhuawei\b|\bzte\b", re.I)),
        ("Meksyk/Kanada", re.compile(r"\bmexic\w*|\bcanad\w*|\bnafta\b|\busmca\b|\btrudeau\b", re.I)),
        ("UE", re.compile(r"\bE\.?U\.?\b|european union|\beurope\w*|\bgerman\w*|\bjuncker\b|\bmacron\b", re.I)),
        ("Japonia/Korea/Indie", re.compile(r"\bjapan\w*|\babe\b|\bkorea\w*|\bindia\w*|\bmodi\b", re.I)),
        ("Stal/aluminium", re.compile(r"\bsteel\b|\balumin(?:um|ium)\b", re.I)),
    ]
    TAG_RE = re.compile(r"<[^>]+>")
    URL_RE = re.compile(r"https?://\S+")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = cfg.out("trump")
        self.raw: dict[str, pd.DataFrame] = {}
        self.df = pd.DataFrame()
        self.missing: list[str] = []
        self.results: dict = {}

    # ------------------------------------------------------------------ wczytanie
    @staticmethod
    def _pick(low: dict, cands: list[str]):
        for c in cands:
            if c in low:
                return low[c]
        return None

    @classmethod
    def clean_text(cls, s: pd.Series) -> pd.Series:
        t = s.astype("string")
        t = t.str.replace(r"<br\s*/?>", " ", regex=True, case=False)
        t = t.str.replace(cls.TAG_RE, " ", regex=True)
        t = t.map(lambda x: html.unescape(x) if isinstance(x, str) else x).astype("string")
        return t.str.replace(r"\s+", " ", regex=True).str.strip()

    def standardize(self, raw: pd.DataFrame, source: str) -> pd.DataFrame:
        low = {str(c).strip().lower(): c for c in raw.columns}
        tcol, xcol = self._pick(low, self.TIME_CANDS), self._pick(low, self.TEXT_CANDS)
        if tcol is None or xcol is None:
            raise ValueError(f"[{source}] nie rozpoznano kolumny czasu/tekstu. Kolumny pliku: {list(raw.columns)}")
        icol, fcol = self._pick(low, self.ID_CANDS), self._pick(low, self.FAV_CANDS)
        rcol, flag = self._pick(low, self.RT_CANDS), self._pick(low, self.RTFLAG_CANDS)
        tser = raw[tcol].astype(str)
        if str(tcol).strip().lower() == "date" and "time" in low:
            tser = tser + " " + raw[low["time"]].astype(str)
        text_raw = raw[xcol].astype("string")
        text = self.clean_text(raw[xcol])
        is_rt = text.str.match(r"^RT\b", na=False)
        if flag is not None:
            is_rt = is_rt | raw[flag].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        out = pd.DataFrame({
            "source": source,
            "post_id": raw[icol].astype(str) if icol else pd.Series(np.nan, index=raw.index),
            "time_utc": parse_utc(tser),
            "text": text,
            "text_len": text.str.len(),
            "favs": pd.to_numeric(raw[fcol], errors="coerce") if fcol else np.nan,
            "rts": pd.to_numeric(raw[rcol], errors="coerce") if rcol else np.nan,
            "is_retweet": is_rt.astype(bool),
            "had_html": text_raw.str.contains(self.TAG_RE, na=False),
        })
        out.attrs["colmap"] = {"czas": tcol, "tekst": xcol, "id": icol, "polubienia": fcol,
                               "retweety": rcol, "flaga_rt": flag}
        return out

    @classmethod
    def _thread(cls, text: pd.Series, mask: pd.Series) -> pd.Series:
        out = pd.Series("", index=text.index, dtype=object)
        out[mask] = "inne"
        for name, rx in reversed(cls.THREADS):          # od końca: pierwszy na liście wygrywa
            out[mask & text.str.contains(rx)] = name
        return out

    @staticmethod
    def _session(ny: pd.Series) -> pd.Series:
        """Sesja giełdy USA (czas NY): weekend / noc / przed_otwarciem / sesja / po_sesji."""
        wd = ny.dt.dayofweek.to_numpy()
        mins = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
        s = np.select([wd >= 5, mins < 240, mins < 570, mins < 960],
                      ["weekend", "noc", "przed_otwarciem", "sesja"], "po_sesji")
        return pd.Series(s, index=ny.index).where(ny.notna())

    def load(self) -> pd.DataFrame:
        parts, self.missing = [], []
        for name, path in self.cfg.trump_sources.items():
            if not Path(path).exists():
                self.missing.append(f"{name}: brak pliku {path}")
                continue
            raw = read_table(path)
            self.raw[name] = raw
            try:
                std = self.standardize(raw, name)
            except ValueError as ex:
                print("UWAGA:", ex)
                continue
            self.results.setdefault("colmap", {})[name] = std.attrs["colmap"]
            parts.append(std)
        for m in self.missing:
            print("UWAGA:", m)
        self.df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(self.df):
            t = self.df["text"].fillna("")
            self.df["relevant_strict"] = t.str.contains(self.TOPIC_RE)
            self.df["china_trade"] = t.str.contains(self.CHINA_RE) & t.str.contains(self.TRADE_CTX_RE)
            self.df["relevant"] = self.df["relevant_strict"] | self.df["china_trade"]
            self.df["thread"] = self._thread(t, self.df["relevant"])
            ny = self.df["time_utc"].dt.tz_convert(NY)
            self.df["ny_hour"] = ny.dt.hour
            self.df["session"] = self._session(ny)
            self.df["in_period"] = self._period_mask(self.df["time_utc"])
        return self.df

    def _period_mask(self, t: pd.Series) -> pd.Series:
        m = pd.Series(False, index=t.index)
        for a, b in self.cfg.period_bounds():
            m |= (t >= a) & (t < b)
        return m

    # ------------------------------------------------------------------ agregaty
    def monthly(self, only_period: bool = True, only_original: bool = True) -> pd.DataFrame:
        d = self.df[self.df["in_period"]] if only_period else self.df
        if only_original:
            d = d[~d["is_retweet"]]
        d = d.dropna(subset=["time_utc"])
        m = d.assign(month=d["time_utc"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
        g = m.groupby(["month", "source"]).agg(posty=("text", "size"), tematyczne=("relevant", "sum")).reset_index()
        g["udział_%"] = (100 * g["tematyczne"] / g["posty"]).round(2)
        return g

    def hourly(self, tz: str = NY, only_original: bool = True, only_topic: bool = False) -> pd.Series:
        d = self.df[self.df["in_period"]].dropna(subset=["time_utc"])
        if only_original:
            d = d[~d["is_retweet"]]
        if only_topic:
            d = d[d["relevant"]]
        return d["time_utc"].dt.tz_convert(tz).dt.hour.value_counts().sort_index().rename(f"posty wg godziny ({tz})")

    def check_timezone(self, source: str = "twitter"):
        """Strefa znaczników: oryginalne tweety Trumpa mają szczyt ok. 7-10 rano czasu NY,
        więc jeśli po konwersji UTC->NY szczyt wypada poza 6-10, założenie UTC jest podejrzane."""
        d = self.df[(self.df["source"] == source) & self.df["in_period"] & ~self.df["is_retweet"]]
        if d.empty:
            print(f"check_timezone: brak postów [{source}] w okresie")
            return None
        h = d["ny_hour"].value_counts(normalize=True).sort_index()
        peak = int(h.idxmax())
        res = {"szczyt_godz_NY": peak, "top3_godz_NY": h.sort_values(ascending=False).head(3).index.tolist(),
               "udział_6-10_NY_%": round(100 * float(h.loc[6:10].sum()), 1),
               "wniosek": "zgodne z założeniem UTC" if 6 <= peak <= 10 else "PODEJRZANE: sprawdź strefę kolumny date"}
        show(res, f"Trump [{source}]: kontrola strefy czasowej")
        return res

    # ------------------------------------------------------------------ eksploracja
    def explore(self):
        d = self.df if len(self.df) else self.load()
        if d.empty:
            print("Brak danych Trumpa: wgraj pliki wskazane w cfg.trump_sources.")
            return
        show(self.results.get("colmap", {}), "Trump: rozpoznane kolumny")
        for name, raw in self.raw.items():
            show(raw.head(3), f"{name}: head (surowe)")
            show(TableProfiler.column_overview(raw), f"{name}: przegląd kolumn (surowe)")
        for name, g in d.groupby("source"):
            ip = g[g["in_period"]]
            ipo = ip[~ip["is_retweet"]]
            dup_id = int(g["post_id"].duplicated().sum()) if g["post_id"].notna().any() else None
            checks = {
                "wiersze": len(g),
                "czas": TableProfiler.datetime_summary(g["time_utc"]),
                "w okresie badania": len(ip),
                "w okresie, bez retweetów": len(ipo),
                "pusty tekst": int((g["text"].fillna("") == "").sum()),
                "nieparsowalny czas": int(g["time_utc"].isna().sum()),
                "duplikaty post_id": dup_id,
                "duplikaty (czas, tekst)": int(g.duplicated(["time_utc", "text"]).sum()),
                "retweety/ReTruth %": round(100 * float(g["is_retweet"].mean()), 1),
                "tylko link (bez treści) %": round(100 * float(g["text"].fillna("").str.replace(self.URL_RE, "", regex=True)
                                                           .str.strip().eq("").mean()), 1),
                "z HTML w treści %": round(100 * float(g["had_html"].mean()), 1),
                "tematyczne ściśle % (okres, bez RT)": round(100 * float(ipo["relevant_strict"].mean()), 1) if len(ipo) else None,
                "tematyczne szeroko % (okres, bez RT)": round(100 * float(ipo["relevant"].mean()), 1) if len(ipo) else None,
            }
            show(checks, f"Trump [{name}]: kontrole jakości")
            show(TableProfiler.yearly_counts(g["time_utc"]), f"Trump [{name}]: posty per rok")
            if len(ip) == 0:
                print(f"UWAGA: źródło '{name}' nie ma ŻADNYCH postów w okresach {list(self.cfg.periods)}. "
                      "(Truth Social ruszył w 2022; dla 2018-2019 potrzebne jest archiwum Twittera.)")
                continue
            show(ip["text_len"].describe().round(1).rename("długość tekstu"), f"Trump [{name}]: długość postu (okres)")
            rel = ipo[ipo["relevant"]]
            if len(rel):
                show(rel["thread"].value_counts().rename("posty"), f"Trump [{name}]: wątki")
                smp = rel.sample(min(12, len(rel)), random_state=0)[["time_utc", "thread", "text"]]
                show(smp.assign(text=smp["text"].str.slice(0, 110)),
                     f"Trump [{name}]: LOSOWE posty o cłach/handlu (oceń szum ręcznie)")
            if name == "twitter":
                self.check_timezone(name)
        self.results["explored"] = True

    # ------------------------------------------------------------------ czyszczenie
    def clean(self, drop_retweets: bool = False, out_name: str = "trump_clean.csv") -> pd.DataFrame:
        d = (self.df if len(self.df) else self.load()).copy()
        log = CleaningLog("Trump")
        if d.empty:
            return log.to_frame()
        n = len(d); d = d.dropna(subset=["time_utc"]); log.add("nieparsowalny czas", n, len(d))
        n = len(d); d = d[d["in_period"]]; log.add("poza okresami badania", n, len(d), str(list(self.cfg.periods)))
        n = len(d); d = d[d["text"].fillna("") != ""]; log.add("pusty tekst", n, len(d))
        n = len(d)
        d = d.drop_duplicates(["source", "post_id"]) if d["post_id"].notna().any() else d
        d = d.drop_duplicates(["source", "time_utc", "text"])
        log.add("duplikaty (id lub czas+tekst)", n, len(d))
        d["flag_tylko_link"] = d["text"].str.replace(self.URL_RE, "", regex=True).str.strip().eq("")
        log.add_count("FLAGA tylko link (zostaje)", int(d["flag_tylko_link"].sum()))
        log.add_count("FLAGA retweet/ReTruth (zostaje)", int(d["is_retweet"].sum()))
        if drop_retweets:
            n = len(d); d = d[~d["is_retweet"]]; log.add("retweety (drop_retweets=True)", n, len(d))
        log.add_count("FLAGA relevant_strict", int(d["relevant_strict"].sum()), "słownik z granicami słów")
        log.add_count("FLAGA relevant (ściśle + Chiny w kontekście handlu)", int(d["relevant"].sum()))
        d = d.sort_values("time_utc").drop(columns=["in_period", "had_html"])
        d.to_csv(self.out_dir / out_name, index=False)
        log.save(self.out_dir / "trump_cleaning_log.csv")
        print(f"Zapisano {len(d):,} postów -> {self.out_dir / out_name}")
        return log.to_frame()