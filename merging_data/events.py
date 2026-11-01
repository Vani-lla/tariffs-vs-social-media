"""Jedna tabela ZDARZEŃ z czystych źródeł danych (bez klasyfikacji tekstowej).

Rodzaje zdarzeń (kolumna kind):
  trump_post      każdy oryginalny post o cłach (precyzja co do sekundy),
  trump_episode   seria postów: nowy epizod po przerwie >= gap (domyślnie 6 h); t = pierwszy post,
  gta_announced   dzień x kraj wdrażający: ogłoszenia interwencji GTA (precyzja dzienna),
  gta_implemented dzień x kraj wdrażający: wejście w życie,
  fr              dokument Federal Register (data podpisu, a gdy brak, publikacji),
  fr_effective    dokument Federal Register: data wejścia w życie.
Wątek (thread) pochodzi ze słownika słów kluczowych (kraj/produkt), nie z analizy znaczenia tekstu.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cleaning_data.base import Config
from .features import eff_day
from .loaders import as_bool, naive_day
from cleaning_data.trump_truth import TrumpExplorer

NY = "America/New_York"
COLS = ["event_id", "kind", "t", "precision", "date_source", "eff_day", "thread", "threads",
        "weight", "n_posts", "favs", "rts", "session", "ny_hour", "title", "ref_id"]

OFFICIAL_THREAD_RX = [
    ("USA-Chiny", r"china|hong kong"),
    ("Meksyk/Kanada", r"mexic|canad"),
    ("UE", r"european|germany|france|ital|spain|netherland|belgium|austria|poland|sweden|ireland|denmark|finland|portugal|greece|czech"),
    ("Japonia/Korea/Indie", r"japan|korea|india"),
]


THREAD_ORDER = [n for n, _ in OFFICIAL_THREAD_RX]


def _join_threads(values) -> str:
    """Wątki w kolejności PRIORYTETU (USA-Chiny pierwsze), nie alfabetycznie."""
    found = {x for v in values for x in str(v).split("|") if x}
    return "|".join(sorted(found, key=THREAD_ORDER.index))


def _main_thread(threads: str) -> str:
    """Jeden wątek zdarzenia: >= 3 regiony naraz to decyzja wielostronna (np. reklasyfikacja taryfowa dla wszystkich)."""
    parts = [x for x in str(threads).split("|") if x]
    if not parts:
        return "inne"
    return "wielostronne" if len(parts) >= 3 else parts[0]


def official_threads(affected, actor) -> str:
    s = str(affected).lower()
    out = [n for n, rx in OFFICIAL_THREAD_RX if re.search(rx, s)]
    if str(actor) == "CHN" and "USA-Chiny" not in out:
        out.insert(0, "USA-Chiny")
    return "|".join(out)


def _frame(n: int, **kw) -> pd.DataFrame:
    d = {}
    for k, v in kw.items():
        if isinstance(v, pd.Series):
            d[k] = v.reset_index(drop=True)
        elif isinstance(v, (pd.Index, np.ndarray, list)):
            d[k] = pd.Series(v)
        else:
            d[k] = v
    return pd.DataFrame(d, index=pd.RangeIndex(n)).reindex(columns=COLS)


def _session(t: pd.Series) -> pd.Series:
    return TrumpExplorer._session(pd.Series(t).reset_index(drop=True).dt.tz_convert(NY))


class EventBuilder:
    def __init__(self, cfg: Config, gap: str = "6h", drop_calendar: bool = True, core_only: bool = True):
        self.cfg, self.gap, self.drop_calendar, self.core_only = cfg, pd.Timedelta(gap), drop_calendar, core_only
        b = cfg.period_bounds()
        self.lo, self.hi = min(a for a, _ in b), max(e for _, e in b)
        self.lo_d, self.hi_d = self.lo.tz_localize(None), self.hi.tz_localize(None)

    # ------------------------------------------------------------------ Trump
    def trump_posts(self, tr: pd.DataFrame) -> pd.DataFrame:
        m = tr["relevant"] & ~tr["is_retweet"] & (tr["time_utc"] >= self.lo) & (tr["time_utc"] < self.hi)
        p = tr[m].sort_values("time_utc").reset_index(drop=True).copy()
        p["eff_day"] = eff_day(p["time_utc"])
        p["session"] = _session(p["time_utc"])
        p["ny_hour"] = p["time_utc"].dt.tz_convert(NY).dt.hour
        return p

    def trump_episodes(self, posts: pd.DataFrame) -> pd.DataFrame:
        if posts.empty:
            return pd.DataFrame()
        d = posts["time_utc"].diff()
        ep = ((d >= self.gap) | d.isna()).cumsum()
        g = posts.groupby(ep)
        e = pd.DataFrame({
            "t": g["time_utc"].min(), "t_last": g["time_utc"].max(), "n_posts": g.size(),
            "favs": g["favs"].sum(min_count=1), "rts": g["rts"].sum(min_count=1), "first_id": g["post_id"].first(),
            "thread": g["thread"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""),
            "text": g["text"].agg(lambda s: " || ".join(s.astype(str))),
        }).reset_index(drop=True)
        e["eff_day"] = eff_day(e["t"])
        e["session"] = _session(e["t"])
        e["ny_hour"] = e["t"].dt.tz_convert(NY).dt.hour
        return e

    @staticmethod
    def _u_posts(p: pd.DataFrame) -> pd.DataFrame:
        return _frame(len(p), event_id="tw_" + p["post_id"].astype(str), kind="trump_post", t=p["time_utc"],
                      precision="exact", date_source="exact", eff_day=p["eff_day"], thread=p["thread"], threads=p["thread"],
                      weight=p["favs"], n_posts=1, favs=p["favs"], rts=p["rts"],
                      session=p["session"], ny_hour=p["ny_hour"], title=p["text"].str.slice(0, 140), ref_id=p["post_id"])

    @staticmethod
    def _u_eps(e: pd.DataFrame) -> pd.DataFrame:
        if e.empty:
            return pd.DataFrame(columns=COLS)
        return _frame(len(e), event_id="ep_" + e["first_id"].astype(str), kind="trump_episode", t=e["t"], precision="exact",
                      date_source="exact", eff_day=e["eff_day"], thread=e["thread"], threads=e["thread"],
                      weight=e["favs"], n_posts=e["n_posts"], favs=e["favs"],
                      rts=e["rts"], session=e["session"], ny_hour=e["ny_hour"], title=e["text"].str.slice(0, 140),
                      ref_id=e["first_id"])

    # ------------------------------------------------------------------ GTA
    def gta(self, g: pd.DataFrame, which: str = "announced") -> pd.DataFrame:
        col = {"announced": "date_announced", "implemented": "date_implemented"}[which]
        if col not in g:
            return pd.DataFrame(columns=COLS)
        d = g.copy()
        d["day"] = naive_day(d[col])
        d = d.dropna(subset=["day"])
        d = d[(d["day"] >= self.lo_d) & (d["day"] < self.hi_d)]
        if self.drop_calendar:                       # 1 stycznia to w GTA zwykle data zastępcza
            d = d[~((d["day"].dt.month == 1) & (d["day"].dt.day == 1))]
        if d.empty:
            return pd.DataFrame(columns=COLS)
        d["_prod"] = pd.to_numeric(d["n_products"], errors="coerce") if "n_products" in d else np.nan
        d["_title"] = d["title"].astype(str) if "title" in d else ""
        d["_id"] = d["intervention_id"] if "intervention_id" in d else np.arange(len(d))
        aff = d["affected"] if "affected" in d else pd.Series("", index=d.index)
        d["_thr"] = [official_threads(a, ac) for a, ac in zip(aff, d["actor"])]
        gr = d.groupby(["day", "actor"]).agg(
            n=("_id", "nunique"), products=("_prod", "sum"),
            threads=("_thr", _join_threads),
            title=("_title", "first")).reset_index()
        pre = "gtaa" if which == "announced" else "gtai"
        return _frame(len(gr), event_id=pre + "_" + gr["day"].dt.strftime("%Y%m%d") + "_" + gr["actor"].astype(str),
                      kind=f"gta_{which}", t=gr["day"].dt.tz_localize("UTC"), precision="day", date_source=col,
                      eff_day=gr["day"], thread=gr["threads"].map(_main_thread), threads=gr["threads"],
                      weight=gr["products"], n_posts=gr["n"],
                      title=gr["actor"].astype(str) + ": " + gr["title"].astype(str).str.slice(0, 110),
                      ref_id=gr["actor"].astype(str))

    # ------------------------------------------------------------------ Federal Register
    def fr(self, f: pd.DataFrame, which: str = "event") -> pd.DataFrame:
        d = f.copy()
        if self.core_only and "core" in d:
            d = d[as_bool(d["core"])]
        d = d.dropna(subset=["document_number"]).drop_duplicates("document_number")
        if which == "event":
            if "event_date" not in d:
                return pd.DataFrame(columns=COLS)
            day = d["event_date"]
            src = d["event_date_source"].astype(str) if "event_date_source" in d else pd.Series("event_date", index=d.index)
            kind, pre = "fr", "fr"
        else:
            if "effective_on" not in d:
                return pd.DataFrame(columns=COLS)
            day, src, kind, pre = d["effective_on"], pd.Series("effective_on", index=d.index), "fr_effective", "fre"
        d = d.assign(_day=naive_day(day), _src=src).dropna(subset=["_day"])
        d = d[(d["_day"] >= self.lo_d) & (d["_day"] < self.hi_d)]
        if d.empty:
            return pd.DataFrame(columns=COLS)
        title = d["title"].fillna("").astype(str) if "title" in d else pd.Series("", index=d.index)
        txt = title + " " + (d["abstract"].fillna("").astype(str) if "abstract" in d else "")
        thr = TrumpExplorer._thread(txt, pd.Series(True, index=txt.index)).replace("", "inne")
        return _frame(len(d), event_id=pre + "_" + d["document_number"].astype(str), kind=kind, t=d["_day"].dt.tz_localize("UTC"),
                      precision="day", date_source=d["_src"], eff_day=d["_day"], thread=thr, threads=thr,
                      title=title.str.slice(0, 140), ref_id=d["document_number"].astype(str))

    # ------------------------------------------------------------------ całość
    def build(self, tr: pd.DataFrame, gta: pd.DataFrame | None = None, fr: pd.DataFrame | None = None) -> dict:
        posts = self.trump_posts(tr)
        eps = self.trump_episodes(posts)
        parts = [self._u_posts(posts), self._u_eps(eps)]
        if gta is not None and len(gta):
            parts += [self.gta(gta, "announced"), self.gta(gta, "implemented")]
        if fr is not None and len(fr):
            parts += [self.fr(fr, "event"), self.fr(fr, "effective")]
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("Brak zdarzeń: sprawdź, czy trump_clean.csv ma posty o cłach w okresie badania.")
        ev = pd.concat(parts, ignore_index=True)
        ev["t"] = pd.to_datetime(ev["t"], utc=True)
        ev["eff_day"] = pd.to_datetime(ev["eff_day"])
        for c in ("weight", "n_posts", "favs", "rts", "ny_hour"):
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
        ev = ev.sort_values(["t", "kind", "event_id"]).reset_index(drop=True)
        dup = ev["event_id"].duplicated()
        if dup.any():
            raise ValueError(f"Zduplikowane event_id: {ev.loc[dup, 'event_id'].head().tolist()}")
        return {"events": ev, "posts": posts, "episodes": eps}

    @staticmethod
    def summary(ev: pd.DataFrame) -> pd.DataFrame:
        g = ev.groupby("kind")
        return pd.DataFrame({"zdarzenia": g.size(), "od": g["t"].min(), "do": g["t"].max(),
                             "precyzja": g["precision"].first(), "wątek_top": g["thread"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")})
