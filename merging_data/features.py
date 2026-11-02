"""Cechy łączenia: efektywny dzień handlowy, statystyki okienne, wolumeny mediów wokół zdarzeń.

Tylko CZAS, LICZBY i FLAGI POKRYCIA. Żadnej analizy treści (sentyment, ton, ramy narracyjne).
Zasada przyczynowości: okno PO zdarzeniu to (t, t+h], okno PRZED to [t-h, t). Element dokładnie w chwili t
(np. sam tweet) NIE jest liczony jako reakcja.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cleaning_data.horizons import HorizonTools

NY = "America/New_York"
DAY = pd.Timedelta("1D")
GKG_STEP = pd.Timedelta("15min")          # rozdzielczość GKG: krótszych okien mediów nie liczymy
GKG_DENSE_MAX = pd.Timedelta("2h")        # do tej długości okna GKG liczymy ze WSZYSTKICH paczek (także gęstych)
# horyzonty MEDIÓW (osobne od rynkowych cfg.horizons): gęsto w pierwszych godzinach, bo tam rozstrzyga się kolejność
MEDIA_HORIZONS = {"15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "3h": "3h", "1d": "1D", "3d": "3D", "7d": "7D"}


# --------------------------------------------------------------------------- efektywny dzień handlowy
def eff_day(t, cutoff_hour: int = 16) -> pd.DatetimeIndex:
    """Dzień sesji USA, w którym wiadomość może zadziałać: data NY, a od 16:00 NY następny dzień.
    Ten sam klucz jest używany dla świec rynkowych (panel.market_daily), więc tweet i zwrot dnia są spójne."""
    ts = pd.to_datetime(pd.Series(t).reset_index(drop=True), utc=True)
    ny = pd.DatetimeIndex(ts).tz_convert(NY)
    day = ny.normalize().tz_localize(None)
    return day + pd.to_timedelta((ny.hour >= cutoff_hour).astype(int), unit="D")


# --------------------------------------------------------------------------- statystyki okienne
def window_stats(event_t, item_t, horizons: dict, values: dict | None = None, post: bool = True, pre: bool = True) -> pd.DataFrame:
    """Dla każdego zdarzenia: liczba (i sumy wartości) elementów w (t, t+h] oraz w [t-h, t). Kolejność = kolejność zdarzeń."""
    e = HorizonTools._ns(pd.Series(event_t).reset_index(drop=True))
    items = pd.to_datetime(pd.Series(item_t).reset_index(drop=True), utc=True, errors="coerce")
    ok = items.notna().to_numpy()
    it = HorizonTools._ns(items[ok]) if ok.any() else np.array([], dtype="int64")
    order = np.argsort(it, kind="stable")
    it = it[order]
    cs = {k: np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.asarray(v, float)[ok][order]))]) for k, v in (values or {}).items()}
    out: dict = {}
    for lab, h in horizons.items():
        hn = int(pd.Timedelta(h).value)
        if post:
            lo, hi = np.searchsorted(it, e, side="right"), np.searchsorted(it, e + hn, side="right")
            out[f"n_post_{lab}"] = (hi - lo).astype(float)
            for k, c in cs.items():
                out[f"{k}_post_{lab}"] = c[hi] - c[lo]
        if pre:
            lo, hi = np.searchsorted(it, e - hn, side="left"), np.searchsorted(it, e, side="left")
            out[f"n_pre_{lab}"] = (hi - lo).astype(float)
            for k, c in cs.items():
                out[f"{k}_pre_{lab}"] = c[hi] - c[lo]
    return pd.DataFrame(out, index=range(len(e)))


def day_window_stats(event_days, daily: pd.DataFrame, horizons_days: dict) -> pd.DataFrame:
    """Sumy kolumn dziennych w [d, d+n-1] (PO, z dniem zdarzenia) i [d-n, d-1] (PRZED).
    Źródło dzienne (FNSPID) nie rozróżnia kolejności w obrębie dnia zdarzenia. Okno, które nie mieści się
    w zakresie danych, daje NaN (a nie zaniżoną sumę) i flagę fn_cover_* = 0."""
    days = pd.DatetimeIndex(pd.to_datetime(pd.Series(event_days).reset_index(drop=True)))
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    d = daily.reindex(full).fillna(0.0)
    cs = np.vstack([np.zeros(d.shape[1]), np.cumsum(d.to_numpy(float), axis=0)])
    pos = (days - full[0]).days.to_numpy()
    n_all = len(full)
    out: dict = {}
    for lab, n in horizons_days.items():
        pl, ph = np.clip(pos, 0, n_all), np.clip(pos + n, 0, n_all)
        ql, qh = np.clip(pos - n, 0, n_all), np.clip(pos, 0, n_all)
        cov_post = (pos >= 0) & (pos + n <= n_all)
        cov_pre = (pos - n >= 0) & (pos <= n_all)
        for j, c in enumerate(d.columns):
            out[f"{c}_post_{lab}"] = np.where(cov_post, cs[ph, j] - cs[pl, j], np.nan)
            out[f"{c}_pre_{lab}"] = np.where(cov_pre, cs[qh, j] - cs[ql, j], np.nan)
        out[f"fn_cover_post_{lab}"] = cov_post.astype(int)
        out[f"fn_cover_pre_{lab}"] = cov_pre.astype(int)
    return pd.DataFrame(out, index=range(len(days)))


# --------------------------------------------------------------------------- dzienny agregat FNSPID
def fnspid_daily(fn: pd.DataFrame) -> pd.DataFrame:
    """Dzienne liczby: niezależne historie o cłach, wszystkie wiersze, notki zbiorcze, wątki (po tickerze/słowniku tematu)."""
    topic = fn[fn["is_topic"]]
    if "is_first_in_story" in topic:
        first = topic[topic["is_first_in_story"]]
    elif "story_key" in topic:
        first = topic.drop_duplicates("story_key")
    else:
        first = topic
    g = pd.DataFrame({
        "fn_n": first.groupby("day").size(),
        "fn_all_n": fn.groupby("day").size(),
        "fn_sector_n": fn[fn["is_sector_note"]].groupby("day").size(),
    })
    if "thread" in first:
        for th in sorted(x for x in first["thread"].dropna().unique() if str(x).strip()):
            g[f"fn_th_{re.sub(r'[^0-9A-Za-z]+', '_', th).strip('_')}_n"] = (first["thread"] == th).groupby(first["day"]).sum()
    g.index = pd.DatetimeIndex(g.index, name="day")
    return g.fillna(0.0).astype(float).sort_index()


# --------------------------------------------------------------------------- wolumeny mediów wokół zdarzeń
def _gkg_windows(te, gkg: pd.DataFrame, files, hz: dict) -> pd.DataFrame:
    """Artykuły o cłach na paczkę w (t, t+h] i [t-h, t) + liczba paczek w oknie + różnica po - przed."""
    a = window_stats(te, gkg["time_utc"], hz)
    f = window_stats(te, pd.Series(files), hz)
    cols = {}
    for lab in hz:
        for side in ("post", "pre"):
            nf, na = f[f"n_{side}_{lab}"].to_numpy(float), a[f"n_{side}_{lab}"].to_numpy(float)
            cols[f"gkg_files_{side}_{lab}"] = nf
            cols[f"gkg_rate_{side}_{lab}"] = np.where(nf > 0, na / np.where(nf > 0, nf, 1), np.nan)
        both = (cols[f"gkg_files_post_{lab}"] > 0) & (cols[f"gkg_files_pre_{lab}"] > 0)
        cols[f"gkg_rate_delta_{lab}"] = np.where(both, cols[f"gkg_rate_post_{lab}"] - cols[f"gkg_rate_pre_{lab}"], np.nan)
    return pd.DataFrame(cols)


def media_features(events: pd.DataFrame, fn_daily: pd.DataFrame | None, gkg: pd.DataFrame | None,
                   gkg_files: pd.DatetimeIndex | None, posts: pd.DataFrame | None, horizons: dict,
                   gkg_sample: pd.DataFrame | None = None, gkg_files_sample: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Jeden wiersz na zdarzenie (klucz event_id). Tylko liczności i stawki, zawsze z flagami pokrycia:
    - FNSPID: dzienne sumy w oknach >= 1 dnia (fn_*_post_/pre_*, fn_cover_*),
    - Twitter: liczba kolejnych/wcześniejszych tweetów o cłach (tw_n_post_/pre_*), tylko zdarzenia z dokładną godziną,
    - GKG: artykuły o cłach NA PACZKĘ w oknach (gkg_rate_*), z liczbą paczek w oknie (gkg_files_*); okno bez paczki = NaN.
      Okna do 2h liczone ze WSZYSTKICH paczek (także gęstych wokół epizodów), dłuższe tylko z próbki równomiernej
      (gkg_sample, gkg_files_sample): gęste okno kończy się kilka godzin po tweecie, więc w oknie 1d zawyżałoby stawkę.
    horizons: horyzonty mediów (MEDIA_HORIZONS); okna < 15 min są pomijane (paczki GKG co 15 min)."""
    ev = events.reset_index(drop=True)
    out = pd.DataFrame({"event_id": ev["event_id"]})
    exact = (ev["precision"] == "exact").to_numpy()

    hd = {l: int(pd.Timedelta(h).days) for l, h in horizons.items() if pd.Timedelta(h) >= DAY}
    if fn_daily is not None and len(fn_daily) and hd:
        out = pd.concat([out, day_window_stats(ev["eff_day"], fn_daily, hd)], axis=1)

    hi = {l: h for l, h in horizons.items() if pd.Timedelta(h) >= GKG_STEP}

    def exact_only(w: pd.DataFrame) -> pd.DataFrame:
        blank = pd.DataFrame(np.nan, index=range(len(ev)), columns=w.columns)
        blank.loc[exact, :] = w.to_numpy()
        return blank

    if posts is not None and len(posts) and exact.any() and hi:
        out = pd.concat([out, exact_only(window_stats(ev.loc[exact, "t"], posts["time_utc"], hi).add_prefix("tw_"))], axis=1)

    if gkg is not None and len(gkg) and gkg_files is not None and len(gkg_files) and exact.any() and hi:
        te = ev.loc[exact, "t"]
        short = {l: h for l, h in hi.items() if pd.Timedelta(h) <= GKG_DENSE_MAX}
        long_ = {l: h for l, h in hi.items() if pd.Timedelta(h) > GKG_DENSE_MAX}
        gs = gkg if gkg_sample is None else gkg_sample
        fs = gkg_files if gkg_files_sample is None else gkg_files_sample
        parts = []
        if short:
            parts.append(_gkg_windows(te, gkg, gkg_files, short))
        if long_ and gs is not None and fs is not None and len(fs):
            parts.append(_gkg_windows(te, gs, fs, long_))
        if parts:
            out = pd.concat([out, exact_only(pd.concat(parts, axis=1))], axis=1)
    return out
